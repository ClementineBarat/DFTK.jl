include("scf_callbacks.jl")
using Dates

"""
Transparently handle checkpointing by either returning kwargs for `self_consistent_field`,
which start checkpointing (if no checkpoint file is present) or that continue a checkpointed
run (if a checkpoint file can be loaded). `filename` is the location where the checkpoint
is saved, `save_ψ` determines whether orbitals are saved in the checkpoint as well.
The latter is discouraged, since generally slow.
See [Saving SCF results on disk and SCF checkpoints](@ref) for details how to use
this function in practice.
"""
function kwargs_scf_checkpoints(basis::AbstractBasis;
                                filename="dftk_scf_checkpoint.jld2",
                                callback=ScfDefaultCallback(),
                                diagtolalg::AdaptiveDiagtol=AdaptiveDiagtol(),
                                ρ=guess_density(basis),
                                τ=any(needs_τ, basis.terms) ? zero(ρ) : nothing,
                                hubbard_n=nothing, ψ=nothing, occupation=nothing,
                                save_ψ=false, kwargs...)
    if isfile(filename)
        # Disable strict checking, since we can live with only the density data
        previous = load_scfres(filename, basis; skip_hamiltonian=true, strict=false)

        # If we can expect the guess to be good, tighten the diagtol.
        if !isnothing(previous.ρ)
            ρ = previous.ρ
            τ = previous.τ
            hubbard_n = previous.hubbard_n
            if hasproperty(previous, :eigenvalues) && hasproperty(previous, :history_Δρ)
                diagtol_first = determine_diagtol(diagtolalg, previous)
            else
                diagtol_first = diagtolalg.diagtol_max
            end
            diagtolalg = AdaptiveDiagtol(; diagtol_first,
                                           diagtolalg.diagtol_max,
                                           diagtolalg.diagtol_min,
                                           diagtolalg.ratio_ρdiff)
        end
        occupation = something(previous.occupation, Some(occupation))
        ψ = something(previous.ψ, Some(ψ))
    end

    callback = callback ∘ ScfSaveCheckpoints(; filename, save_ψ)
    (; callback, diagtolalg, ψ, ρ, τ, hubbard_n, occupation, kwargs...)
end

# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = true
end

compute_τ(info) = compute_kinetic_energy_density(info.basis, info.ψ, info.occupation)
function compute_hubbard_n(info)
    ihubbard = findfirst(t -> t isa TermHubbard, info.basis.terms)
    if isnothing(ihubbard)
        return nothing
    else
        return compute_hubbard_n(info.basis.terms[ihubbard], info.basis, 
                                 info.ψ, info.occupation)
    end
end

"""
Update the SCF variables (based on their names) from the information gathered in `info`.
"""
function update_energies_variables(energies, x_in::ScfVariables, info; compute_consistent_energies=false, kwargs...)
    
    ρ = nothing
    V = nothing
    τ = nothing
    hubbard_n = nothing

    if !isnothing(x_in.ρ)
        ρ = info.ρout
    end
    if !isnothing(x_in.τ)
        τ = compute_τ(info)
    end
    if !isnothing(x_in.hubbard_n)
        hubbard_n = compute_hubbard_n(info)
    end
    # The potential needs to be updated at the end as it might need 'τ', 'hubbard_n', ...
    if !isnothing(x_in.V)
        energies, new_ham = energy_hamiltonian(basis, info.ψ, info.occupation;
                              eigenvalues=info.eigenvalues, εF=info.εF,
                              ρ=info.ρout, τ, hubbard_n, kwargs...)
        V = total_local_potential(new_ham)
    elseif compute_consistent_energies
        (; energies) = energy(basis, info.ψ, info.occupation; 
                              ρ=info.ρout, τ, hubbard_n, info_next.eigenvalues, 
                              info_next.εF, nbandsalg.occupation_threshold)
    end

    x_out = ScfVariables(; ρ, V, τ, hubbard_n)
    return energies, x_out

end

"""
Update the new Hamiltonian `ham`, either from the density (SCF on the density) 
or from the potential (SCF on the potential).
"""
function update_ham(basis, info, x::ScfVariables; kwargs...)
    if !isnothing(x.ρ)  # SCF on density
        energies, ham = energy_hamiltonian(basis, info.ψ, info.occupation;
                                           info.eigenvalues, info.εF, x..., kwargs...)
        return energies, ham
    elseif !isnothing(x.V)  # SCF on potential
        ham = hamiltonian_with_total_potential(info.ham, x.V)
        return nothing, ham
    end
end

"""
Obtain new density ρ by diagonalizing `ham`. Follows the policy imposed by the `bands`
data structure to determine and adjust the number of bands to be computed.
"""
function next_density(ham::Hamiltonian,
                      nbandsalg::NbandsAlgorithm=AdaptiveBands(ham.basis.model),
                      fermialg::AbstractFermiAlgorithm=default_fermialg(ham.basis.model);
                      eigensolver=lobpcg_hyper, ψ=nothing, eigenvalues=nothing,
                      occupation=nothing, kwargs...)
    n_bands_converge, n_bands_compute = determine_n_bands(nbandsalg, occupation,
                                                          eigenvalues, ψ)

    if isnothing(ψ)
        increased_n_bands = true
    else
        @assert length(ψ) == length(ham.basis.kpoints)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
        increased_n_bands = n_bands_compute > size(ψ[1], 2)
    end

    # TODO Synchronize since right now it is assumed that the same number of bands are
    #      computed for each k-Point
    n_bands_compute = mpi_max(n_bands_compute, ham.basis.comm_kpts)

    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute;
                                     ψguess=ψ, n_conv_check=n_bands_converge, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" n_iter=eigres.n_iter)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham.basis, eigres.λ, fermialg;
                                        tol_n_elec=nbandsalg.occupation_threshold)
    minocc = maximum(minimum, occupation)

    # TODO This is a bit hackish, but needed right now as we increase the number of bands
    #      to be computed only between SCF steps. Should be revisited once we have a better
    #      way to deal with such things in LOBPCG.
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    ρout = compute_density(ham.basis, eigres.X, occupation; nbandsalg.occupation_threshold)
    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold,
     n_matvec=mpi_sum(eigres.n_matvec, ham.basis.comm_kpts))
end


@doc raw"""
    self_consistent_field(basis; [tol, mixing, damping, ρ, ψ])

Solve the Kohn-Sham equations with a density-based SCF algorithm using damped, preconditioned
iterations where ``ρ_\text{next} = ρ_\text{in} + α P^{-1} (ρ_\text{out} - ρ_\text{in})``.

Overview of parameters:
- `ρ`:   Initial density
- `ψ`:   Initial orbitals
- `tol`: Tolerance for the density change (``\|ρ_\text{out} - ρ_\text{in}\|``)
  to flag convergence. Default is `1e-6`.
- `is_converged`: Convergence control callback. Typical objects passed here are
  `ScfConvergenceDensity(tol)` (the default), `ScfConvergenceEnergy(tol)` or `ScfConvergenceForce(tol)`.
- `miniter`: Minimal number of SCF iterations
- `maxiter`: Maximal number of SCF iterations
- `maxtime`: Maximal time to run the SCF for. If this is reached without
   convergence, the SCF stops.
- `mixing`: Mixing method, which determines the preconditioner ``P^{-1}`` in the above equation.
  Typical mixings are [`LdosMixing`](@ref), [`KerkerMixing`](@ref), [`SimpleMixing`](@ref)
  or [`DielectricMixing`](@ref). Default is `LdosMixing()`
- `damping`: Damping parameter ``α`` in the above equation. Default is `0.8`.
- `solver`: Fixed-point solver, the default is `scf_anderson_solver()`.
- `iterate_on`: Chooses wether the SCF is done on the density (`:density`) 
                or on the potential (`:potential`). Default is `:density`.
- `nbandsalg`: By default DFTK uses `nbandsalg=AdaptiveBands(model)`, which adaptively determines
  the number of bands to compute. If you want to influence this algorithm or use a predefined
  number of bands in each SCF step, pass a [`FixedBands`](@ref) or [`AdaptiveBands`](@ref).
  Beware that with non-zero temperature, the convergence of the SCF algorithm may be limited
  by the `default_occupation_threshold()` parameter. For highly accurate calculations we thus
  recommend increasing the `occupation_threshold` of the `AdaptiveBands`.
- `callback`: Function called at each SCF iteration. Usually takes care of printing the
  intermediate state.
"""
function self_consistent_field(
    basis::PlaneWaveBasis{T};
    ρ=guess_density(basis),
    ψ=nothing,
    occupation=nothing,
    eigenvalues=nothing,
    tol=1e-6,
    is_converged=ScfConvergenceDensity(tol),
    miniter=0,
    maxiter=100,
    maxtime=Year(1),
    mixing=LdosMixing(),
    damping=0.8,
    solver=scf_anderson_solver(),
    iterate_on=:density,
    eigensolver=lobpcg_hyper,
    diagtolalg=default_diagtolalg(basis; tol),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    exxalg::ExxAlgorithm=AceExx(),
    callback=ScfDefaultCallback(; show_damping=false),
    compute_consistent_energies=true,
    seed=nothing,
    response=ResponseOptions(),  # Dummy here, only for AD
) where {T}

    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    start_ns = time_ns()
    timeout_date = Dates.now() + maxtime
    seed = seed_task_local_rng!(seed, basis.comm_kpts)

    function fixpoint_map(x_in, info)

        n_iter = info.n_iter
        n_iter += 1

        # Define the new Hamiltonian
        energies, ham = update_ham(basis, info, x_in; nbandsalg.occupation_threshold)
    
        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; 
                                 eigensolver, info.ψ, info.eigenvalues, info.occupation, 
                                 miniter=1, tol=determine_diagtol(diagtolalg, info))
        # Update info with results gathered so far
        info_next = (; ham, basis, stage=:iterate, algorithm="SCF",
                       α=damping, n_iter, nbandsalg.occupation_threshold,
                       seed, runtime_ns=time_ns() - start_ns, ρin=x_in.ρ, nextstate...,
                       diagonalization=[nextstate.diagonalization])
        
        # Update the energies and the SCF variables with info_next
        energies, x_out = update_energies_variables(energies, x_in, info_next; 
                                                    compute_consistent_energies, 
                                                    nbandsalg.occupation_threshold)
        Δx = x_out - x_in

        # Update the history in info_next
        history_Etot = vcat(info.history_Etot, energies.total)
        history_Δρ = info.history_Δρ
        if isnothing(x_in.ρ)
            history_Δρ = vcat(history_Δρ, norm(info_next.ρout - info.ρout) * sqrt(basis.dvol))
        else
            history_Δρ = vcat(history_Δρ, norm(Δx.ρ) * sqrt(basis.dvol))
        end
        n_matvec = info.n_matvec + nextstate.n_matvec
        info_next = merge(info_next, (; energies, history_Etot, n_matvec, history_Δρ))

        # Apply mixing to the SCF variables
        x_next = x_in .+ T(damping) .* mix_variables(mixing, basis, Δx; info_next...)

        converged = n_iter ≥ miniter && is_converged(info_next)
        converged = mpi_bcast(converged, 0, basis.comm_kpts)
        info_next = merge(info_next, (; converged))

        timedout = mpi_bcast(Dates.now() ≥ timeout_date, basis.comm_kpts)
        info_next = merge(info_next, (; timedout))

        callback(info_next)

        x_next, info_next
    end

    # Note: it is assumed that, upon entry, the input density ρ is numerically identical
    #       across all MPI ranks. If not, unexpected behavior may occur. It is the caller's
    #       responsibility to ensure this is the case.
    energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ)
    info_init = (; ham, energies, ρin=ρ, ρout=ρ, ψ, occupation, eigenvalues, εF=nothing,
                   n_iter=0, n_matvec=0, timedout=false, converged=false,
                   history_Etot=T[], history_Δρ=T[])

    # Convergence is flagged by is_converged inside the fixpoint_map.
    x = ScfVariables(basis, ρ; iterate_on, ham)
    _, info = solver(fixpoint_map, x, info_init; maxiter)

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map.
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy and to build a Hamiltonian without any compression
    # applied to the exchange operator.
    (; ψ, occupation, eigenvalues, εF, converged) = info
    ρ = info.ρout
    τ = compute_τ(info)
    hubbard_n = compute_hubbard_n(info)
    energies, ham = energy_hamiltonian(basis, ψ, occupation; 
                                       exxalg=VanillaExx(),
                                       eigenvalues, εF, ρ, τ, hubbard_n,
                                       nbandsalg.occupation_threshold)

    # Callback is run one last time with final state to allow callback to clean up
    scfres = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
                ρ, τ, hubbard_n, α=damping, eigenvalues, occupation, εF,
                info.n_bands_converge, info.n_iter, info.n_matvec, ψ, info.diagonalization, 
                stage=:finalize, info.history_Δρ, info.history_Etot, info.timedout, mixing, 
                is_converged, nbandsalg, fermialg, diagtolalg, solver, eigensolver, seed, 
                runtime_ns=time_ns() - start_ns, algorithm="SCF")
    callback(scfres)
    scfres
end
