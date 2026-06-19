# Abstract type for elements (ρ or V, τ, hubbard_n, ...) to converge in the SCF.

@kwdef struct ScfVariables
    ρ = nothing
    V = nothing
    τ = nothing
    hubbard_n = nothing
end

"""Return the field names whose value is not `nothing`."""
active_fields(x::ScfVariables) = filter(k -> !isnothing(getfield(x, k)), fieldnames(ScfVariables))

Base.eltype(x::ScfVariables) = eltype(typeof(flatten(x)))

"""
    mapfields(f, x)  /  mapfields(f, x, y)

Apply `f` field-by-field, skipping `nothing` fields (they stay `nothing`).
The two-argument form requires both arguments to share the same active fields.
"""
function mapfields(f, x::ScfVariables)
    ScfVariables(; (k => f(getfield(x, k)) for k in active_fields(x))...)
end

function mapfields(f, x::ScfVariables, y::ScfVariables)
    kx, ky = active_fields(x), active_fields(y)
    @assert kx == ky "ScfVariables operands have different active fields: $kx vs $ky"
    ScfVariables(; (k => f(getfield(x, k), getfield(y, k)) for k in kx)...)
end

Base.:+(x::ScfVariables, y::ScfVariables)  = mapfields(+, x, y)
Base.:-(x::ScfVariables, y::ScfVariables)  = mapfields(-, x, y)
Base.:*(α::Number,        x::ScfVariables) = mapfields(v -> α*v, x)
Base.:/(x::ScfVariables,  α::Number)       = mapfields(v -> v/α, x)
Base.:-(x::ScfVariables)                   = mapfields(-, x)
Base.broadcastable(x::ScfVariables)        = Ref(x)

LinearAlgebra.dot(x::ScfVariables,  y::ScfVariables) = mapfields(dot, x, y)
LinearAlgebra.norm(x::ScfVariables)                  = mapfields(norm, x)
Base.size(x::ScfVariables)                           = mapfields(size, x)

# Flattening and reconstructing ScfVariables structures to and from 1 dimensional vectors.

# Leaf types
flatten(::Nothing)                      = Float64[]
flatten(x::Real)                        = [x]
flatten(x::Complex)                     = [real(x), imag(x)]
flatten(x::AbstractArray)               = reduce(vcat, flatten.(x))
flatten(x::Tuple)                       = reduce(vcat, flatten.(x))
flatten(x::NamedTuple)                  = reduce(vcat, flatten.(values(x)))

"""Flatten only the active (non-nothing) fields into a single Vector."""
function flatten(x::ScfVariables)
    isempty(active_fields(x)) && return Float64[]
    reduce(vcat, (flatten(getfield(x, k)) for k in active_fields(x)))
end

# Reconstruct leaf types
_reconstruct(::Nothing, v, i)              = nothing, i
_reconstruct(::T, v, i) where {T<:Real}    = T(v[i]), i+1
function _reconstruct(::T, v, i) where {T<:Complex}
    x = T(v[i] + v[i+1]*im)
    return x, i+2
end
function _reconstruct(template::AbstractArray{T,N}, v, i) where {T,N}
    n    = length(template)
    data = Vector{T}(undef, n)
    for j = 1:n
        data[j], i = _reconstruct(template[j], v, i)
    end
    return reshape(data, size(template)), i
end
function _reconstruct(template::Tuple, v, i)
    vals = ()
    for x in template
        y, i  = _reconstruct(x, v, i)
        vals  = (vals..., y)
    end
    return vals, i
end
function _reconstruct(template::NamedTuple, v, i)
    pairs = map(keys(template)) do k
        y, i = _reconstruct(template[k], v, i)
        k => y
    end
    return NamedTuple(pairs), i
end

"""Reconstruct a ScfVariables from a flat vector, using `template` for shapes."""
function _reconstruct(template::ScfVariables, v, i)
    fields = fieldnames(ScfVariables)
    kwargs = map(fields) do k
        val   = getfield(template, k)
        y, i  = _reconstruct(val, v, i)   # nothing → nothing, skips indices
        k => y
    end
    return ScfVariables(; kwargs...), i
end

function reconstruct(template, v::AbstractVector)
    x, i = _reconstruct(template, v, 1)
    @assert i == length(v) + 1
    return x
end

# Adapting mixing functions to ScfVariables

# Default fallbacks
mix_default(mixing, basis, Δx; kwargs...) = Δx
mix_density(mixing, basis, Δx; kwargs...) = Δx
mix_potential(mixing, basis, Δx; kwargs...) = Δx
mix_hubbard_n(mixing, basis, Δx; kwargs...) = Δx

"""
Apply mixing scheme to the ScfVariables object.
"""
function mix_variables(mixing, basis, Δx::ScfVariables{NT}; kwargs...) where {NT}
    ScfVariables(;
        ρ = mix_density(mixing, basis, Δx.ρ, kwargs...),
        V = mix_potential(mixing, basis, Δx.V, kwargs...),
        τ = mix_default(mixing, basis, Δx.τ, kwargs...),
        hubbard_n = mix_hubbard_n(mixing, basis, Δx.hubbard_n, kwargs...),
    )
end

"""
Construct an appropriate ScfVariables object from a basis and guess density (SCF on density).
"""
function ScfVariables(basis::PlaneWaveBasis{T}, ρ; iterate_on=:density, ham=nothing) where {T}
    V = nothing
    τ = nothing
    hubbard_n = nothing
    if any(needs_τ, basis.terms)
        τ=zero(ρ)
    end
    ihubbard = findfirst(t -> t isa TermHubbard, basis.terms)
    if !isnothing(ihubbard)
        hubbard_n=compute_hubbard_n(basis.terms[ihubbard], basis, nothing, nothing)
    end
    if iterate_on == :density
        return ScfVariables(; ρ, τ, hubbard_n)
    else
        if isnothing(ham)
            _, ham = energy_hamiltonian(basis, nothing, nothing; ρ)
        end
        V = total_local_potential(ham)
        return ScfVariables(; V, τ, hubbard_n)
    end
end
