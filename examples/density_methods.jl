# # Density Guesses
#
# We compare four different approaches to charge density initialization for starting a
# density-based SCF.

# First we set up our problem

using AtomsBuilder
using DFTK
using LinearAlgebra
using Printf
using PseudoPotentialData

# We use a numeric norm-conserving PSP in UPF format from the
# [PseudoDojo](http://www.pseudo-dojo.org/) v0.4 scalar-relativistic LDA standard stringency
# family because it contains valence charge density which can be used for a more tailored
# density guess.
pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")

function silicon_scf(guess_method)
    model = model_DFT(bulk(:Si); functionals=LDA(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
    self_consistent_field(basis; ρ=guess_density(basis, guess_method),
                          is_converged=ScfConvergenceDensity(1e-6))
end;

results = Dict();

# ## Random guess
# The random density is normalized to the number of electrons provided.
results["Random"] = silicon_scf(RandomDensity());

# ## Superposition of Gaussian densities
# The Gaussians are defined by a tabulated atom decay length.
results["Gaussian"] = silicon_scf(ValenceDensityGaussian());

# ## Pseudopotential density guess
# If you'd like to force the use of pseudopotential atomic densities, you can use
# the `ValenceDensityPseudo()` method. Note that this will fail if any of the
# pseudopotentials don't provide atomic valence charge densities!
results["Pseudo"] = silicon_scf(ValenceDensityPseudo());

# ## Automatic density guess
# This method will automatically use valence charge densities from from pseudopotentials
# that provide them and Gaussian densities for elements which do not have pseudopotentials
# or whose pseudopotentials don't provide valence charge densities.
results["Auto"] = silicon_scf(ValenceDensityAuto());

@printf "%10s %6s\n" "Method" "n_iter"
for (key, scfres) in results
    @printf "%10s %6d\n" key scfres.n_iter
end
