# Abstract type for elements (ρ or V, τ, hubbard_n, ...) to converge in the SCF.

struct ScfVariables{NT<:NamedTuple}
    data::NT
end
ScfVariables(; kwargs...) = ScfVariables((; kwargs...))

Base.eltype(x::ScfVariables) = eltype(typeof(flatten(x)))
Base.getproperty(x::ScfVariables, s::Symbol) =
    s === :data ? getfield(x, :data) : getproperty(x.data, s)

Base.propertynames(x::ScfVariables) = propertynames(x.data)
Base.iterate(x::ScfVariables) = iterate(pairs(x.data))
Base.iterate(x::ScfVariables, i) = iterate(pairs(x.data), i)

# Defining field-wise elementary operations on ScfVariables.

function stable_mapfields(f, x::ScfVariables{NT}) where {NT}    # Same dimension
    ScfVariables{NT}(NamedTuple(
        k => f(getproperty(x, k))
        for k in propertynames(x)
    ))
end

function stable_mapfields(f, x::ScfVariables{NT}, y::ScfVariables{NT}) where {NT}
    ScfVariables{NT}(NamedTuple(
        k => f(getproperty(x, k), getproperty(y, k))
        for k in propertynames(x)
    ))
end

function mapfields(f, x::ScfVariables)
    ScfVariables(NamedTuple(
        k => f(getproperty(x, k))
        for k in propertynames(x)
    ))
end

function mapfields(f, x::ScfVariables, y::ScfVariables)
    ScfVariables(NamedTuple(
        k => f(getproperty(x, k), getproperty(y, k))
        for k in propertynames(x)
    ))
end

Base.:+(x::ScfVariables, y::ScfVariables) = stable_mapfields(+, x, y)
Base.:-(x::ScfVariables, y::ScfVariables) = stable_mapfields(-, x, y)
Base.:*(α::Number, x::ScfVariables) = stable_mapfields(v -> α * v, x)
Base.:/(x::ScfVariables, α::Number) = stable_mapfields(v -> v / α, x)
Base.:-(x::ScfVariables) = stable_mapfields(-, x)
Base.broadcastable(x::ScfVariables) = Ref(x)

LinearAlgebra.dot(x::ScfVariables, y::ScfVariables) = mapfields(dot, x, y)
LinearAlgebra.norm(x::ScfVariables) = mapfields(norm, x)
Base.size(x::ScfVariables) = mapfields(size, x)

# Flattening and reconstructing ScfVariables structures to and from 1 dimensional vectors.

flatten(x::Nothing) = []
function _reconstruct(::Nothing, v, i)
    return nothing, i
end
flatten(x::Real) = [x]
function _reconstruct(::T, v, i) where {T<:Real}
    return T(v[i]), i + 1
end
flatten(x::Complex) = [real(x), imag(x)]
function _reconstruct(::T, v, i) where {T<:Complex}
    x = T(v[i] + v[i+1] * im)
    return x, i + 2
end
flatten(x::AbstractArray) = reduce(vcat, flatten.(x))   # reduce(vcat, ) not optimal numerically TODO
function _reconstruct(template::AbstractArray{T,N}, v, i) where {T,N}
    n = length(template)

    data = Vector{T}(undef, n)

    for j = 1:n
        data[j], i = _reconstruct(template[j], v, i)
    end

    return reshape(data, size(template)), i
end
flatten(x::Tuple) = reduce(vcat, flatten.(x))
function _reconstruct(template::Tuple, v, i)
    vals = ()

    for x in template
        y, i = _reconstruct(x, v, i)
        vals = (vals..., y)
    end

    return vals, i
end
flatten(x::NamedTuple) = reduce(vcat, flatten(values(x)))
function _reconstruct(template::NamedTuple, v, i)
    vals = NamedTuple()

    data = NamedTuple(
        k => begin
            y, i = _reconstruct(getfield(template, k), v, i)
            y
        end
        for k in keys(template)
    )

    return data, i
end
flatten(x::ScfVariables) = flatten(x.data)
function _reconstruct(template::ScfVariables, v, i)
    data, i = _reconstruct(template.data, v, i)
    return ScfVariables(data), i
end

function reconstruct(template, v::AbstractVector)
    x, i = _reconstruct(template, v, 1)
    @assert i == length(v) + 1
    return x
end

# Adapting mixing functions to ScfVariables

mix_default(mixing, basis, Δx; kwargs...) = Δx
# Default fallbacks
mix_density(mixing, basis, Δx; kwargs...) = Δx
mix_potential(mixing, basis, Δx; kwargs...) = Δx
mix_hubbard_n(mixing, basis, Δx; kwargs...) = Δx
function get_mix_function(s::Symbol)
    if s == :ρ
        return mix_density
    elseif s == :V
        return mix_potential
    elseif s == :hubbard_n
        return mix_hubbard_n
    else
        return mix_default
    end
end

"""
Apply mixing scheme to the ScfVariables object.
"""
function mix_variables(mixing, basis, Δx::ScfVariables{NT}; kwargs...) where {NT}
    ScfVariables{NT}(NamedTuple(
        k => get_mix_function(k)(mixing, basis, getproperty(Δx, k);  kwargs...)
        for k in propertynames(Δx)
    ))
end

"""
Construct an appropriate ScfVariables object from a basis and guess density (SCF on density).
"""
function ScfVariables(basis::PlaneWaveBasis{T}, ρ) where {T}
    data = (; ρ)
    if any(needs_τ, basis.terms)
        data = merge(data, (; τ=zero(ρ)))
    end
    ihubbard = findfirst(t -> t isa TermHubbard, basis.terms)
    if !isnothing(ihubbard)
        data = merge(data, (; hubbard_n=compute_hubbard_n(basis.terms[ihubbard], basis, nothing, nothing)))
    end
    ScfVariables(; data...)
end