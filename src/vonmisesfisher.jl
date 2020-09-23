using SpecialFunctions: besseli, logfactorial, loggamma
import Distributions
using Distributions: ContinuousMultivariateDistribution, AbstractRNG

log_besseli_term(order, m, x) = -logfactorial(m) - loggamma(m + order + 1) + (2*m + order)*log(x/2)

isunitvec(v) = isapprox(sum(x^2 for x in v), 1.)

function logsumexp(arr)
    maxarr = maximum(arr)
    return maxarr + log(sum(exp.(arr .- maxarr)))
end

function _log_besseli(order, x; n_terms=100)
    log_terms = Vector{Float64}(undef, n_terms)
    for m in 0:n_terms-1
        log_terms[m+1] = log_besseli_term(order, m, x)
    end
    return logsumexp(log_terms)
end

const max_lookup_input = 50000
const log_besseli_0_lookup = Vector{Float64}(undef, max_lookup_input) # order 0 (used by plane rotation)
const log_besseli_1_lookup = Vector{Float64}(undef, max_lookup_input) # order 1 (used by 3D rotation)

function log_besseli(order, x; n_terms=Int(floor(x/2)) + 20)
    if x > 500.0 && (order == 0 || order == 1)
        if x > max_lookup_input
            error("log besseli not available for x > $max_loopup_input, got: x = $x")
        end
        floorx = Int(floor(x))
        if order == 0
            if !isassigned(log_besseli_0_lookup, floorx)
                log_besseli_0_lookup[floorx] = _log_besseli(0, floorx; n_terms=n_terms)
            end
            return log_besseli_0_lookup[floorx]
        elseif order == 1
            if !isassigned(log_besseli_1_lookup, floorx)
                log_besseli_1_lookup[floorx] = _log_besseli(1, floorx; n_terms=n_terms)
            end
            return log_besseli_1_lookup[floorx]
        else
            @assert false
        end
    else
        return log(besseli(order, x))
    end
end

# TODO: make a PR to Distributions.jl (and SpecialFunctions.jl for log_besseli_1)

####################################
# begin code from Distributions.jl #
####################################

# von Mises-Fisher distribution is useful for directional statistics
#
# The implementation here follows:
#
#   - Wikipedia:
#     http://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution
#
#   - R's movMF package's document:
#     http://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
#
#   - Wenzel Jakob's notes:
#     http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
#

struct VonMisesFisher{T<:Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T
    logCκ::T

    function VonMisesFisher{T}(μ::Vector{T}, κ::T; checknorm::Bool=true) where T
        if checknorm
            isunitvec(μ) || error("μ must be a unit vector")
        end
        κ > 0 || error("κ must be positive.")
        logCκ = vmflck(Base.length(μ), κ)
        S = promote_type(T, typeof(logCκ))
        new{T}(Vector{S}(μ), S(κ), S(logCκ))
    end
end

VonMisesFisher(μ::Vector{T}, κ::T) where {T<:Real} = VonMisesFisher{T}(μ, κ)
VonMisesFisher(μ::Vector{T}, κ::Real) where {T<:Real} = VonMisesFisher(promote_eltype(μ, κ)...)

function VonMisesFisher(θ::Vector)
    κ = norm(θ)
    return VonMisesFisher(θ * (1 / κ), κ)
end

Base.show(io::IO, d::VonMisesFisher) = show(io, d, (:μ, :κ))

### Conversions
convert(::Type{VonMisesFisher{T}}, d::VonMisesFisher) where {T<:Real} = VonMisesFisher{T}(convert(Vector{T}, d.μ), T(d.κ))
convert(::Type{VonMisesFisher{T}}, μ::Vector, κ, logCκ) where {T<:Real} =  VonMisesFisher{T}(convert(Vector{T}, μ), T(κ))



### Basic properties

Distributions.length(d::VonMisesFisher) = Base.length(d.μ)

meandir(d::VonMisesFisher) = d.μ
concentration(d::VonMisesFisher) = d.κ

insupport(d::VonMisesFisher, x::AbstractVector{T}) where {T<:Real} = isunitvec(x)
params(d::VonMisesFisher) = (d.μ, d.κ)
@inline partype(d::VonMisesFisher{T}) where {T<:Real} = T

### Evaluation

function _vmflck(p, κ)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    logbessel_term = log_besseli(q, κ)
    q * log(κ) - hp * log(2π) - logbessel_term
end
_vmflck3(κ) = log(κ) - log(2π) - κ - Distributions.log1mexp(-2κ)
vmflck(p, κ) = (p == 3 ? _vmflck3(κ) : _vmflck(p, κ))

Distributions._logpdf(d::VonMisesFisher, x::AbstractVector{T}) where {T<:Real} = d.logCκ + d.κ * Distributions.dot(d.μ, x)


### Sampling

struct VonMisesFisherSampler
    p::Int          # the dimension
    κ::Float64
    b::Float64
    x0::Float64
    c::Float64
    Q::Matrix{Float64}
end

function VonMisesFisherSampler(μ::Vector{Float64}, κ::Float64)
    p = Base.length(μ)
    b = _vmf_bval(p, κ)
    x0 = (1.0 - b) / (1.0 + b)
    c = κ * x0 + (p - 1) * log1p(-abs2(x0))
    Q = _vmf_rotmat(μ)
    VonMisesFisherSampler(p, κ, b, x0, c, Q)
end

function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler,
                x::AbstractVector, t::AbstractVector)
    w = _vmf_genw(rng, spl)
    p = spl.p
    t[1] = w
    s = 0.0
    for i = 2:p
        t[i] = ti = randn(rng)
        s += abs2(ti)
    end

    # normalize t[2:p]
    r = sqrt((1.0 - abs2(w)) / s)
    for i = 2:p
        t[i] *= r
    end

    # rotate
    Distributions.mul!(x, spl.Q, t)
    return x
end

_rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractVector) =
    _rand!(rng, spl, x, Vector{Float64}(undef, Base.length(x)))

function _rand!(rng::AbstractRNG, spl::VonMisesFisherSampler, x::AbstractMatrix)
    t = Vector{Float64}(undef, size(x, 1))
    for j = 1:size(x, 2)
        _rand!(rng, spl, view(x,:,j), t)
    end
    return x
end


### Core computation

_vmf_bval(p::Int, κ::Real) = (p - 1) / (2.0κ + sqrt(4 * abs2(κ) + abs2(p - 1)))

function _vmf_genw(rng::AbstractRNG, p, b, x0, c, κ)
    # generate the W value -- the key step in simulating vMF
    #
    #   following movMF's document
    #

    r = (p - 1) / 2.0
    betad = Distributions.Beta(r, r)
    z = rand(rng, betad)
    w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    while κ * w + (p - 1) * log(1 - x0 * w) - c < log(rand(rng))
        z = rand(rng, betad)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
    end
    return w::Float64
end

_vmf_genw(rng::AbstractRNG, s::VonMisesFisherSampler) =
    _vmf_genw(rng, s.p, s.b, s.x0, s.c, s.κ)

function _vmf_rotmat(u::Vector{Float64})
    # construct a rotation matrix Q
    # s.t. Q * [1,0,...,0]^T --> u
    #
    # Strategy: construct a full-rank matrix
    # with first column being u, and then
    # perform QR factorization
    #

    p = Base.length(u)
    A = zeros(p, p)
    copyto!(view(A,:,1), u)

    # let k the be index of entry with max abs
    k = 1
    a = abs(u[1])
    for i = 2:p
        @inbounds ai = abs(u[i])
        if ai > a
            k = i
            a = ai
        end
    end

    # other columns of A will be filled with
    # indicator vectors, except the one
    # that activates the k-th entry
    i = 1
    for j = 2:p
        if i == k
            i += 1
        end
        A[i, j] = 1.0
    end

    # perform QR factorization
    Q = Matrix(Distributions.qr!(A).Q)
    if Distributions.dot(view(Q,:,1), u) < 0.0  # the first column was negated
        for i = 1:p
            @inbounds Q[i,1] = -Q[i,1]
        end
    end
    return Q
end

sampler(d::VonMisesFisher) = VonMisesFisherSampler(d.μ, d.κ)

Distributions._rand!(rng::AbstractRNG, d::VonMisesFisher, x::AbstractVector) =
    _rand!(rng, sampler(d), x)
Distributions._rand!(rng::AbstractRNG, d::VonMisesFisher, x::AbstractMatrix) =
    _rand!(rng, sampler(d), x)


##################################
# end code from Distributions.jl #
##################################