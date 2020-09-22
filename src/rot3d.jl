import Gen
import Distributions
import LinearAlgebra#: norm, cross, det
import Rotations

function logsumexp(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

const Rot3 = Rotations.UnitQuaternion{Float64}

total_area(::Type{Rot3}) = pi * pi

@inline function _to_array(x::Rot3)
    v = Rotations.vector(mu)
    return Float64[Rotations.scalar(mu), v[1], v[2], v[3]]
end

export Rot3

################################################
# uniform distribution (Haar measure on SO(3)) #
################################################

struct Uniform3DRotation <: Gen.Distribution{Rot3}

const uniform_3d_rotation = Uniform3DRotation()

function Gen.logpdf(::Uniform3DRotation, x::Rot3)
    # -log(area of 3-sphere times two)
    return -log(total_area(Rot3))
end

function Gen.logpdf_grad(::Uniform3DRotation, x::Rot3)
    error("Not implemented") # TODO implement
    return (nothing,)
end

function Gen.random(::Uniform3DRotation)
    w = normal(0, 1)
    x = normal(0, 1)
    y = normal(0, 1)
    z = normal(0, 1)
    return Rotations.UnitQuaternion(w, x, y, z, normalize=true)
end

Gen.has_output_grad(::Uniform3DRotation) = false

Gen.has_argument_grads(::Uniform3DRotation) = ()

(::Uniform3DRotation)() = Gen.random(Uniform3DRotation())

export uniform_3d_rotation

#######################################
# von Mises Fisher based distribution #
#######################################

# A distribution on SO(3) induced by the Von Mises Fisher distribution on the
# 3-sphere

struct VonMisesFisher3DRotation <: Distribution{Rot3} end

const vmf_3d_rotation = VonMisesFisher3DRotation()

function Gen.logpdf(::VonMisesFisher3DRotation, x::Rot3, mu::Rot3, k::Real)
    dist = Distributions.VonMisesFisher(_to_array(mu), k)
    x_unit_vec = _to_array(x)
    # mixture of anti-podal Von Mises Fisher distributions
    # NOTE: 0.5 is indeed not needed, because of how the base measure is defined
    l1 = Distributions.logpdf(dist, x_unit_vec)
    l2 = Distributions.logpdf(dist, -x_unit_vec)
    return logsumexp(l1, l2)
    #m = max(l1, l2)
    #return m + log(exp(l1 - m) + exp(l2 - m))
end

function logpdf_grad(::VonMisesFisher3DRotation, x::Rot3, mu::Rot3, k::Real)
    error("Not implemented") # TODO implement
    return (nothing,nothing,nothing)
end

function random(::VonMisesFisher3DRotation, mu::Rot3, k::Real)
    dist = Distributions.VonMisesFisher(_to_array(mu), k)
    v = rand(dist)
    return Rotations.UnitQuaternion(v[1], v[2], v[3], v[4])
end

has_output_grad(::VonMisesFisher3DRotation) = false

has_argument_grads(::VonMisesFisher3DRotation) = (false, false)

(::VonMisesFisher3DRotation)(mu, k) = random(VonMisesFisher3DRotation(), mu, k)

export vmf_3d_rotation

###########################################################################
# mixture of von Mises Fisher based distribution and uniform distribution #
###########################################################################

struct UniformVonMisesFisher3DRotation <: Distribution{Rot3} end

const uniform_vmf_3d_rotation = UniformVonMisesFisher3DRotation()

function logpdf(
        ::UniformVonMisesFisher3DRotation, x::Rot3, mu::Rot3,
        k::Real, prob_uniform::Real)
    lp_vmf = logpdf(vmf_3d_rotation, x, mu, k) + log(1.0 - prob_uniform)
    lp_uniform = logpdf(uniform_3d_rotation, x) + log(prob_uniform)
    return logsumexp(lp_vmf, lp_uniform)
end

function logpdf_grad(::UniformVonMisesFisher3DRotation, x::Rot3, mu::Rot3, k::Real)
    error("Not implemented") # TODO implement
    return (nothing, nothing, nothing, nothing)
end

function random(
        ::UniformVonMisesFisher3DRotation, mu::Rot3,
        k::Real, prob_uniform::Real)
    if bernoulli(prob_uniform)
        return uniform_3d_rotation()
    else
        return vmf_3d_rotation(mu, k)
    end
end

has_output_grad(::UniformVonMisesFisher3DRotation) = false

has_argument_grads(::UniformVonMisesFisher3DRotation) = (false, false, false, false)

function (::UniformVonMisesFisher3DRotation)(mu, k, prob_uniform)
    return random(UniformVonMisesFisher3DRotation(), mu, k, prob_uniform)
end

export uniform_vmf_3d_rotation
