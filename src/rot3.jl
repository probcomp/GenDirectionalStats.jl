import Gen
import Distributions
import Rotations

const Rot3 = Rotations.UnitQuaternion{Float64}

# area of 3-sphere times two
total_area(::Type{Rot3}) = pi * pi

@inline function _to_array(x::Rot3)
    v = Rotations.vector(x)
    return Float64[Rotations.scalar(x), v[1], v[2], v[3]]
end

export Rot3

################################################
# uniform distribution (Haar measure on SO(3)) #
################################################

struct UniformRot3 <: Gen.Distribution{Rot3} end

const uniform_rot3 = UniformRot3()

function Gen.logpdf(::UniformRot3, x::Rot3)
    return -log(total_area(Rot3))
end

function Gen.logpdf_grad(::UniformRot3, x::Rot3)
    error("Not implemented") # TODO implement
    return (nothing,)
end

function Gen.random(::UniformRot3)
    w = randn()
    x = randn()
    y = randn()
    z = randn()
    return Rotations.UnitQuaternion{Float64}(w, x, y, z, true)
end

Gen.has_output_grad(::UniformRot3) = false

Gen.has_argument_grads(::UniformRot3) = ()

(::UniformRot3)() = Gen.random(UniformRot3())

export uniform_rot3

################################################
# von Mises Fisher based distribution on SO(3) #
################################################

struct VMFRot3 <: Gen.Distribution{Rot3} end

const vmf_rot3 = VMFRot3()

function Gen.logpdf(::VMFRot3, x::Rot3, mu::Rot3, k::Real)
    dist = VonMisesFisher(_to_array(mu), k)
    x_unit_vec = _to_array(x)
    # mixture of anti-podal Von Mises Fisher distributions
    # NOTE: 0.5 is indeed not needed, because of how the base measure is defined
    l1 = Distributions.logpdf(dist, x_unit_vec)
    l2 = Distributions.logpdf(dist, -x_unit_vec)
    return _logsumexp(l1, l2)
end

function Gen.logpdf_grad(::VMFRot3, x::Rot3, mu::Rot3, k::Real)
    error("Not implemented") # TODO implement
    return (nothing,nothing,nothing)
end

function Gen.random(::VMFRot3, mu::Rot3, k::Real)
    dist = VonMisesFisher(_to_array(mu), k)
    v = rand(dist)
    return Rotations.UnitQuaternion(v[1], v[2], v[3], v[4])
end

Gen.has_output_grad(::VMFRot3) = false

Gen.has_argument_grads(::VMFRot3) = (false, false)

(::VMFRot3)(mu, k) = Gen.random(VMFRot3(), mu, k)

export vmf_rot3

###########################################################################
# mixture of von Mises Fisher based distribution and uniform distribution #
###########################################################################

struct UniformVMFRot3 <: Gen.Distribution{Rot3} end

const uniform_vmf_rot3 = UniformVMFRot3()

function Gen.logpdf(
        ::UniformVMFRot3, x::Rot3, mu::Rot3,
        k::Real, prob_uniform::Real)
    lp_vmf = logpdf(vmf_rot3, x, mu, k) + log(1.0 - prob_uniform)
    lp_uniform = logpdf(uniform_rot3, x) + log(prob_uniform)
    return _logsumexp(lp_vmf, lp_uniform)
end

function Gen.logpdf_grad(::UniformVMFRot3, x::Rot3, mu::Rot3, k::Real)
    error("Not implemented") # TODO implement
    return (nothing, nothing, nothing, nothing)
end

function Gen.random(
        ::UniformVMFRot3, mu::Rot3,
        k::Real, prob_uniform::Real)
    if bernoulli(prob_uniform)
        return uniform_rot3()
    else
        return vmf_rot3(mu, k)
    end
end

Gen.has_output_grad(::UniformVMFRot3) = false

Gen.has_argument_grads(::UniformVMFRot3) = (false, false, false, false)

function (::UniformVMFRot3)(mu, k, prob_uniform)
    return Gen.random(UniformVMFRot3(), mu, k, prob_uniform)
end

export uniform_vmf_rot3
