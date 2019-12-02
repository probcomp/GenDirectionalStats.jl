using Gen: Distribution
using Geometry: UnitQuaternion
import Gen: logpdf, logpdf_grad, random, has_output_grad, has_argument_grads
import Distributions

########################################
# uniform distribution on 3D rotations #
########################################

struct Uniform3DRotation <: Distribution{UnitQuaternion} end

const uniform_3d_rotation = Uniform3DRotation()

function logpdf(::Uniform3DRotation, r::UnitQuaternion)
    # -log(area of 3 sphere)
    -log(2 * pi * pi)
end

function logpdf_grad(::Uniform3DRotation, r::UnitQuaternion)
    error("Not implemented")
    (nothing,)
end

function random(::Uniform3DRotation)
    w = normal(0, 1)
    x = normal(0, 1)
    y = normal(0, 1)
    z = normal(0, 1)
    n = norm([w, x, y, z])
    UnitQuaternion(w/n, x/n, y/n, z/n)
end

has_output_grad(::Uniform3DRotation) = false

has_argument_grads(::Uniform3DRotation) = ()

(::Uniform3DRotation)() = random(Uniform3DRotation())

export Uniform3DRotation, uniform_3d_rotation

#################################################
# Von Mises Fisher distribution on 3D rotations #
#################################################

# note: this is the Von Mises Fisher distribution on the 3-sphere

struct VonMisesFisher3DRotation <: Distribution{UnitQuaternion} end

const vmf_3d_rotation = VonMisesFisher3DRotation()

function logpdf(::VonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    d = Distributions.VonMisesFisher([mu.w, mu.x, mu.y, mu.z], k)
    Distributions.logpdf(d, [r.w, r.x, r.y, r.z])
end

function logpdf_grad(::VonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    error("Not implemented")
    (nothing,)
end

function random(::VonMisesFisher3DRotation, mu::UnitQuaternion, k::Float64)
    d = Distributions.VonMisesFisher([mu.w, mu.x, mu.y, mu.z], k)
    v = rand(d)
    UnitQuaternion(v[1], v[2], v[3], v[4])
end

has_output_grad(::VonMisesFisher3DRotation) = false

has_argument_grads(::VonMisesFisher3DRotation) = (false, false)

(::VonMisesFisher3DRotation)(mu, k) = random(VonMisesFisher3DRotation(), mu, k)

export VonMisesFisher3DRotation, vmf_3d_rotation

###########################################
# mixture of VMF and uniform distribution #
###########################################

struct UniformVonMisesFisher3DRotation <: Distribution{UnitQuaternion} end

const uniform_vmf_3d_rotation = UniformVonMisesFisher3DRotation()

function logpdf(
        ::UniformVonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion,
        k::Float64, prob_uniform::Float64)
    lp_vmf = logpdf(vmf_3d_rotation, r, mu, k) + log(1. - prob_uniform)
    lp_uniform = logpdf(uniform_3d_rotation, r) + log(prob_uniform)
    logsumexp(lp_vmf, lp_uniform)
end

function logpdf_grad(::UniformVonMisesFisher3DRotation, r::UnitQuaternion, mu::UnitQuaternion, k::Float64)
    error("Not implemented")
    (nothing, nothing, nothing, nothing)
end

function random(
        ::UniformVonMisesFisher3DRotation, mu::UnitQuaternion,
        k::Float64, prob_uniform::Float64)
    if bernoulli(prob_uniform)
        uniform_3d_rotation()
    else
        vmf_3d_rotation(mu, k)
    end
end

has_output_grad(::UniformVonMisesFisher3DRotation) = false

has_argument_grads(::UniformVonMisesFisher3DRotation) = (false, false, false, false)

(::UniformVonMisesFisher3DRotation)(mu, k, prob_uniform) = random(UniformVonMisesFisher3DRotation(), mu, k, prob_uniform)

export UniformVonMisesFisher3DRotation, uniform_vmf_3d_rotation

