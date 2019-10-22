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
    d = Distmuibutions.VonMisesFishemu([mu.w, mu.x, mu.y, mu.z], k)
    logpdf(d, [r.w, r.x, r.y, r.z])
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

