import Gen
import Distributions
import LinearAlgebra#: norm, cross, det
import Rotations
import StaticArrays

const Rot2 = Rotations.RotMatrix{2,Float64,4}

function Base.rand(::Type{Rot2})
    a = Gen.uniform(-pi, pi)
    return Rotations.RotMatrix{2,Float64}(a)
end

total_area(::Type{Rot2}) = 2 * pi

export Rot2

################################################
# uniform distribution (Haar measure on SO(2)) #
################################################

struct UniformRot2 <: Gen.Distribution{Rot2} end

Gen.logpdf(::UniformRot2, x::Rot2) = -log(total_area(Rot2))

function Gen.logpdf_grad(::UniformRot2, x::Rot2)
    error("Not implemented") # TODO implement
    return (nothing,)
end

Gen.random(::UniformRot2) = rand(Rot2)

Gen.has_output_grad(::UniformRot2) = false

Gen.has_argument_grads(::UniformRot2) = ()

(::UniformRot2)() = Gen.random(UniformRot2())

const uniform_rot2 = UniformRot2()

export uniform_rot2

################################################
# von Mises Fisher based distribution on SO(2) #
################################################

struct VMFRot2 <: Gen.Distribution{Rot2} end

const vmf_rot2 = VMFRot2()

function Gen.logpdf(::VMFRot2, x::Rot2, mu::Rot2, k::Real)
    dist = Distributions.VonMisesFisher(_to_array(mu), k)
    x_unit_vec = _to_array(x)
    # mixture of anti-podal Von Mises Fisher distributions
    # NOTE: 0.5 is indeed not needed, because of how the base measure is defined
    l1 = Distributions.logpdf(dist, x_unit_vec)
    l2 = Distributions.logpdf(dist, -x_unit_vec)
    return _logsumexp(l1, l2)
end

function Gen.logpdf_grad(::VMFRot2, x::Rot2, mu::Rot2, k::Real)
    error("Not implemented") # TODO implement
    return (nothing,nothing,nothing)
end

function Gen.random(::VMFRot2, mu::Rot2, k::Real)
    dist = Distributions.VonMisesFisher(_to_array(mu), k)
    v = rand(dist)
    mat = StaticArrays.SMatrix{2,2,Float64,4}(v[1], v[2], -v[2], v[1])
    return Rotations.RotMatrix{2,Float64,4}(mat)
end

Gen.has_output_grad(::VMFRot2) = false

Gen.has_argument_grads(::VMFRot2) = (false, false)

(::VMFRot2)(mu, k) = Gen.random(VMFRot2(), mu, k)

export vmf_rot2
