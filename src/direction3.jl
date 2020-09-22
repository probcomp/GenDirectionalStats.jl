import Distributions
import LinearAlgebra#: norm, cross, det
import Rotations
using StaticArrays: SVector

struct UnitVector3
    v::SVector{3,Float64}
end

Base.isapprox(a::UnitVector3, b::UnitVector3) = isapprox(a.v, b.v)

function UnitVector3(a, b, c)
    v = SVector{3,Float64}(a, b, c)
    @assert isapprox(norm(v), 1.0)
    return UnitVector3(v)
end

function Base.rand(::Type{UnitVector3})
    v = SVector{3,Float64}(randn(), randn(), randn())
    return UnitVector3(v / norm(v))
end

@inline function _to_array(x::UnitVector3)
    return Float64[x.v[1], x.v[2], x.v[3]]
end

total_area(::Type{UnitVector3}) = 4 * pi

export UnitVector3

##############################################
# uniform distribution (Haar measure on S^2) #
##############################################

struct UniformDirection3D <: Gen.Distribution{UnitVector3} end

Gen.logpdf(::UniformDirection3D, x::UnitVector3) = -log(total_area(UnitVector3))

function Gen.logpdf_grad(::UniformDirection3D, x::UnitVector3)
    error("Not implemented") # TODO implement
    return (nothing,)
end

Gen.random(::UniformDirection3D) = rand(UnitVector3)

Gen.has_output_grad(::UniformDirection3D) = false

Gen.has_argument_grads(::UniformDirection3D) = ()

(::UniformDirection3D)() = Gen.random(UniformDirection3D())

const uniform_3d_direction = UniformDirection3D()

export uniform_3d_direction

########################################
# von Mises fisher distribution on S^2 #
########################################

struct VMFDirection3D <: Gen.Distribution{UnitVector3} end

function Gen.logpdf(::VMFDirection3D, x::UnitVector3, mu::UnitVector3, k::Real)
    dist = Distributions.VMFDirection3D(_to_array(mu), k)
    return Distributions.logpdf(_to_array(x))
end

function Gen.logpdf_grad(::VMFDirection3D, x::UnitVector3, mu::UnitVector3, k::Real)
    error("Not implemented") # TODO implement
    return (nothing, nothing, nothing)
end

function Gen.random(::VMFDirection3D, mu::UnitVector3, k::Real)
    v = rand(Distributions.VMFDirection3D(_to_array(mu), k))
    @assert length(v) == 3
    return UnitVector3(v[1], v[2], v[3])
end

Gen.has_output_grad(::VMFDirection3D) = false

Gen.has_argument_grads(::VMFDirection3D) = (false, false)

(::VMFDirection3D)() = Gen.random(VMFDirection3D())

const vmf_3d_direction = VMFDirection3D()

export vmf_3d_direction
