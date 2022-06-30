import Gen
import Distributions
import LinearAlgebra: norm, cross, det, dot
import Rotations
using StaticArrays: SVector, SMatrix

# Hopf fibration, see: https://en.wikipedia.org/wiki/Hopf_fibration
# also see http://lavalle.pl/anna/papers/YerJaiLavMit10.pdf

normalize(vec) = vec / norm(vec)

# Finds the rotation that rotates that minimizes the cosine distance, giving a unique way of defining the new axes
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
function find_minimal_angle_rotation(a::SVector{3,Float64}, b::SVector{3,Float64})
    a = normalize(a)
    b = normalize(b)
    if isapprox(a, -b)
        @warn "Two vectors are almost opposites; beware"
    end
    v = cross(a, b)
    s = norm(v)
    c = dot(a, b)
    Vcross = SMatrix{3,3,Float64}(
        0.0, v[3], -v[2], # column 1
        -v[3], 0.0, v[1], # column 2
        v[2], -v[1], 0.0) # column 3
    R = (
        (SMatrix{3,3,Float64}(
            1.0, 0.0, 0.0, # column 1
            0.0, 1.0, 0.0, # column 2
            0.0, 0.0, 1.0)) # column 3
        .+ Vcross .+ ((Vcross * Vcross) / (1 + c)))
    @assert isapprox(det(R), 1.0)
    return R
end

function check_angle_bounds(angle)
    if angle < -pi || angle > pi
        error("angle out of bounds [-pi, pi]")
    end
end

function to_rotation_matrix(z_axis::SVector{3,Float64}, angle_around_z_axis::Float64)
    if !isapprox(norm(z_axis), 1.0)
        error("z_axis was not a unit vector")
    end
    check_angle_bounds(angle_around_z_axis)
    R2DoF = find_minimal_angle_rotation(SVector{3,Float64}(0.0, 0.0, 1.0), z_axis)
    R1DoF = SMatrix{3,3,Float64}(
        cos(angle_around_z_axis), sin(angle_around_z_axis), 0.0, # column 1
        -sin(angle_around_z_axis), cos(angle_around_z_axis), 0.0, # column 2
        0.0, 0.0, 1.0) # column 3
    @assert isapprox(det(R1DoF), 1.0)
    R = R2DoF * R1DoF
    @assert isapprox(det(R), 1.0)
    return R
end

function is_rotation_around_z(R)
    eps = 1e-6
    return (norm(R * SVector{3,Float64}(0.0, 0.0, 1.0) .- SVector{3,Float64}(0.0, 0.0, 1.0)) < eps &&
        abs(R[3,1]) < eps &&
        abs(R[3,2]) < eps &&
        isapprox(R[2,2], R[1,1]; atol=eps) &&
        isapprox(R[1,2], -R[2,1]; atol=eps))
end

function from_rotation_matrix(R::SMatrix{3,3,Float64})::Tuple{SVector{3,Float64},Float64}
    z_axis = R[:,3]
    R2DoF = find_minimal_angle_rotation(SVector{3,Float64}(0.0, 0.0, 1.0), z_axis)
    R1DoF = R2DoF \ R
    @show is_rotation_around_z(R1DoF)
    @assert is_rotation_around_z(R1DoF)
    angle_around_z_axis = atan(R1DoF[2,1], R1DoF[1,1])
    return (z_axis, angle_around_z_axis)
end

function to_direction_and_plane_rotation(rot3::Rot3)::Tuple{UnitVector3,Rot2}
    R = Rotations.RotMatrix{3,Float64,9}(rot3).mat
    (_z_axis, _angle_around_z_axis) = from_rotation_matrix(R)
    direction = UnitVector3(_z_axis)
    rot2 = Rotations.RotMatrix{2,Float64}(_angle_around_z_axis)
    return (direction, rot2)
end

function from_direction_and_plane_rotation(direction::UnitVector3, plane_rotation::Rot2)::Rot3
    R = to_rotation_matrix(direction.v, Rotations.rotation_angle(plane_rotation))
    return Rot3(R)
end

# the Jacobian determinant of the transformation between S^3 and
# S^2 x S^1 should be constant.
# the total measure of S^3 is 2 * pi * pi
# the total measure of (S^2) x (S^1) is (4 * pi) * (2 * pi) = 8 * pi * pi
# therefore, the RJMCMC acceptance ratio must be multiplied/divided by a factor of 4

const FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION = (
    -log(total_area(UnitVector3)) - log(total_area(Rot2)) + log(total_area(Rot3)))

const TO_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION = -FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION

export to_direction_and_plane_rotation
export from_direction_and_plane_rotation
export FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION
export TO_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION
