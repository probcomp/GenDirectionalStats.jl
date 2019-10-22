using Gen: mh, choicemap, @gen, @trace, uniform, normal, get_args, update, ChoiceMap, NoChange, bernoulli
using LinearAlgebra: norm
using Geometry: qmult, axis_angle_to_quat

###############################
# random walk on some address #
###############################

@gen function random_walk_proposal(trace, address, width)
    prev_value = trace[address]
    @trace(normal(prev_value, width), address)
end

function random_walk_mh(trace, address, width)
    mh(trace, random_walk_proposal, (address, width))
end

export random_walk_mh

########################
# moves on quaternions #
########################

function axis_angle_to_quat(axis, angle)
    [cos(angle/2),
     sin(angle/2) * axis[1],
     sin(angle/2) * axis[2],
     sin(angle/2) * axis[3]]
end

function no_argdiffs(args)
    map((_) -> NoChange(), args)
end

@gen function uniform_angle_fixed_axis_proposal(trace, addr, axis)
    return @trace(uniform(0, 2 * pi), :angle)
end

function uniform_angle_fixed_axis_involution(trace, ::ChoiceMap, angle::Float64, prop_args)
    addr, axis = prop_args
    if length(axis) != 3 || !isapprox(norm(axis), 1)
        error("axis must be unit 3-vector")
    end

    prev_q::UnitQuaternion = trace[addr]
    change_q = axis_angle_to_quat(axis, angle)
    new_q = qmult(change_q, prev_q)
    new_q = new_q * mag

    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_q)))

    back_angle = 2 * pi - angle
    backward_choices = choicemap((:angle, back_angle))
    (new_trace, backward_choices, w)
end

function uniform_angle_fixed_axis_mh(trace, addr, axis)
    mh(
        trace,
        uniform_angle_fixed_axis_proposal,
        (addr, axis),
        uniform_angle_fixed_axis_involution)
end

export uniform_angle_fixed_axis_mh

@gen function flip_proposal(trace, addr, axis)
    nothing
end

function flip_involution(trace, ::ChoiceMap, ::Nothing, prop_args)
    addr, axis = prop_args
    if length(axis) != 3 || !isapprox(norm(axis), 1)
        error("axis must be unit 3-vector")
    end

    prev_q::UnitQuaternion = trace[addr]
    angle = pi
    change_q = axis_angle_to_quat(axis, angle)
    new_q = qmult(change_q, prev_q)
    new_q = new_q * mag

    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_q)))

    backward_choices = choicemap()
    (new_trace, backward_choices, w)
end

function flip_around_fixed_axis_mh(trace, addr, axis)
    mh(trace, flip_proposal, (addr, axis), flip_involution)
end

export flip_around_fixed_axis_mh

@gen function small_angle_fixed_axis_proposal(trace, addr, axis, width)
    angle_magnitude = @trace(uniform(0, width), :angle_magnitude)
    direction = @trace(bernoulli(0.5), :direction)
    (angle_magnitude, direction)
end

function small_angle_fixed_axis_involution(trace, ::ChoiceMap, prop_retval, prop_args)
    addr, axis, width = prop_args
    if length(axis) != 3 || !isapprox(norm(axis), 1)
        error("axis must be unit 3-vector")
    end
    angle_magnitude, direction = prop_retval

    prev_q::UnitQuaternion = trace[addr]
    mag = norm(prev_q)
    prev_q = prev_q / mag
    @assert isapprox(norm(prev_q), 1.)

    if direction
        angle = angle_magnitude
    else
        angle = -angle_magnitude
    end
    change_q = axis_angle_to_quat(axis, angle)
    new_q = qmult(change_q, prev_q)
    @assert isapprox(norm(new_q), 1.)
    new_q = new_q * mag
    @assert isapprox(norm(new_q), mag)

    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, make_quat_choicemap(addr, new_q))

    backward_choices = choicemap((:direction, !direction), (:angle_magnitude, angle_magnitude))
    (new_trace, backward_choices, w)
end

function small_angle_fixed_axis_mh(trace, addr, axis, width)
    mh(
        trace,
        small_angle_fixed_axis_proposal,
        (addr, axis, width),
        small_angle_fixed_axis_involution)
end

export small_angle_fixed_axis_mh

@gen function small_angle_random_axis_proposal(trace, addr, width)
    angle = @trace(uniform(0, width), :angle)
    axis_x = @trace(uniform(0, 1), :x)
    axis_y = @trace(uniform(0, 1), :y)
    axis_z = @trace(uniform(0, 1), :z)
    axis = [axis_x, axis_y, axis_z]
    axis_norm = norm(axis)
    axis = axis / axis_norm
    (angle, axis, axis_norm)
end

function small_angle_random_axis_involution(trace, ::ChoiceMap, prop_retval, prop_args)
    addr, width = prop_args
    angle, axis, axis_norm = prop_retval
    @assert length(axis) == 3
    @assert isapprox(norm(axis), 1.)

    prev_q::UnitQuaternion = trace[addr]
    change_q = axis_angle_to_quat(axis, angle)
    new_q = qmult(change_q, prev_q)

    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, make_quat_choicemap(addr, new_q))

    backward_choices = choicemap(
        (:angle, angle),
        (:x, -axis[1] * axis_norm),
        (:y, -axis[2] * axis_norm),
        (:z, -axis[3] * axis_norm))
    (new_trace, backward_choices, w)
end

function small_angle_random_axis_mh(trace, addr, width)
    mh(
        trace,
        small_angle_random_axis_proposal,
        (addr, width),
        small_angle_random_axis_involution)
end

export small_angle_random_axis_mh