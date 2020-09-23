using Gen: mh, choicemap, @gen, @trace, uniform, normal, get_args, update, ChoiceMap, NoChange, bernoulli
using LinearAlgebra: norm
using GenDirectionalStats: Rot3

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

######################################################
# rotate by uniformly random angle around fixed axis #
######################################################


function compose_rotations(
        prev_q::Rotations.Rotation{3,Float64},
        change_q::Rotations.Rotation{3,Float64}, egocentric::Bool)
    # TODO check
    if egocentric
        return prev_q * change_q
    else
        return change_q * prev_q
    end
end

no_argdiffs(args) = map((_) -> NoChange(), args)

@gen function uniform_angle_fixed_axis_proposal(trace, addr, axis, egocentric)
    angle ~ uniform(0, 2 * pi)
    return angle
end

function uniform_angle_fixed_axis_involution(trace, ::ChoiceMap, angle::Real, prop_args; check=false)
    addr, axis, egocentric = prop_args
    prev_rot::Rot3 = trace[addr]
    change = Rotations.AngleAxis(angle, axis[1], axis[2], axis[3])
    new_rot = compose_rotations(prev_rot, change, egocentric)
    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_rot)))
    back_angle = 2 * pi - angle
    backward_choices = choicemap((:angle, back_angle))
    return (new_trace, backward_choices, w)
end

function uniform_angle_fixed_axis_mh(trace, addr, axis; check=false, egocentric=true)
    return mh(
        trace,
        uniform_angle_fixed_axis_proposal,
        (addr, axis, egocentric),
        uniform_angle_fixed_axis_involution;
        check=check)
end

export uniform_angle_fixed_axis_mh

###########################################
# rotate by 180 degrees around fixed axis #
###########################################

@gen function flip_proposal(trace, addr, axis, egocentric)
    return nothing
end

function flip_involution(trace, ::ChoiceMap, ::Nothing, prop_args; check=false)
    addr, axis, egocentric = prop_args
    prev_rot::Rot3 = trace[addr]
    angle = pi
    change = Rotations.AngleAxis(angle, axis[1], axis[2], axis[3])
    new_rot = compose_rotations(prev_rot, change, egocentric)
    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_rot)))
    backward_choices = choicemap()
    return (new_trace, backward_choices, w)
end

function flip_around_fixed_axis_mh(trace, addr, axis; check=false, egocentric=true)
    return mh(trace, flip_proposal, (addr, axis, egocentric), flip_involution; check=check)
end

export flip_around_fixed_axis_mh

##########################################
# random-walk on angle around fixed axis #
##########################################

@gen function small_angle_fixed_axis_proposal(trace, addr, axis, width, egocentric)
    angle_magnitude ~ uniform(0, width)
    direction ~ bernoulli(0.5)
    return (angle_magnitude, direction)
end

function small_angle_fixed_axis_involution(trace, ::ChoiceMap, prop_retval, prop_args; check=false)
    addr, axis, width, egocentric = prop_args
    angle_magnitude, direction = prop_retval
    prev_rot::Rot3 = trace[addr]
    if direction
        angle = angle_magnitude
    else
        angle = -angle_magnitude
    end
    change = Rotations.AngleAxis(angle, axis[1], axis[2], axis[3])
    new_rot = compose_rotations(prev_rot, change, egocentric)
    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_rot)))
    backward_choices = choicemap((:direction, !direction), (:angle_magnitude, angle_magnitude))
    return (new_trace, backward_choices, w)
end

function small_angle_fixed_axis_mh(trace, addr, axis, width; check=false, egocentric=true)
    return mh(
        trace,
        small_angle_fixed_axis_proposal,
        (addr, axis, width, egocentric),
        small_angle_fixed_axis_involution;
        check=check)
end

export small_angle_fixed_axis_mh

#############################################
# random-walk on angle around a random axis #
#############################################

@gen function small_angle_random_axis_proposal(trace, addr, width, egocentric)
    angle = @trace(uniform(-width, width), :angle)
    axis_x = @trace(uniform(-1, 1), :x)
    axis_y = @trace(uniform(-1, 1), :y)
    axis_z = @trace(uniform(-1, 1), :z)
    axis = [axis_x, axis_y, axis_z]
    axis_norm = norm(axis)
    axis = axis / axis_norm
    (angle, axis, axis_norm)
end

function small_angle_random_axis_involution(trace, ::ChoiceMap, prop_retval, prop_args; check=false)
    addr, width, egocentric = prop_args
    angle, axis, axis_norm = prop_retval
    prev_rot::Rot3 = trace[addr]
    change = Rotations.AngleAxis(angle, axis[1], axis[2], axis[3])
    new_rot = compose_rotations(prev_rot, change, egocentric)
    args = get_args(trace)
    argdiffs = no_argdiffs(args)
    new_trace, w = update(trace, args, argdiffs, choicemap((addr, new_rot)))
    backward_choices = choicemap(
        (:angle, angle),
        (:x, -axis[1] * axis_norm),
        (:y, -axis[2] * axis_norm),
        (:z, -axis[3] * axis_norm))
    return (new_trace, backward_choices, w)
end

function small_angle_random_axis_mh(trace, addr, width; check=false, egocentric=true)
    return mh(
        trace,
        small_angle_random_axis_proposal,
        (addr, width, egocentric),
        small_angle_random_axis_involution;
        check=check)
end

export small_angle_random_axis_mh
