# GenDirectionalStats.jl

[![Build Status](https://travis-ci.com/probcomp/GenDirectionalStats.jl.svg?token=bxXxGvmE2n2G9iCjKFwG&branch=master)](https://travis-ci.com/probcomp/GenDirectionalStats.jl)

This package contains [Gen](https://www.gen.dev) probability distributions for plane rotations, rotations in 3D space, and directions in 3D space.
The package also contains several Gen involutive MCMC moves on 3D rotations.

## Probability distributions

### Probability distributions on plane rotations

Plane rotations are represented as elements of the following concrete type,
which is defined in [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl):

- `GenDirectionalStats.Rot2 = Rotations.RotMatrix{2,Float64,4}`

The supported probability distributions on plane rotations are:

- `GenDirectionalStats.uniform_rot2()`: Uniform distribution on plane rotations.

- `GenDirectionalStats.von_mises_rot2(location::Rot2, concentration::Real)`: The unimodal distribution [von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution) on plane rotations with a mode at `location` and given concentration.

The reference measure for random choices of this type is the uniform measure on the circle S^1.
The circle has a natural identification with the group of plane rotations [SO(2)](https://en.wikipedia.org/wiki/Circle_group), and under this identification the uniform measure coincides with the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure).
The uniform measure is uniquely determined up to a scaling constant; this package chooses the scaling constant so that the measure of the space of all plane rotations is `2 * pi`.
The density functions of all probability distribution(s) are defined relative to this measure.
For example, the probability density of the uniform distribution `uniform_rot2` at any point is `1/(2 * pi)`.

The plot below shows samples from these distributions (where I is the identity rotation).
Rotations are visualized as 2D coordinate frames (x-axis is red, y-axis is green).
The modes for the von Mises distributions (`location`) are shown as the bold coordinate frame,
a single sample from each distribution is shown as a light coordinate frame,
and 250 samples are represented by the endpoints of their two coordinate axis unit vectors.

<img alt="visualization of distributions on plane rotations" src="/examples/vmf_2d_rotation.png" width="600px">

### Probability distributions on 3D directions

Directions in 3D space are represented as elements of the following concrete type:

- `GenDirectionalStats.UnitVector3`

The supported probability distributions on 3D directions are:

- `GenDirectionalStats.uniform_3d_direction()`: Uniform distribution on 3D directions.

- `GenDirectionalStats.vmf_3d_direction(location::UnitVector3, concentration::Real)`: A unimodal [von Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) on the space of unit 3-vectors with a mode at `location` and given concentration.

The reference measure for random choices of this type is the [uniform measure](https://en.wikipedia.org/wiki/Spherical_measure) on the two-dimensional sphere S^2.
The uniform measure is uniquely determined up to a scaling constant; this package chooses the scaling constant so that the measure of the whole space S^2 is `4 * pi` (so the total measure equals the surface area).
The density functions of all probability distribution(s) are defined relative to this Haar measure.
For example, the probability density of the uniform distribution `uniform_3d_direction` is `1/(4 * pi)`.S

The plot below shows samples from these distributions.
The blue unit vectors are the modes of the von Mises Fisher distributions,
and 250 samples of unit 3-vectors from each distribution are shown as their end-points (points on the unit sphere).

<img alt="visualization of distributions on 3D directions" src="/examples/vmf_3d_direction.png" width="1000px">

### Probability distributions on 3D rotations

Rotations in 3D space, i.e. elements of the group [SO(3)](https://en.wikipedia.org/wiki/3D_rotation_group)
are represented as elements of the following concrete type,
which is defined in [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl).

- `GenDirectionalStats.Rot3 = Rotations.UnitQuaternion{Float64}`

The supported probability distributions on 3D rotations are:

- `GenDirectionalStats.uniform_rot3()`: Uniform distribution on 3D rotations.

- `GenDirectionalStats.vmf_rot3(location::Rot3, concentration::Real)`: A unimodal distribution on 3D rotations
that is based on the [von Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) on the space of unit 3-vectors with a mode at `location` and given concentration. The distribution is a mixture of 

- `GenDirectionalStats.uniform_vmf_rot3(location::Rot3, concentration::Real, prob_outlier::Real)`: 

The reference measure for random choices of this type is the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure) on SO(3).
This measure coincides with the [pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) of the [uniform measure](https://en.wikipedia.org/wiki/Spherical_measure) on S^3 by the double cover S^3 -> SO(3) induced by the group action of [quaternions as rotations](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).
This package defines the measure of all of SO(3) as `pi * pi`
(this value is chosen to be half of the surface area of the [3-sphere](https://en.wikipedia.org/wiki/3-sphere) S^3, motivated by the aforementioned double cover).
The density functions of all probability distribution(s) are defined relative to this Haar measure.
For example, the probability density of the uniform distribution `uniform_rot3` at any point is `1/(pi * pi)`.

The plot below shows samples from these distributions (where I is the identity rotation).
3D rotations are visualized as 3D coordinate frames centered at the origin.
The mode of each von Mises Fisher distribution is bold, one sample is shown non-bold, and 250 other samples are shown just via the endpoints of their coordinate axes.

<p align="center">
<img alt="visualization of distributions on 3D rotations" src="/examples/vmf_3d_rotation_1.png" width="1000px">
<img alt="visualization of distributions on 3D rotations" src="/examples/vmf_3d_rotation_2.png" width="400px">
</p>

## Hopf fibration

The following functions convert back and forth between 3D rotations (`Rot3`) and pairs of plane rotations (`Rot2`) and 3D directions (`UnitVector3`):

- `GenDirectionalStats.from_direction_and_plane_rotation(direction::UnitVector3, plane_rotation::Rot2)::Rot3`

- `GenDirectionalStats.to_direction_and_plane_rotation(rot3::Rot3)::Tuple{UnitVector3,Rot2}`

The 3D direction gives the z-axis of the rotated coordinate frame, and the plane rotation gives the rotation around the z-axis.
The 3D rotation corresponding to "zero plane rotation" is the rotation that carries the old z-axis to the new z-axis along a minor arc of great circle (i.e., the shortest possible path).
This parametrization of 3D rotations can be understood in terms of the [Hopf fibration](https://en.wikipedia.org/wiki/Hopf_fibration) S^3 -> S^2: choosing the shortest possible path of z-axis amounts to choosing a distinguished element (up to sign) of the fiber over each element of S^2.  These distinguished elements allow us to locally identify SO(3) with the Cartesian product S^2 x S^1, except at the singularities at `{[0, 0, -1]} x S^1` (anti-parallel z-axis).
Note that this parametrization is not the same as [axis-angle](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) parametrization.
Radon-Nikodym derivatives relating the measures involved are given by the constants:

- `GenDirectionalStats.TO_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION`

- `GenDirectionalStats.FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION`

## MCMC kernels

You can implement random walk or data-driven Metropolis-Hastings kernels using the unimodal probability distributions for each data type.

## Specialized reversible jump MCMC kernels on 3D rotations

This package also includes specialized reversible jump moves on 3D rotations.
Each move takes a trace, the address of the 3D rotation in the trace to move (must have type `GenDirectionalStats.Rot3`), and possible other arguments.

### Rotate by a uniformly chosen angle around a given axis.

This is useful for exploring rotations of objects that have some sort of cylindrical self-similarity (e.g. coffee mugs).

```julia
trace, = GenDirectionalStats.uniform_angle_fixed_axis_mh(trace, addr, axis)
```

### Rotate by a random small angle around a random axis

This is useful for constructing random walks on the space of rotations.

```julia
trace, = GenDirectionalStats.small_angle_random_axis(trace, addr, width)
```

### Rotate by a random small angle around a given axis

```julia
trace, = GenDirectionalStats.small_angle_fixed_axis_mh(trace, addr, axis, width)
```

### Flip 180 degrees around a given axis

```julia
trace, = GenDirectionalStats.flip_around_fixed_axis_mh(trace, addr, axis)
```

## Installing

From the Julia package manager REPL, run:
```
add https://github.com/probcomp/GenDirectionalStats.jl
```
