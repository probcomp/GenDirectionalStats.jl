# GenDirectionalStats.jl

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

The reference measure for random choices of this type is the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure) on the group of plane rotations [SO(2)](https://en.wikipedia.org/wiki/Circle_group).
This package defines the Haar measure of the space of all plane rotations as `2 * pi`.
The density functions of all probability distribution(s) are defined relative to this Haar measure.
For example, the probability density of the uniform distribution `uniform_rot2` is `1/(2 * pi)`.

<img alt="visualization of distributions on plane rotations" src="/examples/vmf_2d_rotation.png" width="1000px">

### Probability distributions on 3D directions

Directions in 3D space are represented as elements of the following concrete type:

- `GenDirectionalStats.UnitVector3`

The supported probability distributions on 3D directions are:

- `GenDirectionalStats.uniform_3d_direction()`: Uniform distribution on 3D directions.

- `GenDirectionalStats.vmf_3d_direction(location::UnitVector3, concentration::Real)`: A unimodal [von Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) on the space of unit 3-vectors with a mode at `location` and given concentration.

The reference measure for random choices of this type is the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure) on the two-dimensional sphere (the group S^2).
This package defines the Haar measure of the space of all 3D directions as `4 * pi`.
The density functions of all probability distribution(s) are defined relative to this Haar measure.
For example, the probability density of the uniform distribution `uniform_3d_direction` is `1/(4 * pi)`.

<img alt="visualization of distributions on 3D directions" src="/examples/vmf_3d_direction.png" width="1000px">

### Probability distributions on 3D rotations

Rotations in 3D space, i.e. elements of the group [SO(3)](https://en.wikipedia.org/wiki/3D_rotation_group)
are represented as elements of the following concrete type,
which is defined in [Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl).
```julia
GenDirectionalStats.Rot3 = Rotations.UnitQuaternion{Float64}
```

The supported probability distributions on 3D rotations are:

- `GenDirectionalStats.uniform_rot3()`: Uniform distribution on 3D rotations.

- `GenDirectionalStats.vmf_rot3(location::Rot3, concentration::Real)`: A unimodal distribution on 3D rotations
that is based on the [von Mises Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) on the space of unit 3-vectors with a mode at `location` and given concentration. The distribution is a mixture of 

- `GenDirectionalStats.uniform_vmf_rot3(location::Rot3, concentration::Real, prob_outlier::Real)`: 

The reference measure for random choices of this type is the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure) on SO(3).
This package defines the Haar measure of all of SO(3) as `pi * pi`
(this value is chosen to be half of the area of the area of the [3-sphere](https://en.wikipedia.org/wiki/3-sphere).
The density functions of all probability distribution(s) are defined relative to this Haar measure.
For example, the probability density of the uniform distribution `uniform_rot3` is `1/(pi * pi)`.

<p align="center">
<img alt="visualization of distributions on 3D rotations" src="/examples/vmf_3d_rotation_1.png" width="1000px">
<img alt="visualization of distributions on 3D rotations" src="/examples/vmf_3d_rotation_2.png" width="400px">
</p>


## Customizable MCMC moves on 3D rotations

Each move takes a trace, the address of the 3D rotation in the trace to move (must have type `UnitQuaternion`), and possible other arguments.

### Rotate by a uniformly chosen angle around a given axis.

This is useful for exploring rotations of objects that have some sort of cylindrical self-similarity (e.g. coffee mugs).

```julia
trace, = Gen3DGeometry.uniform_angle_fixed_axis_mh(trace, addr, axis)
```

### Rotate by a random small angle around a random axis

This is useful for constructing random walks on the space of rotations.

```julia
trace, = Gen3DGeometry.small_angle_random_axis(trace, addr, width)
```

### Rotate by a random small angle around a given axis

```julia
trace, = Gen3DGeometry.small_angle_fixed_axis_mh(trace, addr, axis, width)
```

### Flip 180 degrees around a given axis

```julia
trace, = Gen3DGeometry.flip_around_fixed_axis_mh(trace, addr, axis)
```

### Selection MH

You can also use the `select` variant of `mh`, which will propose from the prior distribution on the 3D rotation:

```julia
trace, = Gen.mh(trace, Gen.select(addr))
```

## Installing

From the Julia package manager REPL, run:
```
add https://github.com/probcomp/GenDirectionalStats.jl
```
