# Gen3DRotations.jl

This package allows you to make random choices that are 3D rotations (represented internally as quaternions), and do custom inference about such choices.

A 3D rotation is represent by a `Geometry.UnitQuaternion`. See the [Geometry](https://github.com/probcomp/Geometry) package. Import it with:

```julia
using Geometry: UnitQuaternion
```

## Probability distributions on 3D rotations

NOTE: These distributions report their densities with respect to the same base measure on `UnitQuaternion`s (the unnormalized Haar measure with normalizing constant 2 * pi^2).

###  Uniform distribution
This is a uniform distribution on the unit 3-sphere a.k.a. normalized Haar measure.

```julia
rot::UnitQuaternion = @trace(Gen3DGeometry.uniform_3d_rotation(), :rot)
```

### [Von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution).

This may be useful for representing uncertainty around a given rotation (`mu`). Higher concentration means less variability around `mu`.

```julia
rot::UnitQuaternion = @trace(Gen3DGeometry.vmf_3d_rotation(mu::UnitQuaternion, concentration), :rot)
```

## Metropolis-Hastings moves on 3D rotations

Each move takes a trace, the address of the 3D rotation in the trace to move (must have type `UnitQuaternion`), and possible other arguments.

### Rotate by a uniformly chosen angle around a given axis.

```julia
trace, = Gen3DGeometry.uniform_angle_fixed_axis_mh(trace, addr, axis)
```

### Rotate by a random small angle around a random axis

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
