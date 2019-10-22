# Gen3DRotations.jl
3D rotations in Gen

## Included probability distributions on 3D rotations

These are distributions of type `Distribution{UnitQuaternion}`:

- Uniform distribution on unit quaternions (uniform distribution on the unit 3-sphere a.k.a. normalized Haar measure)

- [Von Mises-Fisher distribution on unit quaternions](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution). This may be useful for representing uncertainty around a given rotation.

NOTE: These two distributions report their densities with respect to the same base measure on `UnitQuaternion`s (the unnormalized Haar measure with normalizing constant 2 * pi^2).

## Included Metropolis-Hastings moves on 3D rotations

- Rotate by a uniformly chosen angle around a given axis.

- Rotate by a random small angle around a random axis

- Rotate by a random small angle around a given axis

- Flip 180 degrees around a given axis
