# Gen3DRotations.jl
3D rotations in Gen

## Included probability distributions on 3D rotation

- Uniform distribution on unit quaternions (Haar measure)

- [Von Mises-Fisher distribution on unit quaternions](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)

## Included Metropolis-Hastings moves on 3D rotations

- Uniform proposal of new 3D rotation

- Rotate by a random small angle around a random axis

- Rotate by a random small angle around a given axis

- Flip 180 degrees around a given axis
