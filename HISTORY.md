# 0.10.0

## Breaking changes

The minimum supported Julia version is now 1.10, and AbstractMCMC 3 and 4 are no longer supported.

AbstractMCMC 5.10 replaced the boolean `progress` keyword argument with `AbstractProgressKwarg` instances. `nested_isdone` used that value in a boolean context, so sampling failed with a `TypeError` on those releases. The value is now interpreted by type, which is why AbstractMCMC 5.10 or later is required.

`AbstractMCMC.step(rng, model, sampler, state)` was previously defined for every sampler type. It is now restricted to `sampler::Nested`, the only case NestedSamplers ever used it for.

## Other changes

MCMCChains 7, LogExpFunctions 1, and Parameters 0.13 are now allowed.
