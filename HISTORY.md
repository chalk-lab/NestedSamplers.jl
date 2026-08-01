# 0.9.0

## Breaking changes

Requires Julia 1.10 and AbstractMCMC 5.10 or later.

Sampling threw a `TypeError` on AbstractMCMC 5.10 and later, which swapped the boolean `progress` keyword for `AbstractProgressKwarg` instances. `nested_isdone` now reads it by type.

`AbstractMCMC.step(rng, model, sampler, state)` is now restricted to `sampler::Nested` instead of matching every sampler type.

MCMCChains compat is now `"6, 7"`. Earlier versions cap AbstractMCMC below 5, so they cannot resolve anyway.

## Other changes

LogExpFunctions 1 and Parameters 0.13 are now allowed.
