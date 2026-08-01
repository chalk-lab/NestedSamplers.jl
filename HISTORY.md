# 0.9.0

The release that adds AbstractMCMC 5 support, asked for in #107.

## Breaking changes

Requires Julia 1.10 and AbstractMCMC 5.10 or later. AbstractMCMC 3 and 4 are no longer supported.

Sampling threw a `TypeError` on AbstractMCMC 5.10 and later, which swapped the boolean `progress` keyword for `AbstractProgressKwarg` instances. `nested_isdone` now reads it by type.

`AbstractMCMC.step(rng, model, sampler, state)` is now restricted to `sampler::Nested` instead of matching every sampler type.

MCMCChains compat is now `"6, 7"`. Earlier versions cap AbstractMCMC below 5, so they cannot resolve anyway.

## Other changes

The package moved from TuringLang to the chalk-lab organisation. Docs are now served from https://chalk-lab.github.io/NestedSamplers.jl.

Parameters 0.13 is now allowed.
