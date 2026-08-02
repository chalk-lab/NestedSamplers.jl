using Distributions
using NestedSamplers
using StatsBase

@doc raw"""
Models.UnivariateGaussianGaussian()

Creates a model with Normal prior and Normal likelihood.

```math
\mathbf\theta \sim \mathcal{N}\left(\mu_p, \sigma_p\right)
```
```math
\mathbf{d} \sim \mathcal{N}\left(\mathbf\theta, \sigma_d\right)
```
the analytical evidence of the model is

```math
Z = \mathcal{N}\left(\mu_p, \sqrt{\sigma_p^2 + \sigma_d^2}\right)
```

## Examples
```jldoctest
julia> model, lnZ = Models.UnivariateNormalNormal(2, 0.5, 1, 0);

julia> lnZ
-2.6305103088617776
```
"""

function UnivariateGaussianGaussian(μp::Float64, σp::Float64, σd::Float64, d::Float64=0.0)
    priors = [Normal(μp, σp)]                   # θ ~ N(μ, σ)
    loglike(θ) = logpdf(Normal(d, σd),θ[1])     # d ~ N(θ, σd)
    model = NestedModel(loglike, priors)
    true_lnZ = logpdf(Normal(μp, sqrt(σp^2 + σd^2)), d) # lnZ is analytical in the data and the prior hyperparameters
    return model, true_lnZ
end
