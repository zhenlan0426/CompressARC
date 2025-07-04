# Why Sample Solutions Across Epochs

In the ARC-AGI solver we do **online solution counting**: every training step we record one or two candidate output grids and accumulate their (log-)scores.  This means we effectively sample solutions from many different epochs instead of trusting a single forward pass from the final model.  The main reasons are:

1. **Tiny training sets → broad weight posterior**  
   • With ≤10 input-output pairs, many distinct parameter settings $\theta$ achieve nearly identical evidence lower bound (ELBO).  
   • Picking only the maximum-a-posteriori (MAP) $\theta$ risks focusing on a *single* mode of a wide, multimodal posterior.

2. **Bayesian model averaging for free**  
   • Each SGD/Adam update produces a fresh $\theta$ and we already run a forward pass with a new latent sample $z$.  
   • Treating the optimisation trajectory as correlated samples from the posterior $p(\theta\,|\,D)$ lets us integrate over parameter uncertainty without extra compute.

3. **Integrating over both uncertainties**  
   • We want $\hat y^{\ast} = \operatorname*{argmax}_y \int p(y\,|\,z,\theta)\,q(z\,|\,x,\theta)\,p(\theta\,|\,D)\,dz\,d\theta$.  
   • Online logging marginalises over **latent noise** (by sampling $z$ each step) *and* over **weight uncertainty** (via the epoch trajectory).

4. **Reduced sensitivity to unlucky draws**  
   • A single final-epoch sample may hit a low-probability mode due to stochastic $z$.  
   • Accumulating log-probability over hundreds of draws lets the highest-posterior grid dominate.

5. **Early-epoch noise is automatically down-weighted**  
   • The logger scores grids by $-10\times\text{uncertainty}$; high-uncertainty outputs accrue little weight.  
   • An extra penalty on the first ~150 steps further suppresses random early solutions.

6. **Cheap memory footprint**  
   • We store only two hashes (best & runner-up) plus a dictionary of log-counts—$\mathcal O(1)$ per iteration.  
   • A post-training Monte-Carlo sweep at the final $\theta$ would require as many forward passes *in addition* to training.

### Take-away
Sampling solutions throughout training gives a lightweight form of Bayesian model averaging that is empirically more reliable—especially in data-scarce ARC tasks—than relying on any single $\theta, z$ pair from the very end of optimisation. 