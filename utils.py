"""Shared helpers: data loading, model fitting with clustered SEs, table export."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import config as cfg


def load_crash():
    return pd.read_parquet(cfg.OUT / "data" / "crash_level.parquet")


def load_seg():
    return pd.read_parquet(cfg.OUT / "data" / "segment_year.parquet")


# -------------------------------------------------------------- fitting
def fit_logit(formula, df, groups):
    m = smf.logit(formula, df)
    start = sm.GLM(m.endog, m.exog, family=sm.families.Binomial()).fit().params
    return m.fit(start_params=np.asarray(start), disp=0, maxiter=200, cov_type="cluster",
                 cov_kwds={"groups": pd.factorize(df[groups])[0]}, use_t=True)


def fit_nb(formula, df, groups, offset="zero", verbose=True):
    """NB2 with offset, robust to bad starting values.
    Start from a Poisson GLM (IRLS, always stable) and a moment estimate of alpha,
    then try BFGS -> Newton -> Nelder-Mead(+BFGS polish) until a fit converges
    with a finite log-likelihood and finite clustered standard errors."""
    import warnings
    if offset not in df:
        df = df.assign(**{offset: 0.0})
    m = smf.negativebinomial(formula, df, offset=df[offset], loglike_method="nb2")
    pois = sm.GLM(m.endog, m.exog, family=sm.families.Poisson(), offset=m.offset).fit()
    mu, y = pois.mu, m.endog
    alpha0 = float(np.clip(((y - mu) ** 2 - y).sum() / (mu ** 2).sum(), 0.05, 20))
    start = np.r_[pois.params, alpha0]
    ckw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[groups])[0]}, use_t=True)
    tried = []
    for method, kw in (("bfgs", {"maxiter": 2000}), ("newton", {"maxiter": 200}),
                       ("nm", {"maxiter": 50000})):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = m.fit(start_params=start.copy(), method=method, disp=0, **kw, **ckw)
                if method == "nm":      # polish
                    r = m.fit(start_params=np.asarray(r.params), method="bfgs", disp=0,
                              maxiter=2000, **ckw)
            ok = (r.mle_retvals.get("converged", False) and np.isfinite(r.llf)
                  and np.all(np.isfinite(r.bse)) and r.params["alpha"] > 0)
            tried.append(f"{method}: converged={r.mle_retvals.get('converged')} llf={r.llf:.1f}")
            if ok:
                if verbose and len(tried) > 1:
                    print(f"    NB fit needed fallback ({'; '.join(tried)})")
                return r
        except Exception as e:      # singular Hessian, overflow, ...
            tried.append(f"{method}: {type(e).__name__}")
    raise RuntimeError(f"NB fit failed for '{formula}': {'; '.join(tried)}")


def fit_poisson(formula, df, groups, offset="zero"):
    if offset not in df:
        df = df.assign(**{offset: 0.0})
    m = smf.poisson(formula, df, offset=df[offset])
    start = sm.GLM(m.endog, m.exog, family=sm.families.Poisson(), offset=m.offset).fit().params
    return m.fit(start_params=np.asarray(start), disp=0, maxiter=200, cov_type="cluster",
                 cov_kwds={"groups": pd.factorize(df[groups])[0]}, use_t=True)


def refit_cluster(res, df, groups):
    """Same point estimates, different clustering variable."""
    return res.model.fit(start_params=res.params.to_numpy(), disp=0, maxiter=100,
                         cov_type="cluster",
                         cov_kwds={"groups": pd.factorize(df[groups])[0]}, use_t=True)


# -------------------------------------------------- wild cluster bootstrap
def webb_weights():
    return np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def score_wild_bootstrap(res, df, groups, terms, B=9999, seed=cfg.SEED):
    """Score (Kline-Santos) wild cluster bootstrap p-values for few clusters.
    Perturbs cluster score contributions with Webb 6-point weights.
    Returns DataFrame term -> bootstrap p-value."""
    rng = np.random.default_rng(seed)
    g = pd.factorize(df[groups])[0]
    G = g.max() + 1
    S = res.model.score_obs(res.params.to_numpy())                         # n x k
    Sg = np.vstack([S[g == j].sum(0) for j in range(G)])        # G x k
    Hinv = np.linalg.inv(-res.model.hessian(res.params.to_numpy()))
    V = res.cov_params().to_numpy()
    se = np.sqrt(np.diag(V))
    names = list(res.params.index)
    w = webb_weights()
    if 6 ** G <= B:                                             # enumerate all
        import itertools
        W = np.array(list(itertools.product(w, repeat=G)))
    else:
        W = rng.choice(w, size=(B, G))
    # bootstrap coefficient deviations; each draw's own CR variance is
    # approximated by the original (fast, conservative-enough for G=6)
    D = (W @ Sg) @ Hinv.T                                       # B x k
    out = {}
    for t in terms:
        k = names.index(t)
        t0 = res.params.iloc[k] / se[k]
        tb = D[:, k] / se[k]
        out[t] = (np.abs(tb) >= abs(t0)).mean()
    return pd.Series(out, name="wcb_p")


# -------------------------------------------------------- table export
def coef_table(res, exp=True, label=None):
    p = res.params
    ci = res.conf_int()
    tab = pd.DataFrame({"est": p, "lo": ci[0], "hi": ci[1], "p": res.pvalues})
    if exp:   # exponentiate coefficients, never the NB dispersion parameter
        rows = [i for i in tab.index if i != "alpha"]
        tab.loc[rows, ["est", "lo", "hi"]] = np.exp(tab.loc[rows, ["est", "lo", "hi"]])
    if label:
        tab.columns = pd.MultiIndex.from_product([[label], tab.columns])
    return tab


def fit_stats(res, name, n_groups=None):
    llnull = getattr(res, "llnull", np.nan)
    return pd.Series({
        "model": name, "n": int(res.nobs), "k": len(res.params),
        "loglik": res.llf, "loglik_null": llnull,
        "mcfadden_r2": 1 - res.llf / llnull if np.isfinite(llnull) else np.nan,
        "aic": res.aic, "bic": res.bic,
    })


def save(df, name, floatfmt="%.3f"):
    df.to_csv(cfg.OUT / "tables" / f"{name}.csv", float_format=floatfmt)
    print(f"  saved tables/{name}")


def lr_test_alpha(nb, pois):
    """LR test of alpha=0 (boundary): p = 0.5 * P(chi2_1 > LR)."""
    lr = 2 * (nb.llf - pois.llf)
    return lr, 0.5 * stats.chi2.sf(lr, 1)


def vuong(ll1_obs, ll2_obs):
    """Vuong non-nested test; positive z favours model 1."""
    m = ll1_obs - ll2_obs
    z = np.sqrt(len(m)) * m.mean() / m.std(ddof=1)
    return z, 2 * stats.norm.sf(abs(z))
