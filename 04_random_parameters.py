"""
04_random_parameters.py
-----------------------
Random-parameters robustness models (Reviewer 5), simulated maximum
likelihood with scrambled Halton draws, written in PyTorch (autograd gives
exact gradients and the Hessian for standard errors).

  RPL  crash-level random-parameters logit, spec B
       random (normal): curv, abs_grade
  RPNB segment-level random-parameters NB2 panel, spec D
       random (normal): intercept, curv_max, grade_max
       draws shared by the same segment across 2021-2023 (panel), so the
       random intercept absorbs time-invariant unobserved segment risk.

Each random parameter is reported as mean and SD; the share of the
distribution above zero is reported for interpretation. Fixed-parameter
counterparts (B, D) come from 03_models.py for an LR comparison.

pip install torch   (CPU build is enough)
"""
import numpy as np
import pandas as pd
import torch
from patsy import dmatrices
from scipy.stats import norm, qmc, chi2

import config as cfg
from importlib import import_module

torch.set_default_dtype(torch.float64)
M = import_module("03_models")          # reuse prep() and formulas
R_DRAWS = 200                            # raise to 500 for the final run if time allows


def halton_normal(G, K, R, seed=cfg.SEED):
    h = qmc.Halton(d=K, scramble=True, seed=seed).random(G * R)
    return torch.tensor(norm.ppf(np.clip(h, 1e-10, 1 - 1e-10)).reshape(G, R, K))


class SML:
    def __init__(self, formula, df, random_terms, family, group=None, offset=None):
        y, X = dmatrices(formula, df, return_type="dataframe")
        self.names = list(X.columns)
        self.ridx = [self.names.index(t) for t in random_terms]
        self.fidx = [i for i in range(len(self.names)) if i not in self.ridx]
        self.X = torch.tensor(X.to_numpy())
        self.y = torch.tensor(y.to_numpy().ravel())
        self.off = torch.tensor(df.loc[X.index, offset].to_numpy()) if offset else 0.0
        g = df.loc[X.index, group] if group else pd.Series(np.arange(len(X)), index=X.index)
        self.g = torch.tensor(pd.factorize(g)[0])
        self.G = int(self.g.max()) + 1
        self.family = family
        self.Z = halton_normal(self.G, len(self.ridx), R_DRAWS)        # G x R x Kr
        self.random_terms = random_terms

    def unpack(self, th):
        k, kr = len(self.names), len(self.ridx)
        beta = th[:k]
        log_sd = th[k:k + kr]
        log_a = th[k + kr] if self.family == "nb" else None
        return beta, log_sd, log_a

    def loglik(self, th):
        beta, log_sd, log_a = self.unpack(th)
        eta_f = self.X[:, self.fidx] @ beta[self.fidx] + self.off               # N
        Xr = self.X[:, self.ridx]                                                # N x Kr
        b_r = beta[self.ridx] + torch.exp(log_sd) * self.Z                       # G x R x Kr
        eta = eta_f[:, None] + (Xr[:, None, :] * b_r[self.g]).sum(-1)            # N x R
        y = self.y[:, None]
        if self.family == "logit":
            ll = y * eta - torch.nn.functional.softplus(eta)
        else:
            a = torch.exp(log_a)
            mu = torch.exp(eta)
            ll = (torch.lgamma(y + 1 / a) - torch.lgamma(1 / a) - torch.lgamma(y + 1)
                  - (1 / a) * torch.log1p(a * mu) + y * (torch.log(a * mu) - torch.log1p(a * mu)))
        LLg = torch.zeros(self.G, ll.shape[1]).index_add_(0, self.g, ll)        # G x R
        return (torch.logsumexp(LLg, 1) - np.log(LLg.shape[1])).sum()

    def fit(self, start_beta):
        kr = len(self.ridx)
        th0 = list(start_beta) + [np.log(0.1)] * kr + ([np.log(0.5)] if self.family == "nb" else [])
        th = torch.tensor(th0, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1, max_iter=500, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-6)

        def closure():
            opt.zero_grad()
            loss = -self.loglik(th)
            loss.backward()
            return loss
        opt.step(closure)
        self.theta = th.detach()
        self.llf = float(self.loglik(self.theta))
        H = torch.autograd.functional.hessian(lambda t: -self.loglik(t), self.theta)
        self.cov = torch.linalg.inv(H).numpy()
        return self

    def table(self):
        k, kr = len(self.names), len(self.ridx)
        th, se = self.theta.numpy(), np.sqrt(np.diag(self.cov))
        rows = []
        for i, n in enumerate(self.names):
            lab = f"{n} (mean)" if i in self.ridx else n
            rows.append([lab, th[i], se[i]])
        for j, i in enumerate(self.ridx):
            sd = np.exp(th[k + j])
            sd_se = sd * se[k + j]                          # delta method
            rows.append([f"{self.names[i]} (SD)", sd, sd_se])
            rows.append([f"{self.names[i]} (share > 0)", norm.cdf(th[i] / sd), np.nan])
        if self.family == "nb":
            rows.append(["alpha", np.exp(th[-1]), np.exp(th[-1]) * se[-1]])
        t = pd.DataFrame(rows, columns=["term", "est", "se"]).set_index("term")
        t["z"] = t["est"] / t["se"]
        t["p"] = 2 * norm.sf(t["z"].abs())
        return t


def main():
    cr, sg = M.prep()
    mB = M.fit_logit(M.F_B, cr, "district")
    mD = M.fit_nb(M.F_E, sg, "district")          # primary SPF (E)

    rpl = SML(M.F_B, cr, ["curv", "abs_grade"], "logit").fit(mB.params.to_numpy())
    rpnb = SML(M.F_E, sg, ["Intercept", "any_curve", "curv_max"], "nb",
               group="seg_key").fit(mD.params.to_numpy()[:-1])

    out = []
    for m, fixed, name in [(rpl, mB, "RPL"), (rpnb, mD, "RPNB")]:
        t = m.table()
        t.to_csv(cfg.OUT / "tables" / f"tab_{name}.csv", float_format="%.4f")
        lr = 2 * (m.llf - fixed.llf)
        df_ = len(m.ridx)
        out.append([name, fixed.llf, m.llf, lr, df_, chi2.sf(lr, df_),
                    -2 * fixed.llf + 2 * len(fixed.params),
                    -2 * m.llf + 2 * len(m.theta), R_DRAWS])
    pd.DataFrame(out, columns=["model", "LL_fixed", "LL_random", "LR", "df", "p",
                               "AIC_fixed", "AIC_random", "draws"]) \
        .to_csv(cfg.OUT / "tables" / "tab_RP_vs_fixed.csv", index=False, float_format="%.3f")
    print("done")


if __name__ == "__main__":
    main()
