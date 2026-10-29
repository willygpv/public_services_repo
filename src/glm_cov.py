"""HC3 covariance matrix for statsmodels GLM results."""

import numpy as np
import pandas as pd


def cov_hc3(res) -> pd.DataFrame:
    X = np.asarray(res.model.exog, dtype=float)
    y = np.asarray(res.model.endog, dtype=float)
    mu = np.asarray(res.mu, dtype=float)
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    h = mu * np.einsum('ij,jk,ik->i', X, bread, X)
    s = X * ((y - mu) / (1.0 - h))[:, None]
    return pd.DataFrame(bread @ (s.T @ s) @ bread,
                        index=res.params.index, columns=res.params.index)
