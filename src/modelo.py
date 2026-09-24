"""Dixon-Coles com decaimento temporal: máxima verossimilhança (com bootstrap) e bayesiano em Stan.

Parametrização comum aos dois métodos:
    gols do mandante  ~ Poisson(lam), log lam = mu0 + casa + ataque[m] + defesa[v]
    gols do visitante ~ Poisson(mu),  log mu  = mu0 +        ataque[v] + defesa[m]
    defesa alta = sofre mais gols; tau(x, y; rho) corrige os placares 0x0, 1x0, 0x1 e 1x1.
Cada jogo entra na verossimilhança com peso exp(-xi * dias até a data de referência).
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

RAIZ = Path(__file__).resolve().parents[1]
MAX_GOLS = 10
DP_RIDGE = 0.3  # desvio-padrão implícito da penalização das forças; é o "ridge leve" do CRITERIO.md


def preparar(treino, data_ref, xi):
    times = sorted(set(treino.mandante) | set(treino.visitante))
    idx = {t: i for i, t in enumerate(times)}
    dias = (data_ref - treino.data).dt.days.to_numpy()
    return dict(
        times=times, idx=idx,
        m=treino.mandante.map(idx).to_numpy(), v=treino.visitante.map(idx).to_numpy(),
        x=treino.gm.to_numpy(), y=treino.gv.to_numpy(),
        w=np.exp(-xi * dias),
    )


def _log_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.log(np.maximum(t, 1e-10))


def _desempacotar(th, n):
    return th[0], th[1], th[2], th[3:3 + n], th[3 + n:3 + 2 * n]


def _nll(th, p, n):
    mu0, casa, rho, at, de = _desempacotar(th, n)
    lam = np.exp(mu0 + casa + at[p["m"]] + de[p["v"]])
    mu = np.exp(mu0 + at[p["v"]] + de[p["m"]])
    ll = (_log_tau(p["x"], p["y"], lam, mu, rho)
          + p["x"] * np.log(lam) - lam + p["y"] * np.log(mu) - mu)
    pen = (np.sum(at ** 2) + np.sum(de ** 2)) / (2 * DP_RIDGE ** 2)
    return -np.sum(p["w"] * ll) + pen


def _grad(th, p, n):
    mu0, casa, rho, at, de = _desempacotar(th, n)
    m, v, x, y, w = p["m"], p["v"], p["x"], p["y"], p["w"]
    lam = np.exp(mu0 + casa + at[m] + de[v])
    mu = np.exp(mu0 + at[v] + de[m])
    # derivadas de log tau em relação a lam, mu e rho
    dl = np.zeros_like(lam); dm = np.zeros_like(lam); dr = np.zeros_like(lam)
    c00 = (x == 0) & (y == 0); c01 = (x == 0) & (y == 1)
    c10 = (x == 1) & (y == 0); c11 = (x == 1) & (y == 1)
    t00 = np.maximum(1 - lam * mu * rho, 1e-10)
    t01 = np.maximum(1 + lam * rho, 1e-10)
    t10 = np.maximum(1 + mu * rho, 1e-10)
    dl[c00] = (-mu * rho / t00)[c00]; dm[c00] = (-lam * rho / t00)[c00]; dr[c00] = (-lam * mu / t00)[c00]
    dl[c01] = (rho / t01)[c01]; dr[c01] = (lam / t01)[c01]
    dm[c10] = (rho / t10)[c10]; dr[c10] = (mu / t10)[c10]
    dr[c11] = -1 / (1 - rho)
    gl = w * (x - lam + lam * dl)  # derivada em relação a log lam
    gm = w * (y - mu + mu * dm)    # derivada em relação a log mu
    g = np.zeros_like(th)
    g[0] = -(gl.sum() + gm.sum())
    g[1] = -gl.sum()
    g[2] = -(w * dr).sum()
    ga = np.zeros(n); gd = np.zeros(n)
    np.add.at(ga, m, gl); np.add.at(ga, v, gm)
    np.add.at(gd, v, gl); np.add.at(gd, m, gm)
    g[3:3 + n] = -ga + at / DP_RIDGE ** 2
    g[3 + n:] = -gd + de / DP_RIDGE ** 2
    return g


def ajustar_mv(p, th0=None):
    n = len(p["times"])
    if th0 is None:
        th0 = np.concatenate([[0.1, 0.3, 0.0], np.zeros(2 * n)])
    lim = [(None, None), (None, None), (-0.2, 0.2)] + [(None, None)] * (2 * n)
    r = minimize(_nll, th0, args=(p, n), jac=_grad, method="L-BFGS-B", bounds=lim)
    mu0, casa, rho, at, de = _desempacotar(r.x, n)
    return dict(mu0=mu0, casa=casa, rho=rho, ataque=at, defesa=de, times=p["times"], th=r.x)


def bootstrap_mv(p, fit, n_boot=200, semente=1):
    """Bootstrap paramétrico: simula os placares do treino com o ajuste e reajusta."""
    rng = np.random.default_rng(semente)
    n = len(p["times"])
    lam = np.exp(fit["mu0"] + fit["casa"] + fit["ataque"][p["m"]] + fit["defesa"][p["v"]])
    mu = np.exp(fit["mu0"] + fit["ataque"][p["v"]] + fit["defesa"][p["m"]])
    amostras = []
    for _ in range(n_boot):
        pb = dict(p)
        pb["x"], pb["y"] = _amostrar_dc(lam, mu, fit["rho"], rng)
        amostras.append(ajustar_mv(pb, th0=fit["th"])["th"])
    a = np.array(amostras)
    return dict(mu0=a[:, 0], casa=a[:, 1], rho=a[:, 2], ataque=a[:, 3:3 + n],
                defesa=a[:, 3 + n:], times=p["times"])


def _amostrar_dc(lam, mu, rho, rng):
    """Sorteia placares da distribuição Dixon-Coles jogo a jogo (por matriz de placares)."""
    mat = matriz_placar(lam, mu, rho)          # (jogos, G, G)
    plano = mat.reshape(len(lam), -1)
    plano = plano / plano.sum(axis=1, keepdims=True)
    u = rng.random(len(lam))[:, None]
    k = (plano.cumsum(axis=1) < u).sum(axis=1)
    k = np.minimum(k, plano.shape[1] - 1)
    return k // (MAX_GOLS + 1), k % (MAX_GOLS + 1)


def matriz_placar(lam, mu, rho):
    g = np.arange(MAX_GOLS + 1)
    lam = np.atleast_1d(lam)[:, None]; mu = np.atleast_1d(mu)[:, None]
    pm = poisson.pmf(g[None, :], lam)
    pv = poisson.pmf(g[None, :], mu)
    mat = pm[:, :, None] * pv[:, None, :]
    rho = np.broadcast_to(np.atleast_1d(rho), (mat.shape[0],))
    mat[:, 0, 0] *= np.maximum(1 - lam[:, 0] * mu[:, 0] * rho, 0)
    mat[:, 0, 1] *= np.maximum(1 + lam[:, 0] * rho, 0)
    mat[:, 1, 0] *= np.maximum(1 + mu[:, 0] * rho, 0)
    mat[:, 1, 1] *= np.maximum(1 - rho, 0)
    return mat / mat.sum(axis=(1, 2), keepdims=True)


def _medias(par, mand, vis, k=None):
    """lam e mu dos jogos pedidos; time ausente do treino recebe força zero (a média)."""
    idx = {t: i for i, t in enumerate(par["times"])}
    def forca(vetor, times):
        return np.array([vetor[idx[t]] if t in idx else 0.0 for t in times])
    at = par["ataque"] if k is None else par["ataque"][k]
    de = par["defesa"] if k is None else par["defesa"][k]
    mu0 = par["mu0"] if k is None else par["mu0"][k]
    casa = par["casa"] if k is None else par["casa"][k]
    lam = np.exp(mu0 + casa + forca(at, mand) + forca(de, vis))
    mu = np.exp(mu0 + forca(at, vis) + forca(de, mand))
    return lam, mu


def prob_1x2(mat):
    casa = np.tril(np.ones((MAX_GOLS + 1,) * 2), -1)
    return np.stack([(mat * casa).sum(axis=(1, 2)),
                     np.trace(mat, axis1=1, axis2=2),
                     (mat * casa.T).sum(axis=(1, 2))], axis=1)


def prever_pontual(par, mand, vis):
    lam, mu = _medias(par, mand, vis)
    return prob_1x2(matriz_placar(lam, mu, par["rho"]))


def prever_amostras(par, mand, vis, n_max=400):
    """Probabilidade preditiva média sobre as amostras (posterior ou bootstrap)."""
    S = len(par["mu0"])
    ks = np.linspace(0, S - 1, min(S, n_max)).astype(int)
    tot = 0
    for k in ks:
        lam, mu = _medias(par, mand, vis, k)
        tot = tot + prob_1x2(matriz_placar(lam, mu, par["rho"][k]))
    return tot / len(ks)


# ---------------- bayesiano ----------------
_MODELO_STAN = None


def _stan():
    global _MODELO_STAN
    if _MODELO_STAN is None:
        from cmdstanpy import CmdStanModel
        from cmdstanpy.utils import cxx_toolchain_path
        cxx_toolchain_path()  # põe o RTools40 do CmdStan no PATH (o Windows não tem compilador próprio)
        _MODELO_STAN = CmdStanModel(stan_file=str(RAIZ / "stan" / "dc.stan"))
    return _MODELO_STAN


def ajustar_bayes(p, amostras=500, cadeias=4, semente=1):
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    n = len(p["times"])
    dados = dict(N=len(p["x"]), T=n, m=p["m"] + 1, v=p["v"] + 1,
                 x=p["x"].astype(int), y=p["y"].astype(int), w=p["w"])
    fit = _stan().sample(data=dados, chains=cadeias, parallel_chains=cadeias,
                         iter_warmup=500, iter_sampling=amostras, seed=semente,
                         show_progress=False)
    s = fit.stan_variables()
    diag = fit.summary()
    return dict(mu0=s["mu0"], casa=s["casa"], rho=s["rho"], ataque=s["ataque"],
                defesa=s["defesa"], times=p["times"],
                rhat_max=float(diag["R_hat"].max()), divergencias=int(fit.divergences.sum()))
