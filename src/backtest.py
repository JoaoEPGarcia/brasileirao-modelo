"""Previsão fora da amostra, bloco de 7 dias a bloco de 7 dias (ver CRITERIO.md).

    python src/backtest.py xi            # escolhe o decaimento em 2013-2015 (só MV)
    python src/backtest.py teste XI      # gera as previsões de 2016-2026 com os dois métodos
"""
import sys
import time

import numpy as np
import pandas as pd

import dados
import modelo as M

CACHE = dados.RAIZ / "resultados" / "cache"
GRADE_XI = [0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]


def blocos(d, temporada):
    s = d[d.temporada == temporada]
    t = s.data.min()
    while t <= s.data.max():
        yield t, s[(s.data >= t) & (s.data < t + pd.Timedelta(days=7))]
        t += pd.Timedelta(days=7)


def prever_temporada(d, temporada, xi, bayes):
    linhas, th = [], None
    for t, alvo in blocos(d, temporada):
        if alvo.empty:
            continue
        treino = d[d.temporada.isin([temporada - 1, temporada]) & (d.data < t)]
        p = M.preparar(treino, t, xi)
        fit = M.ajustar_mv(p, th0=th if th is not None and len(th) == 3 + 2 * len(p["times"]) else None)
        th = fit["th"]
        r = alvo[["data", "temporada", "mandante", "visitante", "res"]].copy()
        r[["mv_c", "mv_e", "mv_f"]] = M.prever_pontual(fit, alvo.mandante, alvo.visitante)
        if bayes:
            fb = M.ajustar_bayes(p)
            r[["by_c", "by_e", "by_f"]] = M.prever_amostras(fb, alvo.mandante, alvo.visitante)
            r["by_rhat"], r["by_div"] = fb["rhat_max"], fb["divergencias"]
        linhas.append(r)
    return pd.concat(linhas)


def rps(prob, res):
    o = np.stack([res == "H", res == "D", res == "A"], axis=1).astype(float)
    c = np.cumsum(prob, axis=1)[:, :2] - np.cumsum(o, axis=1)[:, :2]
    return 0.5 * (c ** 2).sum(axis=1)


def escolher_xi(d):
    out = []
    for xi in GRADE_XI:
        prev = pd.concat([prever_temporada(d, s, xi, bayes=False) for s in (2013, 2014, 2015)])
        out.append((xi, rps(prev[["mv_c", "mv_e", "mv_f"]].to_numpy(), prev.res.to_numpy()).mean()))
        print(f"xi={xi:<7} RPS={out[-1][1]:.5f}", flush=True)
    melhor = min(out, key=lambda z: z[1])
    print("escolhido:", melhor[0])
    return melhor[0]


def main():
    d = dados.carregar()
    CACHE.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "xi":
        escolher_xi(d)
        return
    xi = float(sys.argv[2])
    for s in range(2016, 2027):
        arq = CACHE / f"prev_{s}_xi{xi}.csv"
        if arq.exists():
            continue
        t0 = time.time()
        prever_temporada(d, s, xi, bayes=True).to_csv(arq, index=False)
        print(f"{s} pronto em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
