"""Simula o restante do Brasileirão 2026 com os dois métodos.   python src/simular.py XI

Cada simulação usa uma amostra dos parâmetros (posterior do Stan, ou réplica do bootstrap
paramétrico da MV), de modo que a incerteza das forças entra na tabela final.
Desempate: pontos, vitórias, saldo, gols pró; o resto (confronto direto, cartões) vira sorteio.
"""
import sys

import numpy as np
import pandas as pd

import dados
import modelo as M

N_SIM = 20_000
TEMPORADA = 2026


def simular(par, jogados, restantes, rng):
    times = sorted(set(jogados.mandante) | set(jogados.visitante))
    ix = {t: i for i, t in enumerate(times)}
    base = dados.classificacao(jogados).reindex(times)
    S = len(par["mu0"])
    m = restantes.mandante.map(ix).to_numpy(); v = restantes.visitante.map(ix).to_numpy()
    pts = np.tile(base.pts.to_numpy(float), (N_SIM, 1)); vit = np.tile(base.v.to_numpy(float), (N_SIM, 1))
    sg = np.tile(base.sg.to_numpy(float), (N_SIM, 1)); gp = np.tile(base.gp.to_numpy(float), (N_SIM, 1))
    amostra = rng.integers(0, S, N_SIM)
    for k in np.unique(amostra):
        linhas = np.flatnonzero(amostra == k)
        lam, mu = M._medias(par, restantes.mandante, restantes.visitante, k)
        mat = M.matriz_placar(lam, mu, par["rho"][k]).reshape(len(lam), -1)
        acum = mat.cumsum(axis=1)
        u = rng.random((len(linhas), len(lam), 1))
        cel = np.minimum((acum[None] < u).sum(axis=2), mat.shape[1] - 1)
        x, y = cel // (M.MAX_GOLS + 1), cel % (M.MAX_GOLS + 1)
        for j in range(len(lam)):
            a, b = m[j], v[j]
            pts[linhas, a] += np.where(x[:, j] > y[:, j], 3, np.where(x[:, j] == y[:, j], 1, 0))
            pts[linhas, b] += np.where(y[:, j] > x[:, j], 3, np.where(x[:, j] == y[:, j], 1, 0))
            vit[linhas, a] += x[:, j] > y[:, j]; vit[linhas, b] += y[:, j] > x[:, j]
            sg[linhas, a] += x[:, j] - y[:, j]; sg[linhas, b] += y[:, j] - x[:, j]
            gp[linhas, a] += x[:, j]; gp[linhas, b] += y[:, j]
    chave = pts * 1e9 + vit * 1e6 + (sg + 500) * 1e3 + gp + rng.random(pts.shape) * 0.5
    posicao = (-chave).argsort(axis=1).argsort(axis=1) + 1
    n = len(times)
    return pd.DataFrame({
        "pts_atual": base.pts.to_numpy(),
        "pts_esperados": pts.mean(axis=0).round(1),
        "campeao": (posicao == 1).mean(axis=0),
        "top4": (posicao <= 4).mean(axis=0),
        "rebaixado": (posicao > n - 4).mean(axis=0),
    }, index=times)


def main():
    xi = float(sys.argv[1])
    d = dados.carregar()
    jogados = d[d.temporada == TEMPORADA]
    restantes = dados.jogos_restantes(jogados)
    treino = d[d.temporada.isin([TEMPORADA - 1, TEMPORADA])]
    ref = jogados.data.max() + pd.Timedelta(days=1)
    p = M.preparar(treino, ref, xi)
    rng = np.random.default_rng(2026)

    fit = M.ajustar_mv(p)
    boot = M.bootstrap_mv(p, fit, n_boot=200)
    bayes = M.ajustar_bayes(p, amostras=1000)
    print(f"Stan: R-hat máximo {bayes['rhat_max']:.3f}, divergências {bayes['divergencias']}")
    tabelas = {"MV+bootstrap": simular(boot, jogados, restantes, rng),
               "Bayes": simular(bayes, jogados, restantes, rng)}
    saida = pd.concat(tabelas, axis=1)
    saida.to_csv(dados.RAIZ / "resultados" / f"simulacao_{TEMPORADA}_{ref.date()}.csv")
    pd.set_option("display.width", 200)
    for nome, t in tabelas.items():
        print(f"\n== {nome}: {N_SIM} simulações dos {len(restantes)} jogos restantes (dados até {jogados.data.max().date()}) ==")
        t = t.sort_values("pts_esperados", ascending=False)
        print(t.assign(**{c: t[c].map("{:.1%}".format) for c in ["campeao", "top4", "rebaixado"]}).to_string())


if __name__ == "__main__":
    main()
