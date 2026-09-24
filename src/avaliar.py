"""Aplica o CRITERIO.md às previsões do backtest.   python src/avaliar.py XI"""
import sys

import numpy as np
import pandas as pd

import dados
from backtest import CACHE, rps

MIN_V, MAX_V = 0.05, 0.25
COMISSAO_BF = 0.065
TESTE = range(2016, 2025)


def carregar(xi):
    prev = pd.concat([pd.read_csv(f, parse_dates=["data"]) for f in sorted(CACHE.glob(f"prev_*_xi{xi}.csv"))])
    d = dados.carregar()
    return prev.merge(d, on=["data", "temporada", "mandante", "visitante", "res"], how="left", validate="1:1")


def apostas(df, cols_odd):
    """Uma aposta unitária por jogo no máximo, segundo a regra do CRITERIO.md."""
    odd = df[cols_odd].to_numpy(float)
    ok_odd = ~np.isnan(odd).any(axis=1)
    v_mv = df[["mv_c", "mv_e", "mv_f"]].to_numpy() * odd - 1
    v_by = df[["by_c", "by_e", "by_f"]].to_numpy() * odd - 1
    passa = (v_mv >= MIN_V) & (v_mv <= MAX_V) & (v_by >= MIN_V) & (v_by <= MAX_V) & ok_odd[:, None]
    media = np.where(passa, (v_mv + v_by) / 2, -np.inf)
    k = media.argmax(axis=1)
    tem = passa.any(axis=1)
    res_k = np.array(["H", "D", "A"])[k]
    lucro = np.where(df.res.to_numpy() == res_k, odd[np.arange(len(df)), k] - 1, -1.0)
    out = df.loc[tem, ["data", "temporada", "mandante", "visitante", "res"]].copy()
    out["aposta"], out["odd"], out["lucro"] = res_k[tem], odd[tem, k[tem]], lucro[tem]
    out["p_by"] = df[["by_c", "by_e", "by_f"]].to_numpy()[tem, k[tem]]
    return out


def ic_bootstrap(ap, n=10_000, semente=7):
    ini = ap.groupby("temporada").data.transform("min")
    bloco = ap.temporada.astype(str) + "_" + ((ap.data - ini).dt.days // 7).astype(str)
    g = ap.groupby(bloco).lucro.agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(g), (n, len(g)))
    roi = g[idx, 0].sum(axis=1) / g[idx, 1].sum(axis=1)
    return np.percentile(roi, [2.5, 97.5])


def kelly(ap, fracao=0.25, teto=0.025):
    banca = 1.0
    for _, a in ap.sort_values("data").iterrows():
        f = min(teto, fracao * (a.p_by * a.odd - 1) / (a.odd - 1))
        banca += banca * f * a.lucro
    return banca


def resumo(xi):
    """Todos os números do critério e da informação complementar, num dicionário."""
    df = carregar(xi)
    df["pin_c"], df["pin_e"], df["pin_f"] = dados.prob_sem_margem(df[["PSCH", "PSCD", "PSCA"]].to_numpy()).T
    df["avg_c"], df["avg_e"], df["avg_f"] = dados.prob_sem_margem(df[["AvgCH", "AvgCD", "AvgCA"]].to_numpy()).T
    o = np.stack([df.res == "H", df.res == "D", df.res == "A"], axis=1)
    for pre in ("mv", "by", "pin", "avg"):
        P = df[[f"{pre}_c", f"{pre}_e", f"{pre}_f"]].to_numpy()
        df[f"rps_{pre}"] = rps(P, df.res.to_numpy())
        df[f"ll_{pre}"] = -np.log(np.clip((P * o).sum(axis=1), 1e-12, None))
    teste = df[df.temporada.isin(TESTE) & df.PSCH.notna()]

    ap = apostas(teste, ["PSCH", "PSCD", "PSCA"])
    roi = ap.lucro.sum() / len(ap)
    lo, hi = ic_bootstrap(ap)
    por = ap.groupby("temporada").lucro.agg(n="size", roi="mean")
    pos = int((por.roi > 0).sum())
    c1, c2 = bool(lo > 0 and roi > 0), pos >= 6

    info = df[df.temporada.isin([2025, 2026])].copy()
    for c in "HDA":
        info[f"BFL{c}"] = 1 + (info[f"BFEC{c}"] - 1) * (1 - COMISSAO_BF)
    extra = {}
    for nome, cols in [("Betfair líquida", ["BFLH", "BFLD", "BFLA"]), ("bet365", ["B365CH", "B365CD", "B365CA"]),
                       ("Pinnacle", ["PSCH", "PSCD", "PSCA"])]:
        a = apostas(info, cols)
        if len(a):
            extra[nome] = (len(a), a.lucro.mean())
    return dict(
        rhat_max=df.by_rhat.max(), div_max=int(df.by_div.max()),
        qualidade={nome: (teste[f"rps_{pre}"].mean(), teste[f"ll_{pre}"].mean())
                   for pre, nome in [("mv", "MV"), ("by", "Bayes"), ("pin", "Pinnacle sem margem"),
                                     ("avg", "Média do mercado")]},
        rps_temporada=teste.groupby("temporada")[["rps_mv", "rps_by", "rps_pin"]].mean(),
        n_jogos=len(teste), n_apostas=len(ap), roi=roi, ic=(lo, hi), por_temporada=por,
        temporadas_positivas=pos, por_resultado=ap.aposta.value_counts().to_dict(),
        c1=c1, c2=c2, aprovado=c1 and c2, kelly=kelly(ap), extra_2025_26=extra,
        rps_2025_26={"MV": info.rps_mv.mean(), "Bayes": info.rps_by.mean(), "Média do mercado": info.rps_avg.mean()},
    )


def main():
    r = resumo(sys.argv[1])
    print(f"R-hat máximo nos ajustes bayesianos: {r['rhat_max']:.3f}; divergências: {r['div_max']} no pior ajuste")
    print("\nRPS e log-loss médios, 2016-2024 (menor é melhor):")
    for nome, (q_rps, q_ll) in r["qualidade"].items():
        print(f"  {nome:<22} RPS {q_rps:.5f}   log-loss {q_ll:.5f}")
    print("\nRPS por temporada:")
    print(r["rps_temporada"].round(4).to_string())
    lo, hi = r["ic"]
    por = r["por_temporada"]
    print(f"\nApostas 2016-2024 na Pinnacle de fechamento: {r['n_apostas']} em {r['n_jogos']} jogos")
    print(f"ROI total {r['roi']:+.2%}   IC 95% bootstrap por bloco de 7 dias [{lo:+.2%}, {hi:+.2%}]")
    print(por.assign(roi=por.roi.map("{:+.1%}".format)).to_string())
    print("\nApostas por resultado:", r["por_resultado"])
    print(f"\nCondição 1 (ROI > 0 e IC acima de zero): {'passa' if r['c1'] else 'REPROVA'}")
    print(f"Condição 2 (ROI > 0 em >= 6 de 9 temporadas: {r['temporadas_positivas']}): {'passa' if r['c2'] else 'REPROVA'}")
    print(f"VEREDITO: {'APROVADO' if r['aprovado'] else 'REPROVADO'}")
    print(f"(fora do critério) banca final no Kelly 1/4 com teto 2,5%, partindo de 1: {r['kelly']:.3f}")
    print("\nInformação, fora do critério (2025-2026):")
    for nome, (n, roi) in r["extra_2025_26"].items():
        print(f"  {nome:<16} {n:>4} apostas, ROI {roi:+.2%}")
    print("  RPS 2025-2026: " + "  ".join(f"{k} {v:.4f}" for k, v in r["rps_2025_26"].items()))


if __name__ == "__main__":
    main()
