"""Leitura do BRA.csv do football-data.co.uk e utilidades de tabela."""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRUTO = RAIZ / "data" / "raw" / "BRA.csv"
URL = "https://www.football-data.co.uk/new/BRA.csv"


def baixar():
    import requests
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    BRUTO.write_bytes(r.content)


def carregar():
    d = pd.read_csv(BRUTO, encoding="utf-8-sig")
    d["data"] = pd.to_datetime(d["Date"], dayfirst=True)
    d = d.rename(columns={"Season": "temporada", "Home": "mandante", "Away": "visitante",
                          "HG": "gm", "AG": "gv", "Res": "res"})
    d = d.dropna(subset=["gm", "gv"]).copy()
    d["gm"] = d["gm"].astype(int)
    d["gv"] = d["gv"].astype(int)
    return d.sort_values(["data", "mandante"]).reset_index(drop=True)


def classificacao(jogos):
    """Pontos, vitórias, saldo e gols pró por time (critérios 1 a 4 da CBF)."""
    linhas = []
    for _, j in jogos.iterrows():
        pm = 3 if j.gm > j.gv else (1 if j.gm == j.gv else 0)
        pv = 3 if j.gv > j.gm else (1 if j.gm == j.gv else 0)
        linhas.append((j.mandante, pm, int(pm == 3), j.gm - j.gv, j.gm, 1))
        linhas.append((j.visitante, pv, int(pv == 3), j.gv - j.gm, j.gv, 1))
    t = pd.DataFrame(linhas, columns=["time", "pts", "v", "sg", "gp", "j"])
    t = t.groupby("time").sum()
    return t.sort_values(["pts", "v", "sg", "gp"], ascending=False)


def jogos_restantes(temporada_jogos):
    """Turno e returno completos: cada par ordenado (mandante, visitante) joga uma vez."""
    times = sorted(set(temporada_jogos.mandante) | set(temporada_jogos.visitante))
    feitos = set(zip(temporada_jogos.mandante, temporada_jogos.visitante))
    return pd.DataFrame([(a, b) for a in times for b in times if a != b and (a, b) not in feitos],
                        columns=["mandante", "visitante"])


def prob_sem_margem(odds):
    """Probabilidade implícita normalizada (remove a margem proporcionalmente)."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    d = carregar()
    s = d[d.temporada == 2026]
    print(classificacao(s))
    r = jogos_restantes(s)
    print("restantes:", len(r))
