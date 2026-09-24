"""PDF de resultados para compartilhar.   python src/relatorio.py XI

Chamado no fim de simular.py, então sai um PDF novo a cada simulação. A ordem das seções
é deliberada: o veredito do teste contra o mercado vem antes de qualquer probabilidade.
"""
import sys
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import avaliar
import dados

REGISTRO = dados.RAIZ / "registros" / "odds_longo_prazo.csv"
MERCADOS = {"campeao": "campeao", "top4": "top4", "g4": "top4", "rebaixado": "rebaixado",
            "rebaixamento": "rebaixado"}

est = getSampleStyleSheet()
TXT = ParagraphStyle("txt", parent=est["BodyText"], fontSize=9.5, leading=13)
PEQ = ParagraphStyle("peq", parent=TXT, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
H1 = ParagraphStyle("h1", parent=est["Title"], fontSize=17, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=est["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
CINZA = colors.HexColor("#DDDDDD")


def pct(x, casas=1):
    return f"{x * 100:.{casas}f}%".replace(".", ",")


def sinal(x, casas=1):
    return ("+" if x >= 0 else "-") + pct(abs(x), casas)


def num(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def tabela(linhas, larguras, destaque_col=None):
    t = Table(linhas, colWidths=larguras, repeatRows=1)
    estilo = [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
              ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F4F4")]),
              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    t.setStyle(TableStyle(estilo))
    return t


def caixa_veredito(r):
    lo, hi = r["ic"]
    cor = colors.HexColor("#1B7837") if r["aprovado"] else colors.HexColor("#B2182B")
    titulo = "APROVADO: há evidência de vantagem sobre o mercado" if r["aprovado"] else \
        "REPROVADO: não há evidência de vantagem sobre o mercado"
    corpo = (f"O modelo foi testado de 2016 a 2024 prevendo cada semana só com os jogos anteriores, "
             f"e apostando quando via pelo menos 5% de vantagem sobre a odd de fechamento da Pinnacle. "
             f"Foram {r['n_apostas']} apostas: ROI de <b>{sinal(r['roi'])}</b> "
             f"(intervalo de 95%: {sinal(lo)} a {sinal(hi)}), positivo em {r['temporadas_positivas']} de 9 temporadas. "
             f"A regra de aprovação foi fixada antes do teste (CRITERIO.md).")
    if not r["aprovado"]:
        corpo += (" <b>Por esta regra, não se aposta.</b> As probabilidades a seguir são estimativas "
                  "do campeonato, não palpites com vantagem.")
    t = Table([[Paragraph(f"<font color='white'><b>{titulo}</b></font>", TXT)], [Paragraph(corpo, TXT)]],
              colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), cor), ("BOX", (0, 0), (-1, -1), 1.2, cor),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    return t


def ler_simulacao():
    arqs = sorted((dados.RAIZ / "resultados").glob("simulacao_*.csv"))
    arq = arqs[-1]
    sim = pd.read_csv(arq, header=[0, 1], index_col=0)
    return sim, arq.stem.split("_")[-1]


def secao_esperanca(sim, historia):
    """Esperança por aposta com as odds anotadas; sem odds, só a explicação com um exemplo."""
    partes = [Paragraph("Vale a pena apostar? A conta da esperança", H2)]
    partes.append(Paragraph(
        "Numa aposta de R$ 1, a esperança é <b>p x odd - 1</b>. A odd justa é 1/p: abaixo dela a esperança "
        "é negativa. A casa sempre paga menos que a odd justa do próprio mercado (a margem), então, "
        "com a probabilidade do mercado, a esperança é negativa por construção. Ela só fica positiva se o "
        "nosso p for melhor que o do mercado, e o teste acima mostra que não é: o RPS do modelo é pior "
        "que o da Pinnacle. Uma esperança positiva calculada com este modelo é, na maioria das vezes, "
        "erro do modelo.", TXT))
    odds = pd.read_csv(REGISTRO) if REGISTRO.exists() else pd.DataFrame()
    odds = odds.dropna(subset=["odd"]) if len(odds) else odds
    if len(odds):
        ult = odds[odds.data_coleta == odds.data_coleta.max()]
        linhas = [["Time", "Mercado", "Casa", "Odd", "Prob. implícita", "p modelo", "Odd justa", "Esperança"]]
        positivas = 0
        for _, o in ult.iterrows():
            col = MERCADOS.get(str(o.mercado).lower())
            if col is None or o.time not in sim.index:
                continue
            p = sim.loc[o.time, ("MV+bootstrap", col)]
            odd = o.odd
            if "betfair" in str(o.casa).lower():  # a Exchange cobra comissão sobre o lucro
                odd = 1 + (o.odd - 1) * (1 - avaliar.COMISSAO_BF)
            positivas += p * odd - 1 > 0
            linhas.append([o.time, col, o.casa, num(odd), pct(1 / odd), pct(p),
                           num(1 / p) if p > 0 else "-", sinal(p * odd - 1)])
        partes.append(Spacer(1, 4))
        partes.append(Paragraph(f"Odds anotadas em {ult.data_coleta.iloc[0]} (Betfair já líquida da comissão):", TXT))
        partes.append(tabela(linhas, [3.2 * cm, 1.9 * cm, 2.3 * cm, 1.3 * cm, 2.2 * cm, 1.9 * cm, 1.8 * cm, 2 * cm]))
        if positivas:
            partes.append(Paragraph(
                "Uma esperança positiva nesta tabela não é recomendação de aposta. Ela diz que o modelo discorda do "
                "mercado, e no teste de 2016 a 2024, apostar nessas discordâncias deu prejuízo.", PEQ))
    else:
        p = sim.loc["Internacional", ("MV+bootstrap", "rebaixado")]
        partes.append(Paragraph(
            f"<i>Exemplo:</i> o modelo dá {pct(p, 0)} para o Internacional cair; a odd justa seria {num(1 / p)}. "
            f"Se a casa pagar 1,50, a esperança seria {pct(p, 0)} x 1,50 - 1 = {sinal(p * 1.5 - 1)}. "
            "Nenhuma odd foi anotada ainda em registros/odds_longo_prazo.csv; quando houver, esta seção "
            "mostra a conta para cada uma.", TXT))
    return partes


def gerar(xi):
    r = avaliar.resumo(xi)
    sim, data_ref = ler_simulacao()
    d = dados.carregar()
    ultimo_jogo = d[d.temporada == 2026].data.max().strftime("%d/%m/%Y")
    saida = dados.RAIZ / "resultados" / f"relatorio_brasileirao_{date.today()}.pdf"
    doc = SimpleDocTemplate(str(saida), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Brasileirão 2026: probabilidades e teste contra o mercado")
    h = [Paragraph("Brasileirão 2026: probabilidades e teste contra o mercado", H1),
         Paragraph(f"Gerado em {date.today().strftime('%d/%m/%Y')}, com jogos até {ultimo_jogo}. "
                   "Modelo Dixon-Coles com decaimento temporal, em dois métodos: máxima verossimilhança "
                   "com bootstrap e bayesiano (Stan).", PEQ), Spacer(1, 8), caixa_veredito(r)]

    h.append(Paragraph("Título, G4 e rebaixamento", H2))
    mv, by = sim["MV+bootstrap"], sim["Bayes"]
    ordem = mv.sort_values("pts_esperados", ascending=False).index
    linhas = [["Time", "Pts", "Pts esperados", "Campeão", "G4", "Rebaixado", "Rebaixado (Bayes)"]]
    for t in ordem:
        m = mv.loc[t]
        linhas.append([t, int(m.pts_atual), num(m.pts_esperados, 1), pct(m.campeao), pct(m.top4),
                       pct(m.rebaixado), pct(by.loc[t, "rebaixado"])])
    h.append(tabela(linhas, [3.6 * cm, 1.2 * cm, 2.3 * cm, 2 * cm, 2 * cm, 2.2 * cm, 3 * cm]))
    h.append(Paragraph("20.000 simulações dos jogos restantes; cada simulação sorteia também as forças dos times, "
                       "para a incerteza da estimativa entrar no resultado. Desempate por vitórias, saldo e gols pró.",
                       PEQ))

    h += secao_esperanca(sim, r)

    det = [Paragraph("Detalhe do teste contra o mercado", H2)]
    por = r["por_temporada"]
    linhas = [["Temporada", "Apostas", "ROI"]] + [[str(s), int(x.n), sinal(x.roi)] for s, x in por.iterrows()]
    linhas.append(["2016-2024", r["n_apostas"], sinal(r["roi"])])
    det.append(tabela(linhas, [3 * cm, 2.5 * cm, 2.5 * cm]))
    q = r["qualidade"]
    det.append(Spacer(1, 4))
    det.append(Paragraph(
        f"Qualidade da previsão jogo a jogo (RPS, menor é melhor): modelo {num(q['MV'][0], 4)}, "
        f"bayesiano {num(q['Bayes'][0], 4)}, Pinnacle {num(q['Pinnacle sem margem'][0], 4)}. "
        f"Apostando um quarto do Kelly, a banca terminaria em {pct(r['kelly'], 0)} do valor inicial.", TXT))
    ex = r["extra_2025_26"]
    if ex:
        txt = "; ".join(f"{k}: {n} apostas, ROI {sinal(v)}" for k, (n, v) in ex.items())
        det.append(Paragraph(
            f"Fora do critério, em 2025 e 2026 ({txt}). Amostra pequena: no teste, temporadas isoladas "
            f"variaram de {sinal(por.roi.max(), 0)} a {sinal(por.roi.min(), 0)}, então um período bom "
            "não indica vantagem.", PEQ))
    h.append(KeepTogether(det))
    doc.build(h)
    return saida


if __name__ == "__main__":
    print(gerar(sys.argv[1]))
