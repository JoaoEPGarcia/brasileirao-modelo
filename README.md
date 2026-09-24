# Brasileirão: modelo de previsão e teste contra o mercado

Dixon-Coles com decaimento temporal em dois métodos (máxima verossimilhança com bootstrap
paramétrico, e bayesiano hierárquico em Stan), testado contra a odd de fechamento da Pinnacle.
A regra de aprovação está em `CRITERIO.md` e foi fixada antes de qualquer resultado.

## Rodar

    .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import dados; dados.baixar()"
    cd src
    ../.venv/Scripts/python.exe backtest.py xi          # decaimento, só no aquecimento
    ../.venv/Scripts/python.exe backtest.py teste 0.003 # previsões 2016-2026 (cache em resultados/cache)
    ../.venv/Scripts/python.exe avaliar.py 0.003        # aplica o CRITERIO.md
    ../.venv/Scripts/python.exe simular.py 0.003        # título, G4 e rebaixamento de 2026

Depois de cada rodada: baixar o CSV de novo, apagar `resultados/cache/prev_2026_*.csv` e
rodar `simular.py`; anotar em `registros/odds_longo_prazo.csv` as odds das três casas
(Pinnacle, Betfair Exchange, bet365) ao lado das probabilidades do modelo, da UFMG e do Infobola.

## Fontes

- Jogos e odds de fechamento: https://www.football-data.co.uk/new/BRA.csv
- Referências publicadas: https://www.mat.ufmg.br/futebol/ e https://www.infobola.com.br/
