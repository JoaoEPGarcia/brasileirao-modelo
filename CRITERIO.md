# Critério de aprovação do teste retroativo

Fixado em 24/09/2026, antes de qualquer previsão do período de teste ter sido gerada.
Mudar qualquer item depois de ver o resultado cria uma versão nova, que é relatada como tal
e não herda a aprovação desta.

## Desenho do teste

- **Período de teste:** temporadas 2016 a 2024 da Série A (9 temporadas).
- **Aquecimento:** 2012 a 2015. Só serve para escolher o decaimento temporal (abaixo);
  nenhuma aposta dessas temporadas conta para o critério.
- **Previsão fora da amostra, semana a semana:** os jogos de cada bloco de 7 dias corridos são
  previstos com um modelo ajustado só com jogos anteriores ao primeiro dia do bloco.
- **Dados de ajuste:** jogos da temporada anterior e da corrente, com peso exp(-xi × dias).
- **Decaimento xi:** escolhido na grade {0; 0,001; 0,002; 0,003; 0,005; 0,0075; 0,01} por dia,
  pelo menor RPS médio do Dixon-Coles de máxima verossimilhança nas temporadas 2013 a 2015.
  O mesmo xi vale para os dois métodos.
- **Métodos:** (1) Dixon-Coles por máxima verossimilhança, com penalização ridge leve nas
  forças dos times, que é o que torna o time recém-promovido estimável no início da temporada;
  (2) Dixon-Coles bayesiano hierárquico em Stan (MCMC).
- **Odd da aposta simulada:** fechamento da Pinnacle (colunas PSCH, PSCD, PSCA). Jogo sem
  essa odd fica fora.

## Regra de aposta

- Vantagem de um resultado k (casa, empate, fora): v_k = p_k × odd_k − 1.
- Aposta-se em k só se **0,05 ≤ v_k ≤ 0,25 nos dois métodos**. Se mais de um resultado do
  mesmo jogo passar, fica o de maior vantagem média entre os métodos.
- Para o critério, a aposta é de valor unitário. O Kelly fracionado (um quarto do Kelly, com teto de
  2,5% da banca) é simulado à parte e não entra na aprovação.

## Aprovação

O modelo é **aprovado** para apostar em 2026 só se as duas condições valerem:

1. ROI total das apostas de 2016 a 2024 positivo, com o limite inferior do intervalo de 95%
   acima de zero. O intervalo é um bootstrap de 10.000 reamostragens por bloco de 7 dias,
   para respeitar a dependência entre jogos da mesma rodada.
2. ROI positivo em pelo menos 6 das 9 temporadas.

Qualquer outro resultado é **reprovação**, e não se aposta nesta temporada.

## Informação, fora do critério

- RPS e log-loss de cada método e da probabilidade sem margem da Pinnacle, por temporada.
  O método de menor RPS médio em 2016 a 2024 é o que fornece as probabilidades de 2026.
- Em 2025 e 2026: o mesmo teste na Betfair Exchange, com a odd líquida de comissão,
  1 + (odd − 1) × (1 − 0,065), e na bet365.
