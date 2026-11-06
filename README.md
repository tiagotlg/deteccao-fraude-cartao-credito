# Detecção de fraude em cartão de crédito

Projeto do desafio da DIO: identificar transações fraudulentas com três modelos que usam o peso das classes, ajustar o limiar de decisão e explicar as previsões com SHAP.

Arquivos do projeto:

- `deteccao_fraude.ipynb`: notebook completo, já executado e com todas as saídas.
- `requirements.txt`: dependências.
- `README.md`: este arquivo.

## 1. O problema e por que o desbalanceamento muda a avaliação

O dataset tem **284.807 transações** de cartão de crédito. **99,83%** são legítimas e só **0,17%** são fraude. As colunas `V1` a `V28` já vêm transformadas por PCA, e também há `Time`, `Amount` e `Class` (1 = fraude).

Com as classes tão desbalanceadas, a avaliação se concentra na **classe fraude**:

- **Recall**: a proporção das fraudes reais que o modelo consegue pegar. É a métrica principal.
- **Precisão** e **F1** aparecem ao lado, porque recall alto com precisão muito baixa significa bloquear muitas transações legítimas.
- **Curva ROC** (AUC) e **curva precisão-recall** (average precision) dos três modelos.

## 2. Preparação dos dados

- Criação de `log_amount = log(1 + Amount)` para reduzir a assimetria do valor da transação: no `describe()`, `Amount` tem mediana de 22,00, média de 88,35 e máximo de 25.691,16.
- Variáveis usadas: `Time`, `V1`–`V28` e `log_amount`.
- Divisão treino/teste **estratificada** (80/20, `random_state=42`): 227.845 transações no treino (394 fraudes) e 56.962 no teste (98 fraudes). A proporção de fraude ficou em 0,1729% no treino e 0,1720% no teste.
- `StandardScaler` ajustado **só no treino** e depois aplicado ao teste.
- Para escolher o limiar sem usar o teste, o treino foi dividido de novo, também de forma estratificada: 182.276 linhas para ajustar os modelos (315 fraudes) e 45.569 para validação (79 fraudes).

## 3. Comparação entre os modelos

Os três modelos dão **mais peso à classe fraude**:

| Modelo | Configuração de peso |
|---|---|
| Regressão logística (baseline) | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"`, `n_jobs=-1`, `random_state=42` |
| XGBoost | `scale_pos_weight = 181.961 / 315 ≈ 577,7`, `random_state=42` |

Resultados no **conjunto de teste**, para a classe fraude, ordenados pelo recall:

| Modelo | Limiar | Recall | Precisão | F1 |
|---|---|---|---|---|
| Regressão logística | padrão (0,5) | **0,9082** | 0,0612 | 0,1146 |
| **Random Forest** | **ajustado (0,41)** | **0,8367** | **0,8913** | **0,8632** |
| XGBoost | padrão (0,5) | 0,8265 | 0,8710 | 0,8482 |
| Random Forest | padrão (0,5) | 0,8265 | 0,9101 | 0,8663 |
| Regressão logística | ajustado (≈ 1,0) | 0,8061 | 0,8316 | 0,8187 |
| XGBoost | ajustado (0,927) | 0,7959 | 0,9398 | 0,8619 |

Curvas no teste:

| Modelo | ROC AUC | Average precision |
|---|---|---|
| Regressão logística | 0,97 | 0,70 |
| Random Forest | 0,96 | 0,84 |
| XGBoost | 0,97 | 0,87 |

Principais pontos:

- A logística com limiar 0,5 tem o maior recall (0,9082), mas precisão de 0,0612: cerca de **1.365 transações legítimas** seriam marcadas como fraude para pegar 89 fraudes. Ajustar o limiar corrige isso (precisão 0,8316, recall 0,8061).
- Pela ROC os três modelos parecem equivalentes (AUC entre 0,96 e 0,97). A curva precisão-recall separa melhor: AP de 0,87 (XGBoost), 0,84 (Random Forest) e 0,70 (logística).
- **Modelo escolhido: Random Forest com limiar ajustado de 0,41**, por ter o maior recall da fraude com o limiar ajustado (desempate pelo F1). No teste ele encontrou **82 das 98 fraudes**, deixou passar 16 e gerou **10 falsos alarmes** em 56.864 transações legítimas.

## 4. Limiar de decisão e o que o SHAP mostrou

**Limiar.** Para cada modelo, o limiar foi escolhido na **validação** (nunca no teste), como o ponto da curva precisão-recall que maximiza o F1 da classe fraude:

| Modelo | Limiar escolhido | F1 na validação |
|---|---|---|
| Regressão logística | 0,99999994 | 0,7692 |
| Random Forest | **0,41** | 0,8378 |
| XGBoost | 0,927 | 0,8493 |

- **Regressão logística:** o limiar de 0,99999994 aparece arredondado como 1,0 nas tabelas. Ela só classifica uma transação como fraude quando a probabilidade calculada é praticamente 100%.
- **Random Forest:** o limiar de 0,41, um pouco abaixo de 0,5, faz o modelo marcar fraude com uma probabilidade menor do que o padrão. Ele tende a pegar mais fraudes, com o risco de mais alarmes falsos.
- **XGBoost:** o limiar de 0,927, bem acima de 0,5, faz o modelo só marcar fraude quando tem muita certeza. Ele tende a ter menos alarmes falsos, com o risco de deixar passar mais fraudes.
- Como a validação tem só 79 fraudes, o limiar escolhido é sensível a poucos casos.

**Importância das variáveis e SHAP (Random Forest).**

- As variáveis mais importantes (`feature_importances_`) são **V14, V10, V4, V12 e V17**.
- O SHAP (`TreeExplainer`, numa amostra de 2.000 transações do teste com todas as fraudes do teste) mostra a direção do efeito: valores **baixos** de V14, V12, V10, V3, V17 e V16 empurram a previsão para fraude, e valores **altos** de V4 e V11 também.
- Na transação de fraude analisada, as variáveis que mais pesaram na decisão foram, nesta ordem: **V14, V10, V12, V17 e V4**.
- Como V1–V28 são componentes de PCA, sabemos quais componentes pesam, mas não o significado de negócio de cada um.

## Como rodar

Requisitos: Python 3 e acesso à internet, porque o notebook lê o CSV direto da URL e não salva nenhuma cópia em disco.

```bash
pip install -r requirements.txt
jupyter notebook deteccao_fraude.ipynb
```

Para executar o notebook inteiro pela linha de comando e salvar as saídas no próprio arquivo:

```bash
jupyter nbconvert --to notebook --execute --inplace deteccao_fraude.ipynb
```
