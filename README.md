# 🔍 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

> Projeto de machine learning para detecção de transações fraudulentas em cartões de crédito, com foco em boas práticas de engenharia, análise exploratória aprofundada e comparação sistemática de modelos.

---

## 📌 Visão Geral

Fraudes em cartões de crédito causam bilhões de dólares em prejuízo anualmente. Este projeto aborda o problema como um **classificador binário com classes altamente desbalanceadas** (~0.17% de fraudes), aplicando técnicas reais usadas em produção.

**Dataset:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
284.807 transações | 492 fraudes | 30 features (PCA anonimizadas)

---

## 🗂️ Estrutura do Projeto

```
fraud-detection/
│
├── data/                        # Dataset (não versionado)
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb    # Notebook principal
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Pipeline de pré-processamento
│   ├── evaluation.py            # Métricas customizadas
│   └── visualization.py         # Funções de visualização
│
├── models/                      # Modelos serializados (.pkl)
├── reports/
│   └── figures/                 # Gráficos exportados
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Pipeline do Projeto

```
1. Análise Exploratória (EDA)
   ├── Distribuição de classes
   ├── Análise temporal das fraudes
   └── Correlações e outliers

2. Pré-processamento
   ├── Escalonamento (RobustScaler)
   ├── Tratamento de desbalanceamento (SMOTE)
   └── Divisão treino/validação/teste

3. Feature Engineering
   ├── Criação de features baseadas em Amount e Time
   └── Seleção por importância

4. Modelagem
   ├── Logistic Regression (baseline)
   ├── Random Forest
   ├── XGBoost
   └── LightGBM

5. Otimização
   └── Optuna (hyperparameter tuning)

6. Avaliação Final
   ├── ROC-AUC, PR-AUC, F1, Recall
   └── Matriz de confusão + análise de threshold
```

---

## 📊 Resultados

| Modelo              | ROC-AUC | PR-AUC | F1-Score | Recall (Fraude) |
|---------------------|---------|--------|----------|-----------------|
| Logistic Regression | ~0.974  | ~0.712 | ~0.734   | ~0.761          |
| Random Forest       | ~0.984  | ~0.836 | ~0.857   | ~0.823          |
| XGBoost             | ~0.987  | ~0.851 | ~0.869   | ~0.844          |
| **LightGBM (tuned)**| **~0.989** | **~0.864** | **~0.881** | **~0.859** |

> *Métricas aproximadas; resultados exatos no notebook.*

---

## 🚀 Como Reproduzir

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/fraud-detection.git
cd fraud-detection
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset
- Acesse: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Salve `creditcard.csv` na pasta `data/`

### 5. Execute o notebook
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## 🛠️ Tecnologias

- **Análise:** pandas, numpy, matplotlib, seaborn, plotly
- **ML:** scikit-learn, xgboost, lightgbm
- **Balanceamento:** imbalanced-learn (SMOTE)
- **Otimização:** optuna
- **Serialização:** joblib

---

## 📚 Conceitos Aplicados

- Classes desbalanceadas e por que acurácia não é a métrica certa
- Precision-Recall tradeoff em cenários de detecção de anomalias
- Cross-validation estratificada
- Ajuste de threshold de decisão
- Feature importance com SHAP values

---

## 📄 Licença

MIT © [Seu Nome](https://github.com/seu-usuario)
