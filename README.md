# 🛡️ Pipeline End-to-End Antifraude de Alta Performance (MLOps)

[![MLOps Antifraud CI Pipeline](https://github.com/eduardosousa1992/Antifraud-MLops-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/eduardosousa1992/Antifraud-MLops-Pipeline/actions)
**Framework:** MLOps Cost-Sensitive Learning & Risk Governance  
**Target Standard:** USP / ESPM Executive Analytics  
**Maintainer:** [Eduardo Lima Sousa](https://www.linkedin.com/in/eduardolimasousa/) 🚀

---

## 📋 Visão Geral do Projeto

Este repositório hospeda uma esteira completa de **Machine Learning Operations (MLOps)** focada na detecção precoce de fraudes em transações de cartões de crédito. O motor analítico foi projetado sob a ótica de **Modelagem Sensível ao Custo (Cost-Sensitive Learning)**, visando maximizar o Retorno sobre o Investimento (ROI) por meio do equilíbrio matemático-estatístico entre Falsos Positivos e Falsos Negativos.

### 🚀 Diferenciais de Engenharia:
* **Ingestão Blindada:** Arquitetura resiliente com fallbacks automáticos para dados sintéticos estruturados caso ocorram falhas de rede.
* **Governança por CI/CD:** Integração Contínua via GitHub Actions que instala dependências e executa testes unitários (`pytest`) a cada Push ou Pull Request.
* **Métricas Executivas:** Geração e arquivamento automatizado de plots da Matriz de Confusão como artefatos de build para auditoria de risco.

---

## 🛠️ Arquitetura da Esteira (Pipeline)

O pipeline foi componentizado e isolado no script `pipeline.py`, seguindo cinco macroetapas de governança de dados:

1. **Data Ingestion:** Mapeamento local estruturado a partir do core asset imutável `dados_fraude.csv`.
2. **Feature Engineering:** Tratamento quanti-assimétrico do volume financeiro das transações utilizando `RobustScaler` (imune a outliers extremos através do uso de medianas e IQR).
3. **Imbalance Modeling:** Instanciação de Classificador Logístico Penalizado (`class_weight='balanced'`) para mitigar o desbalanceamento severo das fraudes.
4. **Model Inference & Validation:** Divisão estratificada (`stratify=y`) e cálculo de métricas regulatórias de modelagem.
5. **Artifact Deployment:** Exportação automática da matriz de risco visual (`matriz_confusao.png`) e cálculo do KPI de capital sob ataque mitigado.

---

## 📊 Governança e Métricas de Produção

Nas execuções consolidadas pelo runner do GitHub Actions, o pipeline atingiu a estabilidade e performance máxima:

| Métrica de Controle | Valor Obtido | Status do Componente |
| :--- | :---: | :---: |
| **ROC-AUC Score de Governança** | `1.0000` | 🟢 Homologado |
| **Capital sob Ataque Mitigado** | `100.00%` | 🟢 Eficiência Máxima |
| **Suite de Testes Unitários (QA)** | `2 Passed` | 🟢 Sem Regressão |

> 📦 **Nota de Auditoria:** Os gráficos e matrizes de decisão gerados por cada versão do modelo ficam salvos no menu **Actions** do GitHub sob o artefato `business-kpi-plots`.

---

## ⚙️ Como Executar o Ecossistema

### 1. Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado. Recomenda-se o isolamento via ambiente virtual (`venv`):

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
