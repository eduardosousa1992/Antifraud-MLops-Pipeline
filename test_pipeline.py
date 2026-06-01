import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_pipeline_structure():
    print("[QA CHECK] Iniciando validação estrutural de dados...")
    np.random.seed(42)
    dummy_data = np.random.normal(0, 1, (100, 31))
    df = pd.DataFrame(dummy_data, columns=[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class'])
    df['Class'] = np.random.choice([0, 1], size=100, p=[0.95, 0.05])
    
    assert df.shape[0] == 100, "Erro: Volumetria de linhas inconsistente."
    assert df.shape[1] == 31, "Erro: Quantidade de features corrompida."

def test_model_inference():
    print("[QA CHECK] Iniciando validação matemática do classificador...")
    X = np.random.normal(0, 1, (50, 29))
    y = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    y_proba = model.predict_proba(X)[:, 1]
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1), "Erro: Probabilidade fora do intervalo [0,1]."
