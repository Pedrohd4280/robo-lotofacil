import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

def train_lotofacil_model():
    # Carrega o dataset da Lotofácil
    df = pd.read_csv('data/resultados_lotofacil.csv')
    
    # Prepara os dados
    X = df[['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 
            'bola6', 'bola7', 'bola8', 'bola9', 'bola10',
            'bola11', 'bola12', 'bola13', 'bola14', 'bola15']]
    
    # Cria um target mais significativo baseado na frequência dos números
    y = df.apply(lambda row: sum([1 for x in range(1, 26) if x in row[:15]]), axis=1)
    
    # Divide os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina o modelo com parâmetros otimizados
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Salva o modelo
    joblib.dump(model, 'models/lotofacil_model.joblib')
    
    return model 