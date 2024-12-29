import random
import joblib
import numpy as np
from itertools import combinations
import pandas as pd

def analyze_frequency(games_history):
    frequency = {}
    for i in range(1, 26):
        frequency[i] = sum(1 for game in games_history if i in game)
    return frequency

def generate_lotofacil_games(num_games=10):
    model = joblib.load('models/lotofacil_model.joblib')
    games = []
    attempts = 0
    max_attempts = 1000
    
    # Carrega histórico para análise
    df = pd.read_csv('data/resultados_lotofacil.csv')
    historical_games = df[['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 
                          'bola6', 'bola7', 'bola8', 'bola9', 'bola10',
                          'bola11', 'bola12', 'bola13', 'bola14', 'bola15']].values.tolist()
    
    frequency = analyze_frequency(historical_games)
    
    while len(games) < num_games and attempts < max_attempts:
        # Gera números com base na frequência
        weights = [frequency[i] for i in range(1, 26)]
        numbers = sorted(np.random.choice(range(1, 26), 15, replace=False, p=np.array(weights)/sum(weights)))
        
        # Avalia com o modelo
        prediction = model.predict_proba([numbers])
        
        # Verifica se o jogo é único e tem boa probabilidade
        if prediction[0][1] > 0.5 and numbers not in games and numbers not in historical_games:
            games.append(numbers)
        
        attempts += 1
    
    return games 