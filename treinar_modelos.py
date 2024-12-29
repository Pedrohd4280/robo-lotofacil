import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import joblib
import logging
from collections import Counter
from datetime import datetime
import math
import warnings

# Ignorar avisos de depreciação
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calcular_metricas_avancadas(df):
    """Calcula métricas avançadas para análise dos números."""
    metricas = {}
    
    # Frequência absoluta e relativa
    numeros_flat = df[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                      'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values.flatten()
    freq = Counter(numeros_flat)
    total_jogos = len(df)
    
    # Frequência relativa normalizada
    freq_relativa = {num: (count/total_jogos) for num, count in freq.items()}
    
    # Análise de pares frequentes
    pares = []
    for _, row in df.iterrows():
        nums = row[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                   'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pares.append(tuple(sorted([nums[i], nums[j]])))
    
    freq_pares = Counter(pares)
    
    # Análise de sequências
    sequencias = []
    for _, row in df.iterrows():
        nums = sorted(row[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                         'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values)
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1:
                sequencias.append((nums[i], nums[i+1]))
    
    freq_sequencias = Counter(sequencias)
    
    # Análise temporal (sem depender do formato da data)
    ultimos_50_jogos = df.tail(50)
    freq_recente = Counter(ultimos_50_jogos[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                                           'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values.flatten())
    
    # Tendências temporais
    freq_relativa_recente = {num: (count/50) for num, count in freq_recente.items()}
    
    # Análise de quadrantes
    def get_quadrante(num):
        if 1 <= num <= 5: return 1
        elif 6 <= num <= 10: return 2
        elif 11 <= num <= 15: return 3
        elif 16 <= num <= 20: return 4
        else: return 5
    
    dist_quadrantes = []
    for _, row in df.iterrows():
        nums = row[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                   'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values
        quad = [get_quadrante(n) for n in nums]
        dist_quadrantes.append(Counter(quad))
    
    quad_medio = {q: sum(d[q] for d in dist_quadrantes)/len(dist_quadrantes) for q in range(1,6)}
    
    # Cálculo de probabilidades condicionais
    prob_condicionais = {}
    for num1 in range(1, 26):
        prob_condicionais[num1] = {}
        for num2 in range(1, 26):
            if num1 != num2:
                # Conta quantas vezes num2 aparece quando num1 está presente
                jogos_com_num1 = df[df[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                                      'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].isin([num1]).any(axis=1)]
                if len(jogos_com_num1) > 0:
                    prob = len(jogos_com_num1[jogos_com_num1[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                                                             'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].isin([num2]).any(axis=1)]) / len(jogos_com_num1)
                    prob_condicionais[num1][num2] = prob
    
    metricas['freq_relativa'] = freq_relativa
    metricas['freq_pares'] = freq_pares
    metricas['freq_sequencias'] = freq_sequencias
    metricas['freq_relativa_recente'] = freq_relativa_recente
    metricas['quad_medio'] = quad_medio
    metricas['prob_condicionais'] = prob_condicionais
    
    return metricas

def criar_features_avancadas(df, metricas):
    """Cria features mais sofisticadas para os modelos."""
    features = []
    
    for _, row in df.iterrows():
        nums = row[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                   'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values
        
        # Features básicas
        pares = sum(1 for n in nums if n % 2 == 0)
        impares = 15 - pares
        soma = sum(nums)
        media = np.mean(nums)
        desvio = np.std(nums)
        
        # Features de distribuição por quadrante
        qtd_1a5 = sum(1 for n in nums if 1 <= n <= 5)
        qtd_6a10 = sum(1 for n in nums if 6 <= n <= 10)
        qtd_11a15 = sum(1 for n in nums if 11 <= n <= 15)
        qtd_16a20 = sum(1 for n in nums if 16 <= n <= 20)
        qtd_21a25 = sum(1 for n in nums if 21 <= n <= 25)
        
        # Features de sequência
        nums_sorted = sorted(nums)
        sequencias = sum(1 for i in range(len(nums_sorted)-1) if nums_sorted[i+1] - nums_sorted[i] == 1)
        
        # Features de probabilidade condicional média
        prob_cond_media = []
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                prob_cond_media.append(metricas['prob_condicionais'].get(nums[i], {}).get(nums[j], 0))
        prob_cond_avg = np.mean(prob_cond_media) if prob_cond_media else 0
        
        # Features de frequência histórica
        freq_historica = [metricas['freq_relativa'].get(n, 0) for n in nums]
        freq_media = np.mean(freq_historica)
        freq_std = np.std(freq_historica)
        
        # Features de frequência recente
        freq_recente = [metricas['freq_relativa_recente'].get(n, 0) for n in nums]
        freq_recente_media = np.mean(freq_recente)
        
        # Distância média entre números
        dist_media = np.mean([nums_sorted[i+1] - nums_sorted[i] for i in range(len(nums_sorted)-1)])
        
        features.append([
            pares, impares, soma, media, desvio,
            qtd_1a5, qtd_6a10, qtd_11a15, qtd_16a20, qtd_21a25,
            sequencias, prob_cond_avg, freq_media, freq_std,
            freq_recente_media, dist_media
        ])
    
    return np.array(features)

def carregar_dados():
    """Carrega e prepara os dados da Lotofácil com análise avançada."""
    try:
        df = pd.read_csv('Lotofacil-12_2024.csv', 
                        names=['Concurso', 'Data', 'D.1', 'D.2', 'D.3', 'D.4', 'D.5',
                              'D.6', 'D.7', 'D.8', 'D.9', 'D.10', 'D.11', 'D.12',
                              'D.13', 'D.14', 'D.15'],
                        skiprows=1)
        
        # Análise estatística avançada
        metricas = calcular_metricas_avancadas(df)
        
        # Criar features avançadas
        X = criar_features_avancadas(df, metricas)
        
        # Target: probabilidade ponderada baseada em múltiplos fatores
        y = []
        for _, row in df.iterrows():
            nums = row[['D.1', 'D.2', 'D.3', 'D.4', 'D.5', 'D.6', 'D.7', 
                       'D.8', 'D.9', 'D.10', 'D.11', 'D.12', 'D.13', 'D.14', 'D.15']].values
            
            # Média ponderada de diferentes métricas
            freq_hist = np.mean([metricas['freq_relativa'].get(n, 0) for n in nums])
            freq_rec = np.mean([metricas['freq_relativa_recente'].get(n, 0) for n in nums])
            
            # Calcula probabilidade baseada em múltiplos fatores
            prob = (freq_hist * 0.4 + freq_rec * 0.6)  # Dá mais peso para tendências recentes
            y.append(prob)
        
        y = np.array(y)
        
        # Normalização
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Salva o scaler e métricas
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(metricas, 'estatisticas.pkl')
        
        logger.info(f"Dados carregados com sucesso. Total de registros: {len(df)}")
        return X, y
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None, None

def treinar_random_forest():
    """Treina e salva o modelo Random Forest com configurações otimizadas."""
    try:
        X, y = carregar_dados()
        if X is None:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(
            n_estimators=500,  # Aumentado para maior robustez
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # Validação cruzada
        scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        logger.info(f"Scores de validação cruzada RF: {scores}")
        
        rf_model.fit(X_train, y_train)
        score = rf_model.score(X_test, y_test)
        logger.info(f"Score do Random Forest: {score:.4f}")
        
        joblib.dump(rf_model, 'modelo_rf.pkl')
        logger.info("Modelo Random Forest treinado e salvo com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao treinar Random Forest: {e}")
        return False

def treinar_lightgbm():
    """Treina e salva o modelo LightGBM com configurações otimizadas."""
    try:
        X, y = carregar_dados()
        if X is None:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=15,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Validação cruzada
        scores = cross_val_score(lgb_model, X_train, y_train, cv=5)
        logger.info(f"Scores de validação cruzada LightGBM: {scores}")
        
        lgb_model.fit(X_train, y_train)
        score = lgb_model.score(X_test, y_test)
        logger.info(f"Score do LightGBM: {score:.4f}")
        
        joblib.dump(lgb_model, 'modelo_lgbm.pkl')
        logger.info("Modelo LightGBM treinado e salvo com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao treinar LightGBM: {e}")
        return False

def treinar_xgboost():
    """Treina e salva o modelo XGBoost com configurações otimizadas."""
    try:
        X, y = carregar_dados()
        if X is None:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Usando DMatrix do XGBoost diretamente
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Parâmetros do modelo
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'eta': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Treinamento
        num_rounds = 500
        xgb_model = xgb.train(params, dtrain, num_rounds)
        
        # Avaliação
        pred = xgb_model.predict(dtest)
        score = 1 - np.mean((y_test - pred) ** 2) / np.var(y_test)
        logger.info(f"Score do XGBoost: {score:.4f}")
        
        # Salvando o modelo no formato binário nativo do XGBoost
        xgb_model.save_model('modelo_xgb.model')
        logger.info("Modelo XGBoost treinado e salvo com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao treinar XGBoost: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando treinamento dos modelos...")
    treinar_random_forest()
    treinar_lightgbm()
    treinar_xgboost()