import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import random
import logging
import os
import sys
from datetime import datetime
import xgboost as xgb

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Variáveis globais
df = None
numeros = None
modelos = {}
metricas = None
scaler = None

def carregar_modelos():
    """Carrega os modelos treinados."""
    global modelos, metricas, scaler
    try:
        # Verifica se os arquivos existem
        arquivos_necessarios = ['modelo_rf.pkl', 'modelo_lgbm.pkl', 'modelo_xgb.model', 'estatisticas.pkl', 'scaler.pkl']
        for arquivo in arquivos_necessarios:
            if not os.path.exists(arquivo):
                logger.error(f"Arquivo não encontrado: {arquivo}")
                return False
        
        # Carrega cada modelo individualmente com tratamento de erro específico
        try:
            modelos['rf'] = joblib.load('modelo_rf.pkl')
            logger.info("Modelo Random Forest carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Random Forest: {e}")
            return False
        
        try:
            modelos['lgb'] = joblib.load('modelo_lgbm.pkl')
            logger.info("Modelo LightGBM carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo LightGBM: {e}")
            return False
        
        try:
            # Carrega o modelo XGBoost usando o formato binário nativo
            modelos['xgb'] = xgb.Booster()
            modelos['xgb'].load_model('modelo_xgb.model')
            logger.info("Modelo XGBoost carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo XGBoost: {e}")
            return False
        
        try:
            metricas = joblib.load('estatisticas.pkl')
            logger.info("Métricas carregadas com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar métricas: {e}")
            return False
        
        try:
            scaler = joblib.load('scaler.pkl')
            logger.info("Scaler carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar scaler: {e}")
            return False
        
        logger.info("Todos os modelos e métricas foram carregados com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {str(e)}")
        return False

def inicializar_dados():
    """Carrega dados históricos da Lotofácil."""
    global df, numeros
    try:
        arquivo_dados = 'Lotofacil-12_2024.csv'
        if not os.path.exists(arquivo_dados):
            logger.error(f"Arquivo de dados não encontrado: {arquivo_dados}")
            return False
            
        df = pd.read_csv(arquivo_dados)
        df = df.dropna()
        numeros = df.iloc[:, 2:17].values.astype(int)
        
        logger.info(f"Dados carregados com sucesso. Total de jogos: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return False

def criar_features_jogo(numeros):
    """Cria features para um único jogo."""
    try:
        # Features básicas
        pares = sum(1 for n in numeros if n % 2 == 0)
        impares = 15 - pares
        soma = sum(numeros)
        media = np.mean(numeros)
        desvio = np.std(numeros)
        
        # Features de distribuição
        qtd_1a5 = sum(1 for n in numeros if 1 <= n <= 5)
        qtd_6a10 = sum(1 for n in numeros if 6 <= n <= 10)
        qtd_11a15 = sum(1 for n in numeros if 11 <= n <= 15)
        qtd_16a20 = sum(1 for n in numeros if 16 <= n <= 20)
        qtd_21a25 = sum(1 for n in numeros if 21 <= n <= 25)
        
        # Features de sequência
        nums_sorted = sorted(numeros)
        sequencias = sum(1 for i in range(len(nums_sorted)-1) if nums_sorted[i+1] - nums_sorted[i] == 1)
        
        # Features de probabilidade condicional média
        prob_cond_media = []
        for i in range(len(numeros)):
            for j in range(i+1, len(numeros)):
                prob_cond_media.append(metricas['prob_condicionais'].get(numeros[i], {}).get(numeros[j], 0))
        prob_cond_avg = np.mean(prob_cond_media) if prob_cond_media else 0
        
        # Features de frequência histórica
        freq_historica = [metricas['freq_relativa'].get(n, 0) for n in numeros]
        freq_media = np.mean(freq_historica)
        freq_std = np.std(freq_historica)
        
        # Features de frequência recente
        freq_recente = [metricas['freq_relativa_recente'].get(n, 0) for n in numeros]
        freq_recente_media = np.mean(freq_recente)
        
        # Distância média entre números
        dist_media = np.mean([nums_sorted[i+1] - nums_sorted[i] for i in range(len(nums_sorted)-1)])
        
        features = np.array([[
            pares, impares, soma, media, desvio,
            qtd_1a5, qtd_6a10, qtd_11a15, qtd_16a20, qtd_21a25,
            sequencias, prob_cond_avg, freq_media, freq_std,
            freq_recente_media, dist_media
        ]])
        
        return scaler.transform(features)
    except Exception as e:
        logger.error(f"Erro ao criar features: {e}")
        return None

def calcular_probabilidade(numeros):
    """Calcula probabilidade real de um jogo baseado em múltiplos fatores."""
    try:
        # Prepara features do jogo
        features = criar_features_jogo(numeros)
        if features is None:
            return 15.0  # Valor default mais baixo em caso de erro
        
        # Previsões dos modelos
        previsoes = []
        pesos_modelos = {'rf': 0.4, 'lgb': 0.3, 'xgb': 0.3}
        
        for nome_modelo, modelo in modelos.items():
            try:
                if nome_modelo == 'xgb':
                    dtest = xgb.DMatrix(features)
                    prev = modelo.predict(dtest)[0]
                else:
                    prev = modelo.predict(features)[0]
                
                # Normaliza a previsão
                prev = max(0, min(1, prev))  # Limita entre 0 e 1
                previsoes.append(prev * pesos_modelos[nome_modelo])
            except Exception as e:
                logger.error(f"Erro ao fazer previsão com modelo {nome_modelo}: {e}")
                continue
        
        if not previsoes:
            return 15.0  # Retorna valor baixo se nenhum modelo funcionou
        
        # Média ponderada das previsões
        prob_modelos = sum(previsoes)
        
        # Análise de frequência dos números
        freq_historica = np.mean([metricas['freq_relativa'].get(n, 0) for n in numeros])
        freq_recente = np.mean([metricas['freq_relativa_recente'].get(n, 0) for n in numeros])
        
        # Análise de probabilidades condicionais
        prob_cond = []
        for i in range(len(numeros)):
            for j in range(i+1, len(numeros)):
                prob_cond.append(metricas['prob_condicionais'].get(numeros[i], {}).get(numeros[j], 0))
        prob_cond_media = np.mean(prob_cond) if prob_cond else 0
        
        # Penalizações
        penalizacoes = 1.0
        
        # 1. Penalização por sequências longas
        nums_sorted = sorted(numeros)
        sequencias = sum(1 for i in range(len(nums_sorted)-1) if nums_sorted[i+1] - nums_sorted[i] == 1)
        if sequencias > 4:
            penalizacoes *= 0.7
        elif sequencias > 3:
            penalizacoes *= 0.85
        
        # 2. Penalização por distribuição muito desigual
        distribuicao = [
            sum(1 for n in numeros if 1 <= n <= 5),
            sum(1 for n in numeros if 6 <= n <= 10),
            sum(1 for n in numeros if 11 <= n <= 15),
            sum(1 for n in numeros if 16 <= n <= 20),
            sum(1 for n in numeros if 21 <= n <= 25)
        ]
        if max(distribuicao) > 5:
            penalizacoes *= 0.6
        elif max(distribuicao) > 4:
            penalizacoes *= 0.8
        
        # 3. Penalização por desequilíbrio par/ímpar
        pares = sum(1 for n in numeros if n % 2 == 0)
        impares = 15 - pares
        if abs(pares - impares) > 5:
            penalizacoes *= 0.7
        elif abs(pares - impares) > 3:
            penalizacoes *= 0.85
        
        # 4. Penalização por soma total fora do padrão histórico
        soma = sum(numeros)
        if soma < 120 or soma > 220:
            penalizacoes *= 0.65
        
        # 5. Penalização por desvio padrão anormal
        desvio = np.std(numeros)
        if desvio < 5 or desvio > 9:
            penalizacoes *= 0.75
        
        # Cálculo da probabilidade final
        pesos = {
            'modelos': 0.35,
            'freq_historica': 0.25,
            'freq_recente': 0.25,
            'prob_condicional': 0.15
        }
        
        prob_final = (
            prob_modelos * pesos['modelos'] +
            freq_historica * pesos['freq_historica'] +
            freq_recente * pesos['freq_recente'] +
            prob_cond_media * pesos['prob_condicional']
        ) * penalizacoes
        
        # Converte para porcentagem e limita entre 15% e 75%
        prob_percentual = prob_final * 75  # Máximo de 75%
        prob_percentual = max(15, min(75, prob_percentual))
        
        return float(prob_percentual)
        
    except Exception as e:
        logger.error(f"Erro ao calcular probabilidade: {e}")
        return 15.0  # Valor default mais baixo em caso de erro

def gerar_jogo():
    """Gera um jogo da Lotofácil usando os modelos treinados."""
    try:
        melhor_prob = 0
        melhor_jogo = None
        tentativas = 0
        max_tentativas = 200  # Aumenta o número de tentativas
        
        # Gera vários jogos e escolhe o melhor
        while tentativas < max_tentativas:
            numeros = sorted(random.sample(range(1, 26), 15))
            prob = calcular_probabilidade(numeros)
            
            # Só aceita jogos com probabilidade acima de um limiar mínimo
            if prob > melhor_prob and prob > 35:  # Mínimo de 35%
                melhor_prob = prob
                melhor_jogo = numeros
                
                # Se encontrar um jogo muito bom, pode parar antes
                if prob > 65:
                    break
            
            tentativas += 1
        
        # Se não encontrou nenhum jogo bom, gera um aleatório
        if melhor_jogo is None:
            melhor_jogo = sorted(random.sample(range(1, 26), 15))
            melhor_prob = calcular_probabilidade(melhor_jogo)
        
        return {
            'numeros': melhor_jogo,
            'probabilidade': float(melhor_prob)
        }
    except Exception as e:
        logger.error(f"Erro ao gerar jogo: {e}")
        return None

@app.route('/status', methods=['GET'])
def status():
    """Endpoint para verificar o status do servidor."""
    return jsonify({
        'status': 'online',
        'dados_carregados': df is not None,
        'modelos_carregados': len(modelos) > 0,
        'total_jogos_historico': len(df) if df is not None else 0,
        'ultima_atualizacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/gerar-jogos', methods=['POST'])
def gerar_jogos():
    try:
        data = request.get_json()
        qtd_jogos = data.get('cartoes', 1)
        
        if not (1 <= qtd_jogos <= 10):
            return jsonify({'error': 'Quantidade de jogos deve ser entre 1 e 10'}), 400
        
        jogos = []
        for _ in range(qtd_jogos):
            jogo = gerar_jogo()
            if jogo:
                jogos.append(jogo)
        
        return jsonify({
            'jogos': jogos,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        logger.error(f"Erro ao gerar jogos: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not inicializar_dados() or not carregar_modelos():
        sys.exit(1)
    
    app.run(debug=False, port=5001, host='127.0.0.1')