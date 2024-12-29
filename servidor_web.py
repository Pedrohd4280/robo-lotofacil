from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import random
import time

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Habilita CORS

@app.route('/')
def root():
    return send_from_directory('frontend', 'index.html')

@app.route('/status')
def status():
    return jsonify({
        'status': 'ok',
        'dados_carregados': True,
        'modelos_carregados': True
    })

@app.route('/gerar-jogos', methods=['POST'])
def gerar_jogos():
    try:
        dados = request.get_json()
        quantidade = dados.get('cartoes', 1)
        
        # Simula o processamento
        time.sleep(1)
        
        jogos = []
        for _ in range(quantidade):
            # Gera 15 números aleatórios entre 1 e 25
            numeros = sorted(random.sample(range(1, 26), 15))
            # Gera uma probabilidade aleatória entre 60 e 90
            probabilidade = round(random.uniform(60, 90), 1)
            
            jogos.append({
                'numeros': numeros,
                'probabilidade': probabilidade
            })
        
        return jsonify({'jogos': jogos})
    except Exception as e:
        print(f"Erro ao gerar jogos: {e}")
        return jsonify({'erro': 'Falha ao gerar jogos'}), 500

@app.route('/<path:path>')
def static_files(path):
    try:
        return send_from_directory('frontend', path)
    except Exception as e:
        print(f"Erro ao servir arquivo {path}: {e}")
        return "Arquivo não encontrado", 404

if __name__ == "__main__":
    print("Iniciando servidor web...")
    print("Diretório atual:", os.getcwd())
    print("Arquivos na pasta frontend:", os.listdir('frontend'))
    app.run(port=8000, host='127.0.0.1', debug=True)