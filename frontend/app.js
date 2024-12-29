document.addEventListener('DOMContentLoaded', () => {
    const btnGerar = document.getElementById('gerarJogos');
    const resultadosDiv = document.getElementById('resultados');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const API_URL = 'http://127.0.0.1:8000';  // URL do backend atualizada

    // Timer
    const timerDiv = document.querySelector('.timer');
    let timerInterval;
    let startTime;

    function atualizarTimer() {
        const tempoDecorrido = (Date.now() - startTime) / 1000;
        timerDiv.textContent = `Tempo: ${tempoDecorrido.toFixed(1)}s`;
    }

    function iniciarTimer() {
        startTime = Date.now();
        timerDiv.classList.add('active');
        timerInterval = setInterval(atualizarTimer, 100);
    }

    function pararTimer() {
        clearInterval(timerInterval);
        timerDiv.classList.remove('active');
    }

    // Verifica status do servidor ao carregar
    async function verificarStatusServidor() {
        try {
            console.log('Verificando status do servidor...');
            const response = await fetch(`${API_URL}/status`);
            if (!response.ok) throw new Error('Servidor não está respondendo');
            
            const data = await response.json();
            console.log('Status do servidor:', data);
            
            if (!data.dados_carregados || !data.modelos_carregados) {
                mostrarErro('Aviso: Sistema não está completamente inicializado.');
                btnGerar.disabled = true;
            }
        } catch (error) {
            console.error('Erro ao verificar status:', error);
            mostrarErro('Erro: Servidor não está disponível. Verifique se o backend está rodando.');
            btnGerar.disabled = true;
        }
    }

    function mostrarErro(mensagem, isTemporary = true) {
        console.log('Mostrando erro:', mensagem);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = mensagem;
        
        if (document.querySelector('.error-message')) {
            document.querySelector('.error-message').remove();
        }
        
        resultadosDiv.insertAdjacentElement('beforebegin', errorDiv);
        
        if (isTemporary) {
            setTimeout(() => errorDiv.remove(), 5000);
        }
    }

    async function gerarJogos() {
        const cartoes = parseInt(document.getElementById('cartoes').value);
        console.log('Gerando', cartoes, 'jogos...');
        
        try {
            loadingSpinner.style.display = 'block';
            resultadosDiv.innerHTML = '';
            btnGerar.disabled = true;
            iniciarTimer();

            console.log('Fazendo requisição para:', `${API_URL}/gerar-jogos`);
            const response = await fetch(`${API_URL}/gerar-jogos`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ cartoes })
            });

            if (!response.ok) {
                throw new Error('Falha ao gerar jogos');
            }

            const data = await response.json();
            console.log('Jogos gerados:', data);
            exibirJogos(data.jogos);
        } catch (error) {
            console.error('Erro:', error);
            mostrarErro(`Erro ao gerar jogos: ${error.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
            btnGerar.disabled = false;
            pararTimer();
        }
    }

    function exibirJogos(jogos) {
        console.log('Exibindo jogos:', jogos);
        resultadosDiv.innerHTML = '';
        
        const cartoesContainer = document.createElement('div');
        cartoesContainer.className = 'cartoes-container';

        jogos.forEach((jogo, index) => {
            const cartao = document.createElement('div');
            cartao.className = 'cartao';
            
            cartao.innerHTML = `
                <div class="cartao-header">
                    <span class="cartao-titulo">Jogo ${index + 1}</span>
                    <span class="probabilidade">${jogo.probabilidade.toFixed(1)}% chance</span>
                </div>
                <div class="numeros">
                    ${jogo.numeros.map(num => `
                        <span class="numero">${String(num).padStart(2, '0')}</span>
                    `).join('')}
                </div>
            `;

            cartoesContainer.appendChild(cartao);
        });

        resultadosDiv.appendChild(cartoesContainer);
    }

    // Adiciona event listeners
    btnGerar.addEventListener('click', gerarJogos);
    verificarStatusServidor();
    
    console.log('Frontend inicializado');
});
