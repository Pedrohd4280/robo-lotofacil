import React, { useState } from 'react';
import GameDisplay from './components/GameDisplay';
import { styles } from './styles';
import { motion } from 'framer-motion';

function App() {
    const [games, setGames] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const generateGames = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/generate-games');
            if (!response.ok) throw new Error('Falha ao gerar jogos');
            const data = await response.json();
            setGames(data.games);
        } catch (error) {
            setError('Erro ao gerar jogos. Tente novamente.');
            console.error('Erro:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <motion.div 
            style={styles.container}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            <h1 style={styles.header}>Robô Lotofácil</h1>
            <motion.button 
                onClick={generateGames}
                style={styles.button}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={isLoading}
            >
                {isLoading ? 'Gerando...' : 'Gerar Jogos'}
            </motion.button>
            
            {error && (
                <div style={{color: '#e94560', textAlign: 'center', margin: '10px 0'}}>
                    {error}
                </div>
            )}
            
            <GameDisplay games={games} />
        </motion.div>
    );
}

export default App; 