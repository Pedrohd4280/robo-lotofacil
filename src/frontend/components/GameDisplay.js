import React from 'react';
import { styles } from '../styles';
import { motion } from 'framer-motion';

const GameDisplay = ({ games }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            {games.map((game, index) => (
                <motion.div
                    key={index}
                    style={styles.gameContainer}
                    initial={{ x: -100, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                >
                    <h3 style={{color: '#e94560', marginBottom: '15px'}}>Jogo {index + 1}</h3>
                    <div>
                        {game.map((number, idx) => (
                            <motion.span
                                key={idx}
                                style={styles.number}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                {number.toString().padStart(2, '0')}
                            </motion.span>
                        ))}
                    </div>
                </motion.div>
            ))}
        </motion.div>
    );
};

export default GameDisplay; 