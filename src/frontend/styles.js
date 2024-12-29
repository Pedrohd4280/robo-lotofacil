export const styles = {
    container: {
        backgroundColor: '#1a1a2e',  // Azul escuro mais elegante
        minHeight: '100vh',
        padding: '20px',
        fontFamily: "'Roboto', sans-serif",
    },
    header: {
        color: '#e94560',  // Vermelho mais suave
        textAlign: 'center',
        fontSize: '2.8em',
        marginBottom: '30px',
        textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
    },
    gameContainer: {
        backgroundColor: '#16213e',  // Azul médio
        borderRadius: '15px',
        padding: '25px',
        margin: '15px 0',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        transition: 'transform 0.2s',
        '&:hover': {
            transform: 'translateY(-5px)',
        },
    },
    number: {
        backgroundColor: '#e94560',  // Vermelho mais suave
        color: 'white',
        width: '45px',
        height: '45px',
        borderRadius: '50%',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '6px',
        fontSize: '20px',
        fontWeight: 'bold',
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
        transition: 'transform 0.2s',
        '&:hover': {
            transform: 'scale(1.1)',
        },
    },
    button: {
        padding: '12px 25px',
        fontSize: '18px',
        backgroundColor: '#e94560',
        color: 'white',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        marginBottom: '25px',
        transition: 'all 0.3s',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        '&:hover': {
            backgroundColor: '#ff4f7a',
            transform: 'translateY(-2px)',
        },
        '&:active': {
            transform: 'translateY(1px)',
        },
    }
} 