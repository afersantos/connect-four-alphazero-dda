import numpy as np

class ConnectFour:
    def __init__(self):
        # Características del juego
        self.row_count = 6 # Número de filas del tablero
        self.column_count = 7 # Número de columnas del tablero
        self.action_size = self.column_count # Espacio de acción
        self.in_a_row = 4 # Condición de victoria. 4 en raya
        
    def __repr__(self):
        return "ConnectFour"
    
    # Método para devolver estado inicial (tablero vacío)
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    # Método para devolver siguiente estado tras tomar una acción
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0)) # Posición de la ficha superior en la columna
        state[row, action] = player # Colocación de la fihca una posición arriba de la ficha superior
        return state
    
    # Método para devolver array con las columnas válidas (no completadas)
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    # Método para devolver booleano en función de sí ha terminado o no la partida con la última jugada realizda
    def check_win(self, state, action):
        if action == None: # El root node no se le asigna una acción
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    # Método para devolver recompensa (valor) y check fin de juego
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action): # Recompensa +1 y fin de juego si se hace 4 en raya
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0: # Recompensa 0 por empate y fin de juego
            return 0, True
        return 0, False # Continua el juego
    
    # Método para cambiar de jugador
    def get_opponent(self, player):
        return -player
    
    # Método para obtener la recompensa (valor) del oponente
    def get_opponent_value(self, value):
        return -value
    
    # Método para cambiar la perspectiva del tablero
    def change_perspective(self, state, player):
        return state * player
    
    # Codificación del state
    # Canal 1: 1 en posiciones ocupadas por Jugador -1. 0 en las demás posiciones
    # Canal 2: 1 en posiciones vacías. 0 en las demás posiciones
    # Canal 3: 1 en posiciones ocupadas por Jugador 1. 0 en las demás posiciones
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        # Reordenamiento de dimensiones a formato canal x fila x columna si el state tiene 3 dimensiones
        # En Conecta 4, el state tiene 2 dimensiones (filas x columnas)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state