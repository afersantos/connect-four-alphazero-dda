import numpy as np

state = np.array([[1, 1, 0],
                  [0, -1, 0],
                  [0, 0, 0]])

def get_encoded_state(state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)

    print(len(state.shape))

    if len(state.shape) == 3:
        encoded_state = np.swapaxes(encoded_state, 0, 1) # Reordenamiento de dimensiones a formato canal x fila x columna
    
    return encoded_state

input = get_encoded_state(state)
print(input.shape)