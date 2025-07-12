from ConnectFour import ConnectFour
import numpy as np
import torch
from nn import ResNet
from MCTS import MCTS
from PIDController import PIDController

# Instanciamos objeto del juego
game = ConnectFour(render_mode=True)

# Declaramos hiperparámetros
args = {
    'C': 1, # 2 (original)
    'num_searches': 600, # 600 (original)
    'dirichlet_epsilon': 0., # 0 (original)
    'dirichlet_alpha': 0.3, # 0.3 (original)
    'temperature_dda': 1.6,
    'max_games': 20
}

# Cargamos el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("Models/250407_2144/model_7_ConnectFour.pt", map_location=device))
model.eval()

# Instanciamos el objeto MCTS
mcts = MCTS(game, args, model)

# Inicialición de variables
num_games = 0
agent_win_rate = 0.
agent_num_wins = 0
temperature_controller = PIDController(Kp=1, Ki=0.0, Kd=0.0, setpoint=0.5)

# Jugamos contra el agente 5 partidas
while num_games < args['max_games']:
    player = 1 # El jugador humano empieza la partida
    #player = np.random.choice([-1,1])
    state = game.get_initial_state()
    game.render(state)

    while True:
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("\nMovimientos válidos:", [i+1 for i in range(game.action_size) if valid_moves[i] == 1])

            try:
                action = int(input("Introduzca nuevo movimiento: ")) - 1
            except ValueError:
                print("\nMovimiento NO válido. Introduzca de nuevo un movimiento.\n")
                continue

            if action+1<1 or action+1>7:
                print("\nMovimiento NO válido. Introduzca de nuevo un movimiento.\n")
                continue

            if valid_moves[action] == 0:
                print("\nMovimiento NO válido. Introduzca de nuevo un movimiento.\n")
                continue
                
        else:
            neutral_state = game.change_perspective(state, player)
            policy = mcts.search(neutral_state) # Devuelve el número de visitas a cada nodo desde el estado actual
            print("\nNúmero de visitas a cada nodo:", policy)
            policy = policy ** (1.0/args['temperature_dda']) # Aplicamos temperatura a la política de MCTS (número de visitas a cada nodo)
            policy = policy / np.sum(policy) # Normalizamos la política de MCTS (número de visitas a cada nodo) para que sume 1
            print("\nNúmero de visitas a cada nodo (con temperatura):", policy)
            action = np.random.choice(game.action_size, p=policy) # Se elige acción según la política de MCTS (número de visitas a cada nodo) con temperatura aplicada
            
        state = game.get_next_state(state, action, player)

        game.render(state, action)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            num_games+=1

            if value == 1:
                if player==1: # Humano
                    print("\n¡HAS GANADO A LA IA!\nFin de la partida")
                elif player==-1: # IA
                    agent_num_wins += 1
                    print("\nLa IA ha ganado\nFin de la partida")
            else:
                print("\nEmpate\nFin de la partida")
            
            # Actualizamos la tasa de victorias del agente
            agent_win_rate = agent_num_wins / num_games
            print(f"\nTasa de victorias de la IA: {agent_win_rate:.2%}\n")

            # Actualizamos la temperatura del agente
            adjustment = temperature_controller.update(current_win_rate=agent_win_rate)
            args['temperature_dda'] -= adjustment
            args['temperature_dda'] = max(0.1, args['temperature_dda'])
            print(f"\nTemperatura ajustada: {args['temperature_dda']:.2f} (ajuste: {-1*adjustment:.2f})\n")

            break
            
        player = game.get_opponent(player)
    
game.close()