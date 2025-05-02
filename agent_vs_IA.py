from ConnectFour import ConnectFour
import numpy as np
import torch
import Players.AdultPlayer
import Players.AdultSmarterPlayer
import Players.ChildPlayer
import Players.ChildSmarterPlayer
import Players.TeenagerPlayer
import Players.TeenagerSmarterPlayer
from nn import ResNet
from MCTS import MCTS
from PDIController import PDIController
import Players
import json
from tqdm import trange

# Declaramos hiperparámetros
args = {
    'C': 1, # 2 (original)
    'num_searches': 600, # 600 (original)
    'dirichlet_epsilon': 0., # 0 (original)
    'dirichlet_alpha': 0.3, # 0.3 (original)
    'temperature_dda': 1.6, # Temperatura del agente
    'num_games': 200, # Número de partidas a jugar contra cada oponente
}

# Inicializamos diccionario para guardar resultados
results = {
    'win_rate': [], # Tasa de victorias del agente
    'temperature_dda': [], # Temperatura del agente
    'moves_per_game': [], # Número de movimientos por partida
    'winner': [], # Ganador de la partida (1=oponente, -1=agente, 0=empate)
    'opponent_name': [], # Nombre del oponente
    'first_player': [], # Jugador que empieza la partida (1=oponente, -1=agente)
}

# Importamos el jugador al que se va a enfrentar el agente
opponents_list = [Players.ChildPlayer.ChildPlayer(),
                  Players.ChildSmarterPlayer.ChildSmarterPlayer(),
                  Players.TeenagerPlayer.TeenagerPlayer(),
                  Players.TeenagerSmarterPlayer.TeenagerSmarterPlayer(),
                  Players.AdultPlayer.AdultPlayer(),
                  Players.AdultSmarterPlayer.AdultSmarterPlayer()
                  ]

# Instanciamos objeto del juego
game = ConnectFour(render_mode=False)

# Cargamos el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("Models/250407_2144/model_7_ConnectFour.pt", map_location=device))
model.eval()

# Instanciamos el objeto MCTS
mcts = MCTS(game, args, model)

for opponent in opponents_list:
    print(f'Jugando vs {opponent.getName()}...')

    # Reiniciamos temperatura del agente
    args['temperature_dda'] = 1.6 # Temperatura inicial del agente

    # Inicialición de variables
    moves_per_game = 0
    agent_win_rate = 0.
    agent_num_wins = 0
    temperature_controller = PDIController(Kp=1, Ki=0.0, Kd=0.0, setpoint=0.5)

    for num_game in trange(1, args['num_games']+1, desc='Partidas'):
        player = 1 # Indica el jugador que empieza la partida (1=oponente, -1=agente)
        #player = np.random.choice([-1,1]) # Indica el jugador que empieza la partida (1=oponente, -1=agente)
        results['first_player'].append('opponent' if player==1 else 'agent') # Guardamos el jugador que empieza la partida (1=oponente, -1=agente)

        # Inicializamos el juego
        state = game.get_initial_state()
        #game.render(state)

        while True:
            # Selección de acción
            if player == 1: # Oponente
                action = opponent.play(state)

            else: # Agente
                neutral_state = game.change_perspective(state, player)
                policy = mcts.search(neutral_state) # Devuelve el número de visitas a cada nodo desde el estado actual
                policy = policy ** (1.0/args['temperature_dda']) # Aplicamos temperatura a la política de MCTS (número de visitas a cada nodo)
                policy = policy / np.sum(policy) # Normalizamos la política de MCTS (número de visitas a cada nodo) para que sume 1
                action = np.random.choice(game.action_size, p=policy) # Se elige acción según la política de MCTS (número de visitas a cada nodo) con temperatura aplicada
            
            moves_per_game += 1

            state = game.get_next_state(state, action, player)

            #game.render(state, action)
            
            value, is_terminal = game.get_value_and_terminated(state, action)
            
            if is_terminal:
                if value == 1:
                    if player==-1: # El agente ha ganado
                        agent_num_wins += 1
                
                # Actualizamos la tasa de victorias del agente
                agent_win_rate = agent_num_wins / num_game

                # Guardamos los resultados de la partida
                results['win_rate'].append(agent_win_rate)
                results['temperature_dda'].append(args['temperature_dda'])
                results['moves_per_game'].append(moves_per_game)
                results['winner'].append('draw' if value!=1 else ('opponent' if player==1 else 'agent'))
                results['opponent_name'].append(opponent.getName())
                moves_per_game = 0

                # Actualizamos la temperatura del agente
                adjustment = temperature_controller.update(current_win_rate=agent_win_rate)
                args['temperature_dda'] -= adjustment
                args['temperature_dda'] = max(0.1, args['temperature_dda'])

                break
                
            player = game.get_opponent(player)
        
    #game.close()

# Guardamos resultados como archivo JSON
with open("Results/resultados.json", "w") as file:
    json.dump(results, file, indent=4)