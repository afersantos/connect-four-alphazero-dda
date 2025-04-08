from ConnectFour import ConnectFour
import numpy as np
import torch
from nn import ResNet
from MCTS import MCTS

game = ConnectFour(render_mode=True)
player = 1

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("Models/250407_2144/model_7_ConnectFour.pt", map_location=device))
model.eval()

mcts = MCTS(game, args, model)

state = game.get_initial_state()
game.render(state)

while True:
    #print(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("Movimientos válidos:", [i+1 for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input("Introduzca nuevo movimiento: ")) - 1

        if valid_moves[action] == 0:
            print("Movimiento NO válido")
            continue
            
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = game.get_next_state(state, action, player)

    game.render(state, action)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        if value == 1:
            if player==1:
                print("\n¡HAS GANADO A LA IA!\nFin de la partida")
            elif player==-1:
                print("\nLa IA ha ganado\nFin de la partida")
        else:
            print("Empate")

        game.close()
        break
        
    player = game.get_opponent(player)