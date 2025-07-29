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
    if player == 1:
        policy = mcts.search(state)
        action = np.argmax(policy)
            
    else:
        neutral_state = game.change_perspective(state, player)
        policy = mcts.search(neutral_state)
        action = np.argmax(policy)
        
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