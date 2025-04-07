from ConnectFour import ConnectFour
import torch
from nn import ResNet
from AlphaZeroParallel import AlphaZeroParallel

torch.manual_seed(0) # Para reproducibilidad de resultados

game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, num_resBlocks=9, num_hidden=128, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,                          # Hiperparámetro de exploración de UCB para MCTS
    'num_searches': 600,             # Número de búsquedas en cada MCTS
    'num_iterations': 8,             # Número de iteraciones. En cada iteración se optimizan los pesos num_epochs veces y se juegan num_selfPlay_iterations partidas Self-Play para generar datos de entrenamiento
    'num_selfPlay_iterations': 500,  # Número de partidas Self-Play en cada iteración
    'num_parallel_games': 100,
    'num_epochs': 4,                 # Número de épocas en cada iteración (número de veces que se actualizan los pesos en cada iteración)
    'batch_size': 128,               # Tamaño de lote del conjunto de entrenamiento en cada epoch
    'temperature': 1.25,             # Temperatura para la distribución de probabilidades de las acciones determinada por MCTS (π)
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn() # Método learn para ejecutar ciclo de Self-Play -> Generar datos de training -> Training -> Optimizar modelo para Self-PLay