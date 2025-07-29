import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange
from MCTS import MCTS

# Clase AlphaZero con sus dos métodos Self-Play y Train
# Método learn para ejecutar ciclo de Self-Play -> Generar datos de training -> Training -> Optimizar modelo para Self-PLay
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    # Método Self-Play devuelve datos de partidas para el training
    # Datos en formato tupla: states de la partida, π de MCTS antes de aplicar temperatura en cada state y recompensas en cada state    
    def selfPlay(self):
        memory = [] # Lista para almacenar los datos de una partida Self-Play
        player = 1
        state = self.game.get_initial_state()
        
        while True: # Bucle hasta finalizar pàrtida Self-Play
            neutral_state = self.game.change_perspective(state, player) # Cada vez que llamamos a MCTS hay que verlo desde la perspectiva del Jugador 1
            visit_counts = self.mcts.search(neutral_state) # Devuelve el número de visitas a cada nodo desde el estado actual
            action_probs /= np.sum(visit_counts) # Normalización de valores en formato de probabilidades (rango [0, 1])
            
            memory.append((neutral_state, action_probs, player)) # Se guarda en la memoria una tupla con el state (perspectiva de Jugador 1), π y resultado de la partida (player indica el jugador que ganó, perdió o empató)
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature']) # Divide temperature_action_probs with its sum in case of an error. Se aplica temperatura a π
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Se realiza acción basada en π con parámetro de temperatura
            
            state = self.game.get_next_state(state, action, player) # Siguiente state
            
            value, is_terminal = self.game.get_value_and_terminated(state, action) # Obtenemos recompensa y check estado terminal
            
            if is_terminal: # Fin de la partida
                returnMemory = [] # Lista con toda la información definitiva de la partida Self-Play
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state), # State codificado
                        hist_action_probs, # π (antes de aplicar temperatura)
                        hist_outcome # Recompensas de cada estado desde la perspectiva de cada jugador
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player) # Cambio de jugador

    # Método train para optimizar pesos de la red con los datos de entrenamiento de Self-Play           
    def train(self, memory):
        random.shuffle(memory) # Mezclado de los datos de entrenamiento para no usar siempre la misma secuencia de datos
        for batchIdx in range(0, len(memory), self.args['batch_size']): # Recorrido de los datos de entrenamiento por lotes
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error. Muestra del conjunto de entrenamiento
            state, policy_targets, value_targets = zip(*sample) # Desempaquetamiento y agrupación de los datos de la muestra de entrenamiento
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1) # Cambio de formato a arrays numpy
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device) # Conversión del conjunto de estados a un tensor
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device) # Conversión de π de los distintos estados a un tensor
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device) # Conversión de recompensas en cada estado a un tensor
            
            out_policy, out_value = self.model(state) # p_a's y v's predichos por la red para el conjunto de states
            
            policy_loss = F.cross_entropy(out_policy, policy_targets) # Término de error de política en función de coste
            value_loss = F.mse_loss(out_value, value_targets) # Término de error de valor en función de coste
            loss = policy_loss + value_loss # Función de coste. No incluye el término de regularización: l2_reg = sum((p**2).sum() for p in model.parameters()) * l2_coeff
            
            self.optimizer.zero_grad() # Reset de gradientes del modelo antes de calcular nuevos gradientes en backpropagation 
            loss.backward() # Cálculo de gradientes de la función de coste
            self.optimizer.step() # Actualización de parámetros de la red usando los gradientes de loss.backward()
    
    # Ciclo de Self-Play -> Generar datos de entrenamiento -> Training -> Optimizar modelo para Self-PLay 
    def learn(self):
        for iteration in range(self.args['num_iterations']): # Bucle num_iterations veces
            memory = [] # Lista para almacenar los datos generados durante Self-Play en cada iteración
            
            self.model.eval() # Modo Inferencia del modelo: desactiva dropout y usa valores promedios acumulados en lugar de calcular estadísticas en cada mini-lote en BatchNorm
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'], desc=f'Self-Play Iteration: {iteration}', unit='selfPlay_iteration'): # Generación de datos de entrenamiento mediante Self-Play. trange para visualizar barra de progreso
                memory += self.selfPlay()
   
            self.model.train() # Modo entrenamiento del modelo: activa dropout y calcula estadísticas en cada mini-lote en lugar de usar valores precalculados en BatchNorm
            for epoch in trange(self.args['num_epochs'], desc=f'Training Iteration: {iteration}', unit='epoch'): # Bucle num_epochs veces. En cada epoch se actualizan los pesos con los datos generados de Self-Play. trange para visualizar barra de progreso
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt") # Guardado de pesos en cada iteración
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt") # Guardado del estado del optimizador en cada iteración
            # PARA REANUDAR ENTRENAMIENTO
            # Cargar los pesos del modelo
            #model.load_state_dict(torch.load(f"model_{iteration}_{game}.pt"))
            # Cargar el estado del optimizador
            #optimizer.load_state_dict(torch.load(f"optimizer_{iteration}_{game}.pt"))
            # Poner el modelo en modo entrenamiento
            #model.train()