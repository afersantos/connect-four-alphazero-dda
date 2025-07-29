import numpy as np
import math
import torch

# Clase para definición de un nodo del MCTS
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
    
    # Método para comprobar si un nodo está expandido (no es leaf node)
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    # Método para seleccionar un nodo en base a UCB
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    # Método para obtener el PUCB de un nodo
    def get_ucb(self, child):
        if child.visit_count == 0: # Se fuerza q_value=0 cuando no se ha visitado el nodo para evitar indeterminación 0/0
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 # Normalización de q_value (tasa de victorias del nodo) de rango [-1, 1] a [0, 1]. Se resta 1 - q_value para verlo desde la perspectiva del nodo padre
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    # Método para expandir un leaf node (devuelve el nodo child creado)
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0: # Solo se comprueban las acciones con jugadas válidas (p_a!=0)
                child_state = self.state.copy() # Copiamos state actual en el nodo child
                child_state = self.game.get_next_state(child_state, action, player=1) # Obtenemos nuevo state tras realizar la acción
                child_state = self.game.change_perspective(child_state, player=-1) # Cambiamos perspectiva del state (el nodo child corresponde al oponente)

                child = Node(self.game, self.args, child_state, self, action, prob) # Creación del nodo child
                self.children.append(child) # Añadimos el nodo child a la lista de nodos childs del nodo padre
                
        return child
    
    # Método para realizar backpropagation en MCTS
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value) # Cambiamos de oponente porque el nodo padre corresponde a la perspectiva del oponente
        # Llamada recursiva al método backpropagate hasta llegar al root node
        if self.parent is not None:
            self.parent.backpropagate(value)  


# Clase para ejecutar MCTS desde un estado dado
class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args # Hiperparámetros de MCTS
        self.model = model
        
    @torch.no_grad() # p_a y v del método search son predicciones (no hay entrenamiento). Decorador de PyTorch que desactiva el cálculo de gradiente en el método search (optimización de memoria y velocidad de cálculo)
    # Método search para ejecutar todas las búsquedas de MCTS. El método devuelve la distribución de probabilidades de las acciones determinada por MCTS (π)
    def search(self, state):
        # Establecemos el root node en el estado actual
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state) # Obtenemos los movimientos válidos (1=movimiento válido, 0=movimiento no válido)
        policy *= valid_moves # Se pone 0 a los movimientos no válidos en la política
        policy /= np.sum(policy) # Se transforma cada valor de la política a formato probabilidad
        root.expand(policy)
        
        # Realizamos tantas búsquedas como num_searches definidas. Entendemos por búsqueda a la combinación de selección, expansión y backpropagation
        for search in range(self.args['num_searches']):
            # Establecemos primer nodo como root
            node = root
            
            # Etapa de selección de MCTS (recorremos el árbol hasta llegar a leaf node seleccionado con UBC)
            while node.is_fully_expanded():
                node = node.select()
            
            # Obtenemos recompensa (valor) y check si es un nodo terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value) # Invertimos valor de recompensa (valor) para verlo desde la perspectiva del jugador actual
            
            # Etapa de expansión
            if not is_terminal:
                policy, value = self.model( # PREDICCION: La red devuelve policy (p_a) y value (v) del nodo
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0) # unsqueeze(0) agrega una nueva dimensión al inicio del tensor obteniendo un formato de batch x canal x fila x columna 
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() # Normalización de valores a distribución de probabilidades mediante softmax
                valid_moves = self.game.get_valid_moves(node.state) # Enmascaramiento de jugadas inválidas
                policy *= valid_moves # Todas las jugadas ilegales se les asigna 0 en p_a
                policy /= np.sum(policy) # Normalización de valores a distribución de probabilidades (suma de valores igual a 1)
                
                value = value.item() # Valor escalar v predicho por la red
                
                node.expand(policy)
            
            # Etapa de backpropagation
            node.backpropagate(value)    
            
        # Calculo y return del número de visitas a cada nodo del primer nivel de MCTS
        visit_counts = np.zeros(self.game.action_size)
        for child in root.children:
            visit_counts[child.action_taken] = child.visit_count
        #visit_counts /= np.sum(visit_counts) # Normalización de valores en formato de probabilidades (rango [0, 1])
        
        return visit_counts