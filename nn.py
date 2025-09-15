import torch.nn as nn
import torch.nn.functional as F

# Clase de la red neuronal
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__() # La clase hereda métodos y atributos de nn.Module
        
        self.device = device

        # Capa convolucional inicial
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1), # Padding 1, se añaden 0 en los bordes para que la capa conv no cambie la forma de la imagen de entrada
            nn.BatchNorm2d(num_hidden), # Sirve para acelerar el entrenamiento, reducir la covariancia interna y hacer que la red sea más estable
            nn.ReLU() # Convierte los valores negativos en 0
        )
        
        # Backbone (secuencia de bloques residuales)
        self.backBone = nn.ModuleList( # Con ModuleList podemos definir un bucle de capas secuenciadas
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        # Policy-head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size) # Linear Fully-Connected. Input 32 canales de 6x7 (state). Output array 7 valores (p_a)
        )
        
        # Value-head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1), # Linear Fully-Connected. Input 3 canales de 6x7 (state). Output escalar (v)
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x) # Capa conv inicial
        for resBlock in self.backBone: # Secuencia de bloques residuales
            x = resBlock(x)
        policy = self.policyHead(x) # Policy-head devuelve p_a
        value = self.valueHead(x) # Value head devuelve v
        return policy, value
        
# Clase de los bloques residuales que forman la ResNet
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__() # La clase hereda métodos y atributos de nn.Module

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x))) # Con F se puede realizar ReLU sin necesidad de registrar el módulo self.ReLU()
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x) # Con F se puede realizar ReLU sin necesidad de registrar el módulo self.ReLU()
        return x