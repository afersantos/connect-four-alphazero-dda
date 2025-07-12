# Implementación de AlphaZero con Ajuste Dinámico de Dificultad en Connect Four

Este proyecto implementa el algoritmo **AlphaZero** con **Ajuste Dinámico de Dificultad** para el juego **Connect Four**.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos y carpetas principales:

- **`Models/`**: Carpeta que contiene los agentes/modelos entrenados y sus optimizadores.
- **`Players/`**: Carpeta que contiene los oponentes programados con estrategias heurísticas para ser enfrentados contra el agente AlphaZero.
- **`Results/resultados.json`**: Archivo donde se almacenan los resultados de las partidas del agente AlphaZero frente a los oponentes con estrategias heurísticas.
- **`AlphaZero.py`**: Contiene la clase principal `AlphaZero`, que implementa el entrenamiento mediante partidas de auto-juego (Self-Play) y la integración con MCTS.
- **`AlphaZeroParallel.py`**: Versión mejorada de `AlphaZero.py` que permite realizar entrenamientos simultáneos en paralelo.
- **`ConnectFour.py`**: Contiene la clase que simula el juego de Connect Four.
- **`MCTS.py`**: Contiene la clase que ejecuta la búsqueda de MCTS.
- **`MCTSParallel.py`**: Versión mejorada de `MCTS.py` que permite realizar entrenamientos simultáneos en paralelo.
- **`PDIController.py`**: Contiene la clase encargada del ajuste dinámico del hiperparámetro de temperatura mediante Control PID.
- **`agent_vs_IA.py`**: Permite evaluar el rendimiento del agente AlphaZero contra los diferentes oponentes con ajuste dinámico de dificultad.
- **`agent_vs_agent.py`**: Simula partidas entre dos agentes AlphaZero.
- **`play_vs_agent.py`**: Permite a un humano jugar contra el agente AlphaZero.
- **`plot_results.ipynb`**: Notebook con resultados obtenidos de los enfrentamientos del agente AlphaZero frente a los oponentes.
- **`training.py`**: Script para entrenamiento del agente AlphaZero.

## Uso

### Clonar el repositorio en local
Clona este repositorio en tu máquina local:
```
git clone https://github.com/afersantos/connect-four-alphazero-dda.git

cd connect-four-alphazero-dda
```

### Instalar dependencias
Asegúrate de tener Python instalado. La versión de python empleada en este proyecto es `Python 3.12.7`.

Luego, instala las dependencias necesarias:
```
pip install -r requirements.txt
```

### Jugar contra AlphaZero con Ajuste Dinámico de Dificultad
Ejecuta el siguiente archivo:
```
python play_vs_agent.py
```