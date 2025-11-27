📘 Proyecto: Deep Q-Learning para Atari Seaquest

Autor: André Chávez Contreras
Universidad de Xalapa — Ingeniería en Inteligencia Artificial

🐠 Descripción del Proyecto

Este repositorio contiene el desarrollo de un agente de Deep Q-Learning (DQN) entrenado para jugar Seaquest, un videojuego de Atari incluido en los entornos de Gymnasium.
El entrenamiento se realizó utilizando Stable-Baselines3, siguiendo buenas prácticas como:

Preprocesamiento estándar para Atari (84×84, escala de grises, frame stacking).

Entrenamiento con entornos vectorizados (n_envs=4).

Versiones de las librerias:
Python 3.12.7
gymnasium 1.2.2
ale-py 0.9.0
stable-baselines3 2.7.0
torch 2.9.1+cu130
opencv-python 4.10.0.84


Uso de experience replay, redes objetivo y exploración epsilon-greedy.

Continuación del entrenamiento desde checkpoints.

Evaluación determinista del desempeño final del agente.

Este trabajo forma parte del reporte académico en formato IEEE correspondiente a la segunda unidad.

📂 Contenido del Repositorio
📁 Proyecto-DQL-Seaquest
│
├── modelos/
│   └── dqn_seaquest_run9_finished_11429248_steps.zip                 # Último modelo entrenado
│
├── video/
│   └── seaquest_30s_demo-episode-0.mp4              # Video de evaluación del agente
│
├── scripts/
│   ├── Entrenamiento_inicial.py         # Fase I: entrenamiento base
│   ├── ContinuarEntrenamiento.py        # Fase II: continuar + optimizar
│   └── Visualizar_modelo.py             # Fase III: evaluación / videos
│
├── reporte/
│   └── Proyecto_Agentes_p2.pdf                 # Documento oficial del proyecto
│
└── README.md                            # 📘 Este archivo

🚀 Cómo Ejecutar el Proyecto
1️⃣ Instalar dependencias
pip install \
  gymnasium==1.2.2 \
  gymnasium[atari]==1.2.2 \
  gymnasium[accept-rom-license]==1.2.2 \
  ale-py==0.9.0 \
  stable-baselines3==2.7.0 \
  torch==2.9.1+cu130 \
  torchvision==0.20.1+cu130 \
  torchaudio==2.9.1+cu130 \
  opencv-python==4.10.0.84

pip install typing-extensions


2️⃣ Entrenar desde cero
python scripts/Entrenamiento_inicial.py

3️⃣ Continuar el entrenamiento
python scripts/ContinuarEntrenamiento.py

4️⃣ Visualizar el agente
python scripts/Visualizar_modelo.py

📊 Resultados principales

El modelo final entrenado alcanzó:

≈ 11.4M timesteps totales

Recompensa promedio (greedy): ~1152 puntos

Aprendizaje de comportamientos clave:
✓ Eliminar enemigos
✓ Rescatar buzos
✓ Subir para recargar oxígeno

🔧 Trabajo Futuro

Probar variantes avanzadas de DQN: Double DQN, PER, Dueling DQN, NoisyNets.

Comparar con algoritmos PPO, A2C o Rainbow.

Añadir métricas más completas y análisis de curvas de aprendizaje.

Mejorar reproducibilidad y limpieza del repositorio.

📚 Referencias

Stable Baselines 3: https://stable-baselines3.readthedocs.io

Gymnasium Atari: https://gymnasium.farama.org/environments/atari/

RL Baselines3 Zoo: https://github.com/DLR-RM/rl-baselines3-zoo

Deep RL Course (HuggingFace): https://huggingface.co/learn/deep-rl

Mnih et al., “Human-level control through deep reinforcement learning” (Nature, 2015)