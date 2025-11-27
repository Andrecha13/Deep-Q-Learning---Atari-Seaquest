import time
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy # <-- Nueva Importación

# ====================================================================
# CONFIGURACIÓN
# ====================================================================

ENV_ID = "SeaquestNoFrameskip-v4"
BASE_LOG_DIR = r"C:\Users\andre\OneDrive\UX\Quinto semestre\Agentes inteligentes\Codigos\Gymnasium\Logs"

# Ruta del modelo que quieres ver jugar
MODEL_PATH = os.path.join(BASE_LOG_DIR, "best_models", "dqn_seaquest_run9_finished_11429248_steps.zip")
# MODEL_PATH = os.path.join(BASE_LOG_DIR, "checkpoints", "dqn_seaquest_1000000_steps.zip")

# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        # Intenta buscar el interrumpido si no existe el específico
        MODEL_PATH = os.path.join(BASE_LOG_DIR, "checkpoints", "dqn_seaquest_INTERRUPTED.zip")
        print(f"⚠️ Intentando cargar backup: {MODEL_PATH}")

    print(f"📺 Visualizando modelo: {os.path.basename(MODEL_PATH)}")

    # 1. Crear entorno con render_mode='human'
    # Se usa n_envs=1 para la visualización en tiempo real
    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
    
    # 2. Apilar Frames 
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # 3. Cargar Agente
    model = DQN.load(MODEL_PATH, env=eval_env)
    
    # 4. (NUEVO) Evaluación Estadística Formal
    # Ejecutamos 5 episodios para obtener un promedio de recompensa confiable
    # Nota: Este cálculo puede tomar unos segundos.
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=5, 
        deterministic=True # Usar la política greedy (sin exploración)
    )
    print("=" * 50)
    print(f"📈 RENDIMIENTO PROMEDIO EVALUADO:")
    print(f"   Recompensa Media: {mean_reward:.2f} (comparable con los 785 del reporte)")
    print(f"   Desviación Estándar: {std_reward:.2f}")
    print("=" * 50)

    # 5. Bucle de Juego para Visualización
    obs = eval_env.reset()
    current_score = 0 # Variable para acumular puntos del episodio actual
    
    try:
        while True:
            # CAMBIO CLAVE: Usamos deterministic=False para reintroducir la exploración (epsilon)
            # Esto debería mejorar el rendimiento si el agente se estancaba.
            # En la visualización se puede dejar en False para ver más variedad.
            action, _states = model.predict(obs, deterministic=False)
            
            # Ejecutar acción
            obs, rewards, dones, info = eval_env.step(action)
            
            # Sumar recompensa (rewards es una lista porque es un entorno vectorizado)
            current_score += rewards[0]
            
            # Control de velocidad (FPS)
            time.sleep(0.01) # Más rápido para verlo jugar sin tanto lag

            if dones[0]:
                print(f"💀 Game Over - Score Final de esta partida: {current_score:.0f}")
                current_score = 0 # Reiniciar contador
                time.sleep(1.0) 

    except KeyboardInterrupt:
        print("\n👋 Cerrando visualización.")
        eval_env.close()