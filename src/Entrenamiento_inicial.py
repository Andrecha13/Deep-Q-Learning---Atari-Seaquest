import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# ====================================================================
# CONFIGURACIÓN DE RUTAS
# ====================================================================

# Usamos la versión NoFrameskip-v4 porque SB3 maneja el skip frame internamente
ENV_ID = "SeaquestNoFrameskip-v4"

# Ajusta esta ruta a tu carpeta real
BASE_LOG_DIR = r"C:\Users\andre\OneDrive\UX\Quinto semestre\Agentes inteligentes\Codigos\Gymnasium\Logs"

CHECKPOINT_DIR = os.path.join(BASE_LOG_DIR, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(BASE_LOG_DIR, "logs_tb")
FINAL_MODEL_DIR = os.path.join(BASE_LOG_DIR, "best_models")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print(f"🚀 Iniciando entrenamiento nuevo en {ENV_ID}...")

    # 1. Crear el entorno Atari optimizado
    # make_atari_env aplica: Grayscale, Resize(84x84), NoopReset, etc.
    n_envs = 32
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=42)
    
    # 2. Apilar Frames (Frame Stacking)
    # Fundamental para que la IA entienda la dirección de movimiento y velocidad
    env = VecFrameStack(env, n_stack=4)

    # 3. Hiperparámetros (Optimizados para Atari/CNN)
    HYPERPARAMS = {
        "learning_rate": 1e-4,
        "buffer_size": 100_000,       # Memoria de repetición
        "learning_starts": 100_000,   # Pasos aleatorios antes de entrenar (llenar buffer)
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,              # Entrenar cada 4 pasos
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,  # Reducir epsilon durante el 10% del total
        "exploration_final_eps": 0.01,
        "optimize_memory_usage": False, # Pon True si tienes poca RAM
        "verbose": 1
    }

    # 4. Inicializar Modelo DQN
    # Usamos "CnnPolicy" porque la entrada son imágenes (píxeles)
    model = DQN(
        "CnnPolicy", 
        env, 
        tensorboard_log=TENSORBOARD_LOG_DIR, 
        **HYPERPARAMS
    )

    # 5. Configurar Checkpoints
    # Guardamos cada 100,000 pasos para poder reanudar
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, # Frecuencia global (se divide entre n_envs internamente)
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_seaquest"
    )

    # 6. Entrenar
    TOTAL_TIMESTEPS = 2_000_000 # Ajustar según tiempo disponible (Atari suele requerir 10M+)
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name="DQN_Seaquest_Run1" # Nombre para Tensorboard
        )
        # Guardar modelo final al terminar
        model.save(os.path.join(FINAL_MODEL_DIR, "dqn_seaquest_finished"))
        print("✅ Entrenamiento finalizado y guardado.")
        
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento pausado manualmente. Se guardará el estado actual...")
        model.save(os.path.join(CHECKPOINT_DIR, "dqn_seaquest_INTERRUPTED"))
        print("💾 Modelo de emergencia guardado.")