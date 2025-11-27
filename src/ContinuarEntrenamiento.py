import os
import glob 
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage # <-- Importación añadida
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
import torch 
from typing import Union, List

# ====================================================================
# CONFIGURACIÓN DE RUTAS Y MODELOS
# ====================================================================

ENV_ID = "SeaquestNoFrameskip-v4"
# 🚨 AJUSTA ESTA RUTA BASE A TU UBICACIÓN REAL DE LOGS 🚨
BASE_LOG_DIR = r"C:\Users\andre\OneDrive\UX\Quinto semestre\Agentes inteligentes\Codigos\Gymnasium\Logs"
CHECKPOINT_DIR = os.path.join(BASE_LOG_DIR, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(BASE_LOG_DIR, "logs_tb")

# --- PARAMETRIZACIÓN DEL RUN DE CONTINUACIÓN (RUN 9) ---
# Decaimiento total de Epsilon en 5M pasos adicionales para alcanzar 10M
STEPS_TO_ADD = 3_570_752 
N_ENVS = 32 # Usar 32 entornos vectorizados para alta velocidad

# ====================================================================
# HIPERPARÁMETROS ESTABLES (Resultados de 1270 puntos)
# ====================================================================
# VALORES LIMPIOS: Sin comas, forzando tipo FLOAT o INT
NEW_LR = 0.00015
NEW_GAMMA = 0.995
NEW_BATCH_SIZE = 256
# OPTIMIZACIÓN DE MEMORIA: Reducimos el buffer de 1M (defecto) a 100K
NEW_BUFFER_SIZE = 100_000

NEW_EXPLORATION_FRACTION = 1.0 
NEW_EXPLORATION_INITIAL_EPS = 0.1
NEW_EXPLORATION_FINAL_EPS = 0.01 

# ====================================================================
# LÓGICA DE DETECCIÓN Y CALLBACKS
# ====================================================================

class CustomMetricCallback(BaseCallback):
    """
    Callback personalizado para registrar métricas críticas de la política
    directamente en TensorBoard.
    """
    def __init__(self, verbose: int = 0):
        super(CustomMetricCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Solo registra si es un agente DQN
        if isinstance(self.model, DQN):
            # Epsilon Actual (el valor que realmente usa el agente)
            current_epsilon = self.model.exploration_rate
            self.logger.record("rollout/current_epsilon", current_epsilon)
            
            # Otros Hiperparámetros fijos
            self.logger.record("train/learning_rate", self.model.learning_rate)
            self.logger.record("train/batch_size", self.model.batch_size)
            self.logger.record("gamma", self.model.gamma)
            # Ahora debería mostrar 100000 como capacidad máxima del buffer
            self.logger.record("dqn/buffer_size", self.model.replay_buffer.buffer_size) 
        return True 

def find_latest_checkpoint(path: str, prefix: str = "dqn_seaquest_") -> Union[str, int]:
    """Busca el archivo de checkpoint con el mayor número de pasos."""
    files = glob.glob(os.path.join(path, f'{prefix}*.zip'))
    if not files:
        return None, 0

    def get_steps(file_path):
        try:
            name = os.path.basename(file_path)
            # Extracción robusta del número de pasos
            if '_steps.zip' in name:
                steps_str = name.replace('_steps.zip', '').split('_')[-1]
                return int(steps_str)
            return 0
        except:
            return 0
    
    latest_file = max(files, key=get_steps)
    latest_steps = get_steps(latest_file)
    return latest_file, latest_steps

# ====================================================================
# MAIN: Lógica de Carga, Transferencia y Continuación
# ====================================================================

if __name__ == "__main__":
    # 0. Detección de último checkpoint
    LOAD_PATH, START_TIMESTEPS = find_latest_checkpoint(CHECKPOINT_DIR)
    if LOAD_PATH is None:
        print("❌ Error: No se encontraron archivos de checkpoint. Imposible reanudar.")
        exit()

    print(f"🔄 Cargando pesos desde: {os.path.basename(LOAD_PATH)} ({START_TIMESTEPS} pasos)...")
    
    # 1. Diagnóstico de GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Dispositivo detectado: {device.upper()}")

    # 2. Recrear el entorno
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=42) 
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env) 

    # 3a. Inicializar el NUEVO modelo DQN con la configuración optimizada (buffer_size=100K)
    print(f"✨ Inicializando el NUEVO modelo (destino) con buffer_size={NEW_BUFFER_SIZE}...")
    # NOTA: Al pasar 'env' ya traspuesto, DQN NO aplicará VecTransposeImage internamente.
    model = DQN(
        CnnPolicy, 
        env,
        learning_rate=NEW_LR, 
        buffer_size=NEW_BUFFER_SIZE, # ¡Configuración correcta!
        learning_starts=10_000, 
        batch_size=NEW_BATCH_SIZE,
        gamma=NEW_GAMMA,
        target_update_interval=1000, 
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        device=device # Asignar el dispositivo correcto (GPU/CPU)
    )

    # 3b. Carga el modelo viejo (origen) en una instancia temporal 
    print(f"🧠 Cargando modelo temporal (origen) para transferir los pesos...")
    
    # Usamos None en env para que no envuelva el entorno internamente, solo cargue los datos.
    try:
        temp_model = DQN.load(LOAD_PATH, env=None, device=device) 
    except Exception as e:
        print(f"⚠️ Aviso: Fallo la carga del modelo temporal sin entorno. Intentando con entorno...")
        try:
            # Si falla, cargamos con el env, el cual ya tiene el wrapper de VecTransposeImage
            temp_model = DQN.load(LOAD_PATH, env=env, device=device) 
        except Exception as e_final:
            print(f"❌ ERROR CRÍTICO: Fallo al cargar el modelo (origen). Error: {e_final}")
            exit()


    # 3c. Transferir los pesos del modelo temporal (origen) al modelo nuevo (destino)
    model.set_parameters(temp_model.get_parameters())
    
    # Extraemos el contador de pasos real del modelo temporal por seguridad.
    START_TIMESTEPS = max(START_TIMESTEPS, temp_model.num_timesteps)

    # El temp_model se descarta
    del temp_model
    torch.cuda.empty_cache() # Liberar memoria de la GPU

    # 3d. **FIX FINAL Y FORZADO: REEMPLAZO DEL REPLAY BUFFER**
    print(f"⚔️ Forzando Replay Buffer de tamaño {NEW_BUFFER_SIZE}...")
    
    # El observation_space del env ya es (H, W, C) gracias al VecTransposeImage
    new_replay_buffer = ReplayBuffer(
        NEW_BUFFER_SIZE, # 100_000
        env.observation_space, # Ahora tiene el shape correcto (H, W, C)
        env.action_space,
        device=model.device,
        n_envs=model.n_envs,
        handle_timeout_termination=True, 
        optimize_memory_usage=False 
    )
    
    # Aseguramos que el nuevo buffer esté vacío y listo para ser llenado
    model.replay_buffer = new_replay_buffer

    # 3e. Forzar el contador de pasos. 
    model.num_timesteps = START_TIMESTEPS 
    print(f"✅ Transferencia de pesos completa. Contador de pasos ajustado a: {model.num_timesteps}")
    # Esta línea ahora usará el tamaño correcto: 100000
    print(f"✅ Nuevo buffer_size (GARANTIZADO por reemplazo): {model.replay_buffer.buffer_size}")


    # 4. Callbacks para continuación
    PERIODIC_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "run9_periodic")
    os.makedirs(PERIODIC_CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=PERIODIC_CHECKPOINT_DIR, 
        name_prefix="dqn_seaquest_periodic" 
    )
    
    metric_callback = CustomMetricCallback(verbose=1)

    # 5. Continuar Aprendizaje
    print(f"📈 Comenzando entrenamiento avanzado (Run 9). Entrenando {STEPS_TO_ADD} pasos adicionales...")
    
    try:
        model.learn(
            total_timesteps=STEPS_TO_ADD,
            callback=[checkpoint_callback, metric_callback],
            tb_log_name="DQN_Seaquest_Run9", 
            # CRUCIAL: Mantiene el total_timesteps global
            reset_num_timesteps=False 
        )
        
        # Guardado del modelo final 
        final_steps = START_TIMESTEPS + STEPS_TO_ADD
        final_path = os.path.join(BASE_LOG_DIR, "best_models", f"dqn_seaquest_run9_finished_{final_steps}_steps.zip")
        model.save(final_path)
        print("✅ Entrenamiento optimizado finalizado.")

    except KeyboardInterrupt:
        print("\n🛑 Pausa manual. Guardando último progreso...")
        # Usa el nuevo modelo 'model' que tiene el buffer_size correcto y los pesos transferidos
        model.save(os.path.join(CHECKPOINT_DIR, f"dqn_seaquest_run9_INTERRUPTED_{model.num_timesteps}_steps.zip"))