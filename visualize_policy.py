from cgpax import get_config
import jax.numpy as jnp

from cgpax.evaluation import evaluate_lgp_genome
from mario_gym.mario_env import MarioEnv

if __name__ == '__main__':
    folder_path = "results/trial_2"
    config = get_config(f"{folder_path}/config.yaml")
    fitnesses = jnp.load(f"{folder_path}/fitnesses.npy")
    genotypes = jnp.load(f"{folder_path}/genotypes.npy")

    mario_env = MarioEnv.make(config["level"], observation_space_limit=config["obs_size"], port=config["start_port"])
    mario_env.render()
    best_genome = genotypes[jnp.argmax(fitnesses)]
    evaluate_lgp_genome(best_genome, config, mario_env, episode_length=1000)
    mario_env.stop_render()
