import os

from cgpax import get_config
import jax.numpy as jnp

from cgpax.evaluation import evaluate_lgp_genome
from mario_gym.mario_env import MarioEnv

if __name__ == '__main__':
    folder_paths = ["curriculum_sequential_long_seed_0_0", "curriculum_parallel_long_seed_0_0",
                    "difficult_sequential_long_seed_0_0", "difficult_parallel_long_seed_0_0"]
    for folder_path in folder_paths:
        generation = "max"
        if generation == "max":
            generation = -1
            for file_name in os.listdir(f"results/{folder_path}"):
                if file_name.startswith("fitnesses"):
                    file_generation = int(file_name.replace(".npy", "").split("_")[1])
                    if file_generation > generation:
                        generation = file_generation

        config = get_config(f"results/{folder_path}/config.yaml")
        curriculum_config = get_config(f"results/{folder_path}/curriculum.yaml")
        curriculum_levels = curriculum_config.values()

        fitnesses = jnp.load(f"results/{folder_path}/fitnesses_{generation}.npy")
        genotypes = jnp.load(f"results/{folder_path}/genotypes_{generation}.npy")
        best_genome = genotypes[jnp.argmax(fitnesses)]

        for difficulty, levels in enumerate(curriculum_levels):
            print(difficulty)
            for inner_index, level in enumerate(levels):
                print(inner_index)
                mario_env = MarioEnv.make(level, observation_space_limit=config["obs_size"], port=config["start_port"])
                mario_env.render(delay=.000000000001)
                evaluate_lgp_genome(best_genome, config, mario_env, episode_length=500)
                mario_env.save_video(f"{os.getcwd()}/videos/{folder_path}_{difficulty}_{inner_index}.mp4")
                mario_env.stop_render()
