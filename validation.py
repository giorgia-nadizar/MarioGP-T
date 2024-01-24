import os
from typing import List, Dict

import jax.numpy as jnp
import pandas as pd

import cgpax
from cgpax.evaluation import evaluate_lgp_genome
from mario_gym.mario_env import MarioEnv


def get_final_best(path: str) -> jnp.ndarray:
    last_generation = 0
    for filename in os.listdir(path):
        if filename.startswith("fitnesses"):
            generation = int(filename.replace("fitnesses_", "").replace(".npy", ""))
            if generation > last_generation:
                last_generation = generation
    genotypes = jnp.load(f"{path}/genotypes_{last_generation}.npy")
    fitnesses = jnp.load(f"{path}/fitnesses_{last_generation}.npy")
    return genotypes[jnp.argmax(fitnesses)]


def get_levels(validation_config: Dict) -> Dict[str, str]:
    levels = {}
    for train_level_prompt in validation_config["levels"].split(";"):
        for train_seed in validation_config["train_seeds"].split(";"):
            with open(f"levels/{train_level_prompt}_{train_seed}.txt", "r") as file:
                levels[f"train_{train_level_prompt}_{train_seed}"] = file.read()
    for test_level_prompt in validation_config["test_levels"].split(";"):
        for test_seed in validation_config["test_seeds"].split(";"):
            with open(f"test_levels/{test_level_prompt}_{test_seed}.txt", "r") as file:
                levels[f"test_{test_level_prompt}_{test_seed}"] = file.read()
    for original_level in validation_config["original_levels"].split(";"):
        with open(f"original_levels/lvl-{original_level}.txt", "r") as file:
            levels[f"original_{original_level}"] = file.read()
    return levels


if __name__ == '__main__':
    folders = []
    start_seed = 6
    extra = 3
    seed = start_seed + extra
    for set_up in ["curriculum_adapt", "difficult", "second_level", "third", "curriculum_seq", "curriculum_fixed"]:
        folders.append(f"results/updated_run_p1/{set_up}_seed_{seed}_{seed}")

    for folder_path in folders:
        print(folder_path)
        best_genome = get_final_best(folder_path)
        config = cgpax.get_config(f"{folder_path}/config.yaml")
        validation_levels = get_levels(cgpax.get_config("configs/validation_config.yaml"))
        validation_dicts = []
        for validation_prompt, validation_level in validation_levels.items():
            print(f"\t{validation_prompt}")
            mario_env = MarioEnv.make(validation_level, observation_space_limit=config["obs_size"],
                                      port=config["start_port"] + extra)
            result = evaluate_lgp_genome(best_genome, config, mario_env)
            result["prompt"] = validation_prompt
            validation_dicts.append(result)
        validation_df = pd.DataFrame(validation_dicts)
        validation_df.to_csv(f"{folder_path}/validation.csv", index=False)
