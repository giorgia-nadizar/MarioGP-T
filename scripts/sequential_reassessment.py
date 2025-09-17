import os

import pandas as pd

import cgpax
import jax.numpy as jnp

from cgpax.evaluation import evaluate_lgp_genome
from mario_gym.mario_env import MarioEnv

if __name__ == '__main__':
    seed = 0
    types = ['curriculum', 'difficult']
    for run_type in types:
        print(f"{seed} - {run_type}")
        base_folder = f"results/{run_type}_sequential_long_seed_{seed}_{seed}"
        progression_data = []
        config = cgpax.get_config(f"{base_folder}/config.yaml")
        levels = cgpax.get_config(f"{base_folder}/curriculum.yaml")
        change_gens = sorted(list(levels.keys()))
        level_id = 0
        current_levels = levels[change_gens[level_id]]
        checkpoints = {}
        for file_name in os.listdir(base_folder):
            if file_name.startswith("fitnesses"):
                generation = int(file_name.replace(".npy", "").split("_")[1])
                fitnesses = jnp.load(f'{base_folder}/{file_name}')
                genotypes = jnp.load(f'{base_folder}/genotypes_{generation}.npy')
                best_genotype = genotypes[jnp.argmax(fitnesses)]
                checkpoints[generation] = best_genotype
        generations = sorted(list(checkpoints.keys()))
        for gen in generations:
            if level_id + 1 < len(change_gens) and gen >= change_gens[level_id + 1]:
                level_id += 1
                current_levels = levels[change_gens[level_id]]
            best_genome = checkpoints[gen]
            print(f"{gen} / {max(generations)}")
            percentages = []
            for lvl in current_levels:
                mario_env = MarioEnv.make(lvl, observation_space_limit=config["obs_size"],
                                          port=config["start_port"] + seed)
                result = evaluate_lgp_genome(best_genome, config, mario_env)
                percentages.append(result["final_percentage"])
            progression_data.append({
                "generation": gen,
                "avg_percentage": sum(percentages) / len(percentages)
            })
        df = pd.DataFrame(progression_data)
        df.to_csv(f"{base_folder}/progression.csv", index=False)
