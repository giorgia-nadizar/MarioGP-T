import os
import sys
import time
from typing import Dict

import yaml
from jax import random
import jax.numpy as jnp

import cgpax
from cgpax.evaluation import parallel_evaluate_lgp_genomes
from cgpax.individual import generate_population
from cgpax.run_utils import update_config_with_env_data, compute_masks, compute_weights_mutation_function, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection
from cgpax.utils import CSVLogger
from curriculum.curriculum_learning import curriculum_learning_from_config
from mario_gym.mario_env import MarioEnv


def run(config: Dict):
    curriculum_learning = curriculum_learning_from_config(config)
    config["level"] = curriculum_learning.level

    run_name = f"{config['run_name']}_{config['seed']}"
    os.makedirs(f"results/{run_name}", exist_ok=True)

    rnd_key = random.PRNGKey(config["seed"])

    n_servers = config.get("n_servers", config["n_individuals"])
    ports = list(range(config["start_port"], config["start_port"] + n_servers))
    mario_template_env = MarioEnv.make(config["level"], observation_space_limit=config["obs_size"], port=ports[0])
    update_config_with_env_data(config, mario_template_env)

    genome_mask, mutation_mask = compute_masks(config)
    weights_mutation_function = compute_weights_mutation_function(config)
    select_parents = compile_parents_selection(config)
    select_survivals = compile_survival_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, weights_mutation_function)

    # note: this can be used for bootstrapping the initial population
    if config.get("genomes_path") is not None:
        genomes = jnp.load(config["genomes_path"])
    else:
        rnd_key, genome_key = random.split(rnd_key, 2)
        genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key,
                                      weights_mutation_function=weights_mutation_function)

    csv_logger = CSVLogger(
        filename=f"results/{run_name}/metrics{'_contd' if config.get('genomes_path') is not None else ''}.csv",
        header=["generation", "max_fitness", "max_percentage", "max_dead_time", "eval_time"]
    )

    best_fitnesses = []
    solved = False
    for _generation in range(config["n_generations"]):
        print(f"{_generation}/{config['n_generations']}")
        start_eval = time.time()
        results = parallel_evaluate_lgp_genomes(genomes, config, ports, episode_length=1000)
        rearranged_results = {key: [i[key] for i in results] for key in results[0]}
        _, percentages, dead_times = jnp.asarray(rearranged_results["reward"]), jnp.asarray(
            rearranged_results["final_percentage"]), jnp.asarray(rearranged_results["dead_time"])
        end_eval = time.time()
        eval_time = end_eval - start_eval

        # note: we take the fitness as the completion percentage and not the actual reward, it can be changed
        fitnesses = percentages
        best_fitnesses.append(max(fitnesses))
        metrics = {
            "generation": _generation,
            "max_fitness": max(fitnesses),
            "max_percentage": max(percentages),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time
        }
        csv_logger.log(metrics)

        solved = max(percentages) == 1. and config.get("adaptive", True)
        # note: perform update of level according to the curriculum, if needed/possible
        update_done, config["level"] = curriculum_learning.update_level(_generation, best_fitnesses, solved)
        if solved and not update_done:
            break

        # note: save genomes and fitnesses every 50 generations and when levels are solved/changed
        if solved or update_done or _generation % config.get("saving_interval", 50) == 0:
            jnp.save(f"results/{run_name}/genotypes_{_generation}.npy", genomes)
            jnp.save(f"results/{run_name}/fitnesses_{_generation}.npy", fitnesses)

        # note: select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        parents = select_parents(genomes, fitnesses, select_key)

        # note: compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        if config.get("crossover", False):
            parents1, parents2 = jnp.split(parents, 2)
            rnd_key, *xover_keys = random.split(rnd_key, len(parents1) + 1)
            offspring1, offspring2 = crossover_genomes(parents1, parents2, jnp.array(xover_keys))
            new_parents = jnp.concatenate((offspring1, offspring2))
        else:
            new_parents = parents
        offspring_matrix = mutate_genomes(new_parents, mutate_keys)
        offspring = jnp.reshape(offspring_matrix, (-1, offspring_matrix.shape[-1]))

        # note: select survivals
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitnesses, survival_key)

        # note: update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))

    # note: save information at the end of the run
    jnp.save(f"results/{run_name}/genotypes_{_generation}.npy", genomes)
    jnp.save(f"results/{run_name}/fitnesses_{_generation}.npy", fitnesses)
    with open(f"results/{run_name}/config.yaml", "w") as file:
        yaml.dump(config, file)
    with open(f"results/{run_name}/curriculum.yaml", "w") as file:
        yaml.dump(curriculum_learning.history, file)


if __name__ == '__main__':
    config_file = "configs/cv_adaptive_config.yaml"
    # note: read config file name if passed
    if len(sys.argv) > 1:
        config_file = f"configs/{sys.argv[1]}"
    configs = cgpax.process_dictionary(cgpax.get_config(config_file))
    for count, cfg in enumerate(configs):
        print(f"Run {count + 1}/{len(configs)} starting")
        run(cfg)
