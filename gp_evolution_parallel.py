import os
import time

import yaml
from jax import random
import jax.numpy as jnp
from cgpax.evaluation import evaluate_lgp_genome, parallel_evaluate_lgp_genomes
from cgpax.individual import generate_population
from cgpax.run_utils import update_config_with_env_data, compute_masks, compute_weights_mutation_function, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection
from cgpax.utils import CSVLogger
from mario_gym.mario_env import MarioEnv

# TODO find how many time steps are used in the mario game in java
# in java the best agent has 20 * 1000 as timer, and every tick decreases the timer by 30 (~700 time steps)


if __name__ == '__main__':

    config = {
        "n_rows": 20,
        "n_extra_registers": 5,
        "seed": 3,
        "n_individuals": 100,
        "solver": "lgp",
        "p_mut_lhs": 0.3,
        "p_mut_rhs": 0.1,
        "p_mut_functions": 0.1,
        "n_generations": 100,
        "selection": {
            "elite_size": 1,
            "type": "tournament",
            "tour_size": 2
        },
        "survival": "truncation",
        "crossover": False,
        "run_name": "trial",
        "obs_size": 8,
        "start_port": 25000,
        "genomes_path": "results/trial_3/genotypes.npy",
        "level": "----------------------------------------------------------------------------------------------------\n"
                 "----------------------------------------------------------------------------------------------------\n"
                 "----------------------------------------------------------------------------------------------------\n"
                 "----------------------------------------------------------------------------------------------------\n"
                 "---------------------------------------E------------------------------------------------------------\n"
                 "-----------------------------------SSSSSSSSSS-------------------------------------------------------\n"
                 "----------------------------------------------------------------------------------------------------\n"
                 "----------------------------------------------------------------------------------------------------\n"
                 "B--------------------------------------------------X------------------------------------------------\n"
                 "b------------------------------------------SSSSSSSSX-------------------------xxx--------------------\n"
                 "----?QQ-----------------------------xxxxx-----------------------------------xxX-x-------------------\n"
                 "x------------------xxx------xxxxx--xx-X--x-------------------------------B-xx-X--x------------------\n"
                 "-xxxxxxxxxxxxxxxxxxx--xxxxxxx----xxx--X---xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx--X---xxxxxxxxxxxxxxxxxx\n"
                 "XXXXXXXXXXXXXXXXXXXX--XXXXXXX--X-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXXXXXXXXX"
        # "level": "----------------------------------------------------------------------------------------------------\n"
        #          "----------------------------------------------------------------------------------------------------\n"
        #          "----------------------------------------------------------------------------------------------------\n"
        #          "----------------------------------------------------------------------------------------------------\n"
        #          "---------------------------------------E------------------------------------------------------------\n"
        #          "-----------------------------------SSSSSSSSSS-------------------------------------------------------\n"
        #          "----------------------------------------------------------------------------------------------------\n"
        #          "----------------------------------------------------------------------------------------------------\n"
        #          "B--------------------------------------------------X------------------------------------------------\n"
        #          "b------------------------------------xxx---SSSSSSSSX-------------------------xxx--------------------\n"
        #          "----?QQ-----------------------------xxX-x-----------------------------------xxX-x-------------------\n"
        #          "x------------------xxx------xxxxx--xx-X--x-------------------------------B-xx-X--x------------------\n"
        #          "-xxxxxxxxxxxxxxxxxxx--xxxxxxx----xxx--X---xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx--X---xxxxxxxxxxxxxxxxxx\n"
        #          "XXXXXXXXXXXXXXXXXXXX--XXXXXXX--X-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXXXXXXXXX"
    }

    run_name = f"{config['run_name']}_{config['seed']}"
    os.makedirs(f"results/{run_name}", exist_ok=True)

    rnd_key = random.PRNGKey(config["seed"])

    ports = [config["start_port"] + x for x in range(config["n_individuals"])]
    mario_template_env = MarioEnv.make(config["level"], observation_space_limit=config["obs_size"], port=ports[0])
    update_config_with_env_data(config, mario_template_env)

    genome_mask, mutation_mask = compute_masks(config)
    weights_mutation_function = compute_weights_mutation_function(config)
    select_parents = compile_parents_selection(config)
    select_survivals = compile_survival_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, weights_mutation_function)

    rnd_key, genome_key = random.split(rnd_key, 2)
    if config.get("genomes_path") is not None:
        genomes = jnp.load(config["genomes_path"])
    else:
        genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key,
                                      weights_mutation_function=weights_mutation_function)

    csv_logger = CSVLogger(
        filename=f"results/{run_name}/metrics{'_contd' if config.get('genomes_path') is not None else ''}.csv",
        header=["generation", "max_fitness", "max_percentage", "max_dead_time", "eval_time"]
    )

    for _generation in range(config["n_generations"]):
        print(_generation)
        start_eval = time.time()
        results = parallel_evaluate_lgp_genomes(genomes, config, ports, episode_length=1000)
        rearranged_results = {key: [i[key] for i in results] for key in results[0]}
        fitnesses, percentages, dead_times = jnp.asarray(rearranged_results["reward"]), jnp.asarray(
            rearranged_results["final_percentage"]), jnp.asarray(rearranged_results["dead_time"])

        end_eval = time.time()
        eval_time = end_eval - start_eval
        metrics = {
            "generation": _generation,
            "max_fitness": max(fitnesses),
            "max_percentage": max(percentages),
            "max_dead_time": max(dead_times),
            "eval_time": eval_time
        }
        csv_logger.log(metrics)

        # select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        parents = select_parents(genomes, fitnesses, select_key)

        # compute offspring
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

        # select survivals
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitnesses, survival_key)

        # update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))

    jnp.save(f"results/{run_name}/genotypes.npy", genomes)
    jnp.save(f"results/{run_name}/fitnesses.npy", fitnesses)
    file = open(f"results/{run_name}/config.yaml", "w")
    yaml.dump(config, file)
    file.close()

    mario_template_env.render()
    best_genome = genomes[jnp.argmax(fitnesses)]
    evaluate_lgp_genome(best_genome, config, mario_template_env, episode_length=1000)
    mario_template_env.stop_render()
