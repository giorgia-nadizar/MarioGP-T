import os

import jax.numpy as jnp

import cgpax
from cgpax.utils import cgp_expression_from_genome, compute_active_graph, lgp_expression_from_genome

if __name__ == '__main__':
    seed = 5
    policy_folder = f"results/lgp_curriculum_parallel_seed_{seed}_{seed}"

    config_file = "config.yaml"
    fitnesses_file = "fitnesses_667.npy"
    genotypes_file = "genotypes_667.npy"

    config = cgpax.get_config(f"{policy_folder}/config.yaml")
    fitnesses = jnp.load(os.path.join(policy_folder, fitnesses_file))
    genotypes = jnp.load(os.path.join(policy_folder, genotypes_file))

    best_genotype = genotypes[jnp.argmax(fitnesses)].astype(int)
    print(f"Fitness: {jnp.max(fitnesses)}")
    if "cgp" in policy_folder:
        n_nodes = config["n_nodes"]
        n_out = config["n_out"]
        x_genes, y_genes, f_genes, out_genes, weights = jnp.split(best_genotype, jnp.asarray(
            [n_nodes, 2 * n_nodes, 3 * n_nodes, 3 * n_nodes + n_out]))
        active = compute_active_graph(x_genes.astype(int), y_genes.astype(int), f_genes.astype(int), out_genes.astype(int),
                                      config)
        active_nodes = active[-n_nodes:]
        for idx, active in enumerate(active_nodes):
            if active:
                print(idx, x_genes[idx], y_genes[idx], f_genes[idx])

    if "cgp" in policy_folder:
        expression = cgp_expression_from_genome(best_genotype, config)
    else:
        expression = lgp_expression_from_genome(best_genotype, config)
    print(expression)