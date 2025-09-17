import os

import jax.numpy as jnp

import cgpax
from cgpax.utils import cgp_expression_from_genome, compute_active_graph, lgp_expression_from_genome, \
    readable_lgp_program_from_genome

if __name__ == '__main__':
    for x in [-1,0,1]:
        print(x, jnp.log(jnp.abs(x)))



    seed = 2
    # policy_folder = f"results_50_100_False/connect4_trial_{seed}/"
    policy_folder = "results_c4"

    schedule = "10_0_90"
    config_file = f"config_{schedule}.yaml"
    fitnesses_file = f"fitnesses_{schedule}.npy"
    genotypes_file = f"genotypes_{schedule}.npy"

    fitnesses = jnp.load(os.path.join(policy_folder, fitnesses_file))
    genotypes = jnp.load(os.path.join(policy_folder, genotypes_file))

    best_genotype = genotypes[jnp.argmax(fitnesses)].astype(int)
    print(f"Fitness: {jnp.max(fitnesses)}")

    config = cgpax.get_config(f"{policy_folder}/{config_file}")

    if "cgp" in policy_folder:
        n_nodes = config["n_nodes"]
        n_out = config["n_out"]
        x_genes, y_genes, f_genes, out_genes, weights = jnp.split(best_genotype, jnp.asarray(
            [n_nodes, 2 * n_nodes, 3 * n_nodes, 3 * n_nodes + n_out]))
        active = compute_active_graph(x_genes.astype(int), y_genes.astype(int), f_genes.astype(int),
                                      out_genes.astype(int),
                                      config)
        active_nodes = active[-n_nodes:]
        for idx, active in enumerate(active_nodes):
            if active:
                print(idx, x_genes[idx], y_genes[idx], f_genes[idx])

    if "cgp" in policy_folder:
        expression = cgp_expression_from_genome(best_genotype, config)
    else:
        expression = lgp_expression_from_genome(best_genotype, config)

    if "cgp" not in policy_folder:
        program = readable_lgp_program_from_genome(best_genotype, config)
        program_lines = program.split("\n")
        for line in program_lines:
            for operation in ["<", ">", "*", "+", "-", "/"]:
                if operation in line:
                    line = line.replace(f"{operation}(", "").replace(")", "").replace(",", f" {operation}")
            print(line)

    for input_var in range(37, -1, -1):
        expression = expression.replace(f"i{input_var}", f"i_{{{input_var}}}")
    for output_var in range(35, -1, -1):
        expression = expression.replace(f"o{output_var}", f"o_{{{output_var}}}")
    expression_lines = expression.split("\n")
    for expression_line in expression_lines:
        if "= 0" in expression_line or "(0>0)" in expression_line or "(0*0)" in expression_line:
            continue
        else:
            # \alignedbox[c3!50]{o_2 }{= \text{abs}(i_{13}) }\\
            left, right = expression_line.split("=")
            output_id = int(left.replace("o_{","").replace("}","").strip())
            color_id = output_id % 6 + 1
            print(f"\\alignedbox[c{color_id}!50]{{{left}}}{{={right}}} \\\\")
            # print(expression_line)
