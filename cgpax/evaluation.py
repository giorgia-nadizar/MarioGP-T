from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import gymnasium
import jax.numpy as jnp

from cgpax.encoding import genome_to_cgp_program, genome_to_lgp_program
from mario_gym.mario_env import MarioEnv


def _evaluate_program(program: Callable, program_state_size: int, env: gymnasium.Env,
                      episode_length: int = 1000) -> Dict:
    obs, info = env.reset()
    program_state = jnp.zeros(program_state_size)
    cumulative_reward = 0.

    done_time = episode_length
    dead_time = episode_length
    final_percentage = 0
    for i in range(episode_length):
        inputs = jnp.asarray(obs)
        new_program_state, actions = program(inputs, program_state)
        boolean_actions = (actions > 0).tolist()
        obs, reward, done, truncated, info = env.step(boolean_actions)
        final_percentage = reward
        if not done:
            cumulative_reward -= (1. - reward)
        else:
            cumulative_reward -= (1. - reward) * (episode_length - i)
            done_time = i
            if truncated:
                dead_time = i
            break

    return {
        "reward": cumulative_reward,
        "done": done_time,
        "dead_time": dead_time,
        "final_percentage": final_percentage
    }


def evaluate_cgp_genome(genome: jnp.ndarray, config: Dict, env: gymnasium.Env,
                        episode_length: int = 1000,
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    return inner_evaluator(genome_to_cgp_program(genome, config), config["buffer_size"], env, episode_length)


def evaluate_lgp_genome(genome: jnp.ndarray, config: Dict, env: gymnasium.Env,
                        episode_length: int = 1000,
                        inner_evaluator: Callable = _evaluate_program) -> Dict:
    return inner_evaluator(genome_to_lgp_program(genome, config), config["n_registers"], env, episode_length)


def _evaluate_cgp_genome_for_parallels(config, episode_length, inner_evaluator,
                                       genome_env_pair: Tuple[jnp.ndarray, int]) -> Dict:
    genome, port = genome_env_pair
    env = MarioEnv.make(level=config["level"], observation_space_limit=config["obs_size"], port=port)
    return evaluate_cgp_genome(genome, config, env, episode_length, inner_evaluator)


def _evaluate_lgp_genome_for_parallels(config, episode_length, inner_evaluator,
                                       genome_env_pair: Tuple[jnp.ndarray, int]) -> Dict:
    genome, port = genome_env_pair
    env = MarioEnv.make(level=config["level"], observation_space_limit=config["obs_size"], port=port)
    return evaluate_lgp_genome(genome, config, env, episode_length, inner_evaluator)


def _parallel_evaluate_genomes(genomes: jnp.ndarray, ports: List[int], evaluator: Callable) -> List[Dict]:
    genome_port_pairs = [(genomes[i], ports[i % len(ports)]) for i in range(len(genomes))]

    results = []
    start_idx = 0
    with Pool(len(ports)) as p:
        while start_idx < len(genome_port_pairs):
            current_pairs = genome_port_pairs[start_idx: min(start_idx + len(ports), len(genome_port_pairs))]
            res = p.map(evaluator, current_pairs)
            results = results + res
            start_idx += len(ports)

    return results


def parallel_evaluate_cgp_genomes(genomes: jnp.ndarray, config: Dict, ports: List[int],
                                  episode_length: int = 1000,
                                  inner_evaluator: Callable = _evaluate_program) -> List[Dict]:
    eval_func = partial(_evaluate_cgp_genome_for_parallels, config, episode_length, inner_evaluator)
    return _parallel_evaluate_genomes(genomes, ports, eval_func)


def parallel_evaluate_lgp_genomes(genomes: jnp.ndarray, config: Dict, ports: List[int],
                                  episode_length: int = 1000,
                                  inner_evaluator: Callable = _evaluate_program) -> List[Dict]:
    eval_func = partial(_evaluate_lgp_genome_for_parallels, config, episode_length, inner_evaluator)
    return _parallel_evaluate_genomes(genomes, ports, eval_func)
