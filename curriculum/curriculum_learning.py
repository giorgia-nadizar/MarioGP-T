from __future__ import annotations
from typing import List, Tuple, Dict


def curriculum_learning_from_config(config: Dict) -> CurriculumLearning:
    level_prompts = config["levels"].split(";")
    levels = []
    for level_file in level_prompts:
        with open(f"levels/{level_file}.txt", "r") as file:
            levels.append(file.read())
    interval = config["n_generations"] / len(levels)
    return FixedIntervalCurriculumLearning(levels, interval)


class CurriculumLearning:

    def __init__(self, levels: List[str]):
        self.levels = levels
        self.level_idx = 0
        self.level = self.levels[self.level_idx]
        self.history = {
            0: self.level
        }

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        raise NotImplementedError

    def update_level(self, generation: int, best_fitnesses: List[float], solved: bool = False) -> Tuple[bool, str]:
        if solved or self.check_update_condition(generation, best_fitnesses):
            return self._increase_level(generation)
        else:
            return False, self.level

    def _increase_level(self, generation: int) -> Tuple[bool, str]:
        flag = False
        if self.level_idx < len(self.levels) - 1:
            self.level_idx += 1
            self.level = self.levels[self.level_idx]
            self.history[generation] = self.level
            flag = True
        return flag, self.level


class FixedIntervalCurriculumLearning(CurriculumLearning):

    def __init__(self, levels: List[str], interval: int):
        super().__init__(levels)
        self.upper_bounds = [interval * (x + 1) for x in range(len(levels))]

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        return generation >= self.upper_bounds[self.level_idx]


class FitnessBasedCurriculumLearning(CurriculumLearning):

    def __init__(self, levels: List[str], stagnation_based: bool, interval: int):
        super().__init__(levels)
        self.stagnation_based = stagnation_based
        self.interval = interval
        self.steps_until_next_update = self.interval

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        if self.steps_until_next_update > 0:
            return False
        stagnation = len(set(best_fitnesses[-self.interval:])) == 1
        return self.stagnation_based == stagnation

    def update_level(self, generation: int, best_fitnesses: List[float], solved: bool = False) -> Tuple[bool, str]:
        self.steps_until_next_update -= 1
        return super().update_level(generation, best_fitnesses)

    def _increase_level(self, generation: int) -> Tuple[bool, str]:
        self.steps_until_next_update = self.interval
        return super()._increase_level(generation)
