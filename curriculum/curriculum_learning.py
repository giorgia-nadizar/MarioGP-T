from __future__ import annotations
from typing import List, Tuple, Dict


def curriculum_learning_from_config(config: Dict) -> CurriculumLearning:
    level_prompts = config["levels"].split(";")
    levels_seeds = config.get("levels_seeds", "").split(";")
    levels = []
    for level_prompt in level_prompts:
        current_levels = []
        for level_seed in levels_seeds:
            with open(f"levels/{level_prompt}_{level_seed}.txt", "r") as file:
                current_levels.append(file.read())
        levels.append(current_levels)
    if config.get("gradual_level_increase", False):
        levels_with_gradual_increase = []
        for current_levels_idx in range(len(levels) - 1):
            current_levels = levels[current_levels_idx]
            next_levels = levels[current_levels_idx + 1]
            current_levels_mask = [True for _ in current_levels]
            levels_with_gradual_increase.append(current_levels)
            for mask_idx in range(len(current_levels_mask )- 1):
                current_levels_mask[mask_idx] = False
                current_gradual_levels = []
                for gradual_level_idx in range(len(current_levels_mask)):
                    current_gradual_levels.append(
                        current_levels[gradual_level_idx] if current_levels_mask[gradual_level_idx]
                        else next_levels[gradual_level_idx]
                    )
                levels_with_gradual_increase.append(current_gradual_levels)
        levels_with_gradual_increase.append(levels[len(levels) - 1])
        levels = levels_with_gradual_increase
    interval = config["n_generations"] / len(levels)
    return FixedIntervalCurriculumLearning(levels, interval)


class CurriculumLearning:

    def __init__(self, levels: List[List[str]]):
        self.levels_curriculum = levels
        self.level_idx = 0
        self.current_levels = self.levels_curriculum[self.level_idx]
        self.history = {
            0: self.current_levels
        }

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        raise NotImplementedError

    def update_level(self, generation: int, best_fitnesses: List[float],
                     solved: bool = False) -> Tuple[bool, List[str]]:
        if solved or self.check_update_condition(generation, best_fitnesses):
            return self._step(generation)
        else:
            return False, self.current_levels

    def _step(self, generation: int) -> Tuple[bool, List[str]]:
        flag = False
        if self.level_idx < len(self.levels_curriculum) - 1:
            self.level_idx += 1
            self.current_levels = self.levels_curriculum[self.level_idx]
            self.history[generation] = self.current_levels
            flag = True
        return flag, self.current_levels


class FixedIntervalCurriculumLearning(CurriculumLearning):

    def __init__(self, levels: List[List[str]], interval: int):
        super().__init__(levels)
        self.upper_bounds = [interval * (x + 1) for x in range(len(levels))]

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        return generation >= self.upper_bounds[self.level_idx]


class FitnessBasedCurriculumLearning(CurriculumLearning):

    def __init__(self, levels: List[List[str]], stagnation_based: bool, interval: int):
        super().__init__(levels)
        self.stagnation_based = stagnation_based
        self.interval = interval
        self.steps_until_next_update = self.interval

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        if self.steps_until_next_update > 0:
            return False
        stagnation = len(set(best_fitnesses[-self.interval:])) == 1
        return self.stagnation_based == stagnation

    def update_level(self, generation: int, best_fitnesses: List[float],
                     solved: bool = False) -> Tuple[bool, List[str]]:
        self.steps_until_next_update -= 1
        return super().update_level(generation, best_fitnesses)

    def _step(self, generation: int) -> Tuple[bool, List[str]]:
        self.steps_until_next_update = self.interval
        return super()._step(generation)
