from typing import List, Tuple


class CurriculumLearning:

    def __init__(self, levels: List[str]):
        self.levels = levels
        self.level_idx = 0
        self.level = self.levels[self.level_idx]

    def check_update_condition(self, generation: int, best_fitnesses: List[float]) -> bool:
        raise NotImplementedError

    def update_level(self, generation: int, best_fitnesses: List[float]) -> Tuple[bool, str]:
        if self.check_update_condition(generation, best_fitnesses):
            return True, self.force_increase_level()
        else:
            return False, self.level

    def force_increase_level(self) -> str:
        if self.level_idx < len(self.levels) - 1:
            self.level_idx += 1
            self.level = self.levels[self.level_idx]
        return self.level


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

    def update_level(self, generation: int, best_fitnesses: List[float]) -> Tuple[bool, str]:
        self.steps_until_next_update -= 1
        return super().update_level(generation, best_fitnesses)

    def force_increase_level(self) -> str:
        self.steps_until_next_update = self.interval
        return super().force_increase_level()
