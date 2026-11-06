"""World rules. This module knows nothing about evolution or graphics."""

from brain import Brain

WORLD_SIZE = 11
MAX_STEPS = 10


def run_episode(brain: Brain, start: int, food: int) -> tuple[float, list[int]]:
    """Move toward one piece of food; return fitness and a replay path."""
    if not (0 <= start < WORLD_SIZE and 0 <= food < WORLD_SIZE):
        raise ValueError("Start and food must be inside the world")
    position = start
    path = [position]
    if position == food:
        return 20.0, path

    for _ in range(MAX_STEPS):
        # Normalize the sensor to [-1, 1]. Positive means food is right.
        observation = (food - position) / (WORLD_SIZE - 1)
        position += brain.decide(observation)
        position = max(0, min(WORLD_SIZE - 1, position))
        path.append(position)
        if position == food:
            # Reaching food quickly is better than reaching it slowly.
            return 20.0 - (len(path) - 1), path

    # Failed attempts get a negative score based on remaining distance.
    return -float(abs(food - position)), path


def evaluate(brain: Brain, scenarios: list[tuple[int, int]]) -> float:
    return sum(run_episode(brain, start, food)[0] for start, food in scenarios) / len(scenarios)
