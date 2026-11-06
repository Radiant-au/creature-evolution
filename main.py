"""Run a reproducible evolution experiment: python3 main.py."""

from random import Random
from statistics import median

from brain import Brain
from world import WORLD_SIZE, evaluate, run_episode

POPULATION_SIZE = 20
GENERATIONS = 15
ELITE_COUNT = 4
SEED = 7

# All candidates face identical tasks, on both sides of the line.
TRAINING_SCENARIOS = [(5, 1), (5, 9), (2, 7), (8, 3)]
TEST_SCENARIOS = [(0, 10), (10, 0), (4, 6), (6, 4)]


def main() -> None:
    rng = Random(SEED)
    population = [Brain(rng.uniform(-1, 1), rng.uniform(-1, 1))
                  for _ in range(POPULATION_SIZE)]
    initial_best = max(population, key=lambda brain: evaluate(brain, TRAINING_SCENARIOS))
    print("A tiny creature learns which way to move through evolution.")
    print("Generation | Best fitness | Median fitness")

    for generation in range(GENERATIONS + 1):
        ranked = sorted(population, key=lambda brain: evaluate(brain, TRAINING_SCENARIOS), reverse=True)
        scores = [evaluate(brain, TRAINING_SCENARIOS) for brain in ranked]
        print(f"{generation:10} | {scores[0]:12.2f} | {median(scores):14.2f}")
        if generation == GENERATIONS:
            break
        # Elitism preserves winners; mutation explores nearby parameters.
        elites = ranked[:ELITE_COUNT]
        population = elites + [rng.choice(elites).mutate(rng)
                               for _ in range(POPULATION_SIZE - ELITE_COUNT)]

    champion = ranked[0]
    print(f"\nChampion: weight={champion.weight:.3f}, bias={champion.bias:.3f}")
    print(f"Initial best on held-out tasks: {evaluate(initial_best, TEST_SCENARIOS):.2f}")
    print(f"Champion on held-out tasks:     {evaluate(champion, TEST_SCENARIOS):.2f}")
    print("Held-out tasks were not used for selection. One seed is not proof of generalization.")
    print("\nReplay: C = creature, F = food, * = reached food")
    _, path = run_episode(champion, start=5, food=1)
    for step, position in enumerate(path):
        cells = ["·"] * WORLD_SIZE
        cells[1] = "F"
        cells[position] = "*" if position == 1 else "C"
        print(f"Step {step:2}: {' '.join(cells)}")


if __name__ == "__main__":
    main()
