# Creature Evolution — lesson 1

A deliberately tiny starting point for an AI Creature Evolution Arena.
A creature moves along 11 cells to find food. Its brain is one artificial
neuron with two parameters. A genetic algorithm searches for useful parameters.
Python 3.10+; no dependencies, API key, server, or GPU. The project uses
[uv](https://docs.astral.sh/uv/) to manage its Python environment.

## Run

```bash
uv sync
uv run main.py
```

Run from this folder. You will see generation scores, a held-out evaluation,
and a text replay. Fitness is a task score, not a percentage or accuracy.

## Read the files in this order

| File | Why it exists | What to understand |
| --- | --- | --- |
| `brain.py` | Keeps decision-making separate from the world | `Brain` holds a weight and bias. `decide` computes weight × observation + bias and chooses left or right. `mutate` creates a slightly changed child. |
| `world.py` | Defines the problem independently of the training method | `run_episode` reads a sensor, asks the brain for an action, moves the creature, and scores the result. `evaluate` averages scores across tasks. |
| `main.py` | Connects brains, evaluation, and evolution | Create random brains, rank them, retain elites, mutate offspring, repeat, and evaluate the winner. The main guard prevents training when another file imports this module. |
| `.gitignore` | Keeps generated Python files out of commits | Python creates caches automatically; they are not source code. |
| `README.md` | Explains how to run and learn from the project | This guide. |

## Follow one decision

If the creature is at cell 5 and food is at cell 1, its observation is
`(1 - 5) / 10 = -0.4`. For a brain with weight `0.8` and bias `0.1`,
the activation is `0.8 * -0.4 + 0.1 = -0.22`. A negative activation
chooses left. The world applies that action and the creature moves to cell 4.
The numbers here are illustrative, not a hardcoded trained solution.

The world provides the food offset as a sensor. The brain chooses direction.
It does not receive fitness as input; evolution uses fitness to select parents.

## Follow one generation

1. Evaluate every brain on the same four training scenarios.
2. Sort by average fitness.
3. Keep four elites unchanged.
4. Select parents from those elites and perturb their weights/biases.
5. Evaluate the resulting population in the next generation.

This is neuroevolution: neural-network parameters change between generations.
It is not backpropagation or reinforcement learning. A creature does not update
its brain during an episode. We evolve brains, not body shapes.

## Honest limitations

This is intentionally a trivial learning problem: a positive weight and near-zero
bias already solve it. A random initial brain may solve it immediately, so the
best score need not improve. Median fitness shows population behavior, though
mutation means it can fall. Elitism keeps the best training fitness from falling.
The held-out scenarios provide a small sanity check, not statistical evidence.
The replay is plain text; Phaser will come later.

## Small experiments

- Change `SEED` in `main.py` and observe different runs.
- Change the mutation standard deviation in `brain.py` from `0.3` to `0.05`.
- Set `ELITE_COUNT` to 1, then compare results (keep it between 1 and 20).
- Change training scenarios while leaving test scenarios fixed.

Next lesson: move to a 2D grid, then introduce obstacles and richer sensors.
Keep rendering separate so the same world can eventually feed a Phaser viewer.

## Put this starter on GitHub

Create an empty repository called `creature-evolution` under your own account.
Do not initialize it with a README if using the commands below. In this folder:

```bash
git init -b main
git add .
git commit -m "Add minimal creature neuroevolution lesson"
git remote add origin https://github.com/YOUR_USERNAME/creature-evolution.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub login. Git will require authentication.
