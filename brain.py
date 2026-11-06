"""A one-neuron brain: turn one observation into a movement decision."""

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Brain:
    weight: float
    bias: float

    def decide(self, food_offset: float) -> int:
        """Return -1 (left) or +1 (right), using learned parameters."""
        activation = self.weight * food_offset + self.bias
        return 1 if activation >= 0 else -1

    def mutate(self, rng: Random) -> "Brain":
        """Create a child; leave its parent unchanged."""
        return Brain(
            self.weight + rng.gauss(0, 0.3),
            self.bias + rng.gauss(0, 0.3),
        )
