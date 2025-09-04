"""
C108 Random Generator Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
import random
from typing import Iterable

# Local ----------------------------------------------------------------------------------------------------------------
from c108.tools import fmt_value


# Methods --------------------------------------------------------------------------------------------------------------

def random_factor(
        start: int = 0,
        end: int = 10 ** 21,
        factors: Iterable[int] | None = None,
        seed: int | None = None) -> int | None:
    """
    Return a random integer N in the inclusive range [start, end] that is divisible
    by all given factors. If no such integer exists, return None.

    Args:
        start: Inclusive lower bound.
        end: Inclusive upper bound.
        factors: Iterable of non-zero integers. If None, treated as [1] (i.e., no constraint).
        seed: If provided, use a dedicated deterministic RNG seeded with this value.
              If None, use random.SystemRandom (cryptographically strong).

    Returns:
        An integer divisible by the LCM of factors within [start, end], or None if none exists.

    Raises:
        TypeError: If factors type unsupported.
        ValueError: If any factor is zero.
    """
    if not isinstance(factors, (Iterable, type(None))):
        raise TypeError(f"factors must be an iterable of integers or None: {fmt_value(factors)}")

    # Normalize factors
    factors = factors or [1]
    facs = [abs(int(f)) for f in factors]
    if any(f == 0 for f in facs):
        raise ValueError(f"factors must be non-zero: {fmt_value(factors)}")

    # Compute LCM of factors (LCM of empty sequence should behave like 1)
    lcm_val = 1
    for f in facs:
        lcm_val = math.lcm(lcm_val, f)

    # Compute multiplier bounds (ceil(start/lcm), floor(end/lcm))
    # lcm_val is positive here
    min_multiplier = -(-start // lcm_val)  # ceil division
    max_multiplier = end // lcm_val  # floor division

    if min_multiplier > max_multiplier:
        return None

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    multiplier = rng.randint(min_multiplier, max_multiplier)
    return multiplier * lcm_val
