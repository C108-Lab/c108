#
# C108 Random Generator Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import random
from typing import Iterable
from math import gcd

# TODO fix for multiplatform
SIGNED_INT_MAX_X32 = 2 ** 31 - 1  # Max signed int on 32-bit OS, = 2_147_483_647
SIGNED_INT_MAX_X64 = 2 ** 63 - 1  # Max signed int on 64-bit OS, = 9_223_372_036_854_775_807


# Methods --------------------------------------------------------------------------------------------------------------

def random_factor(start: int = 0, end: int = SIGNED_INT_MAX_X64, factors: Iterable[int] = (), seed: int = None):
    """
    Generates a random number within a specified range
    that is divisible by all factors.

    Args:
        start (int, optional): The lower bound of the range (inclusive).
                               Defaults to 1.
        end (int, optional): The upper bound of the range (inclusive).
                              Defaults to 10000.
        factors (iterable, optional): An iterable of factors that the random
                                     number should be divisible by.
                                     Defaults to (2, 3, 7).
        seed (int, optional): Seed for the deterministic random number generator. If None, uses
                              the more secure system's entropy source.

    Returns:
        int: A random number divisible by Least Common Multiplier (LCM) within the
             specified range, or None if no such number exists.
    """

    factor = 1
    for f in factors:
        factor = (factor * f) // gcd(factor, f)

    # Calculate the valid range for the multiplier
    min_multiplier = -(-start // factor)
    max_multiplier = end // factor

    if min_multiplier <= max_multiplier:
        if seed is not None:
            random.seed(seed)
            multiplier = random.randint(min_multiplier, max_multiplier)
        else:
            multiplier = random.SystemRandom().randint(min_multiplier, max_multiplier)
            # WARNING:
            # Non-System pure Python random.randint() call can be cached by PyCharm+PyTest
            # or other Dev. environments, it may work like a constant-seed generator in tests.
            # This was the case with aiart.ImageGen.seed unit tests, but NO caching like that in Jupyter was found
        return multiplier * factor
    else:
        return None
