#
# C108 - Random Tests
#

# Local ----------------------------------------------------------------------------------------------------------------
from c108.random import random_factor


# Tests ----------------------------------------------------------------------------------------------------------------

def test_random_factor():
    print()
    print(random_factor(start=0, end=108, factors=[3, 7]))
    print(random_factor(start=0, end=108, factors=[3, 7]))
    print(random_factor(start=0, end=108, factors=[3, 7], seed=52))
    print(random_factor(start=0, end=108, factors=[3, 7], seed=52))
