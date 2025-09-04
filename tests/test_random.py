#
# C108 - Random Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
import random
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.random import random_factor


# Tests ----------------------------------------------------------------------------------------------------------------

class TestRandomFactor:

    @pytest.mark.parametrize(
        "start,end,factors,seed",
        [
            (0, 100, [2, 3], 123),
            (5, 37, None, 7),
            (-20, 85, [4], 42),
        ],
    )
    def test_seeded_matches_expected(self, start, end, factors, seed):
        # Compute expected using the same algorithm
        facs = [abs(int(f)) for f in (factors or [1])]
        lcm_val = 1
        for f in facs:
            lcm_val = math.lcm(lcm_val, f)
        min_mult = -(-start // lcm_val)
        max_mult = end // lcm_val
        assert min_mult <= max_mult  # sanity for these cases

        rng = random.Random(seed)
        expected = rng.randint(min_mult, max_mult) * lcm_val

        actual = random_factor(start=start, end=end, factors=factors, seed=seed)
        assert actual == expected
        assert start <= actual <= end
        assert actual % lcm_val == 0

    def test_inclusive_bounds_single_candidate(self):
        # start == end == multiple of LCM => must return that exact number
        result = random_factor(start=30, end=30, factors=[6, -5], seed=999)
        assert result == 30

    def test_no_solution_returns_none(self):
        # No integer in [1, 5] divisible by 6
        assert random_factor(start=1, end=5, factors=[6], seed=0) is None

    def test_unseeded_invariants_divisible_and_in_range(self):
        # Can't assert exact value; check invariants
        start, end, factors = -100, 100, [12, 5]
        res = random_factor(start=start, end=end, factors=factors, seed=None)
        lcm_val = math.lcm(1, *[abs(f) for f in factors])
        assert res is not None
        assert start <= res <= end
        assert res % lcm_val == 0

    def test_zero_factor_raises(self):
        with pytest.raises(ValueError, match=r"non-zero") as exc:
            random_factor(start=0, end=100, factors=[2, 0, 3], seed=1)

    def test_invalid_factors_type_raises(self):
        # Factors must be an iterable or None
        with pytest.raises(TypeError) as exc:
            random_factor(start=0, end=10, factors=123, seed=1)
        msg = str(exc.value).lower()
        # Be flexible across implementations/messages
        assert any(hint in msg for hint in ("iterable", "factors", "protocol"))

    def test_global_state_does_not_affect_seeded(self):
        # Seeding global RNG should not change deterministic output
        args = dict(start=0, end=10 ** 6, factors=[7, 5], seed=42)
        r1 = random_factor(**args)
        random.seed(999)
        r2 = random_factor(**args)
        assert r1 == r2