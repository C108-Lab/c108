#
# C108 - Validators Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
from pytest import raises

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import validate_language_code


# Tests ----------------------------------------------------------------------------------------------------------------


class TestLanguage:
    def test_validate_language_code(self):
        # Allow Test
        assert validate_language_code("en") == "en"
        assert validate_language_code("en-US") == "en-US"
        # Disallow Test
        with raises(ValueError):
            validate_language_code("en-US", allow_BCP47=False)


#
