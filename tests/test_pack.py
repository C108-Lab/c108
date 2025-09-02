#
# C108 - Pack Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import sys

# Local ----------------------------------------------------------------------------------------------------------------
from c108.pack import py_basename, py_package_version


# Tests ----------------------------------------------------------------------------------------------------------------

class TestPack:

    def test_py_basename(self):
        print()
        print(f"py_base_name: {py_basename(file_name="test_pack.py")}")

    def test_py_package_version(self):
        print()
        print(f"py_package_version(<module>): {py_package_version(sys)}")
        print(f"py_package_version(<str>)   : {py_package_version('sys')}")
