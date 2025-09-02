#
# C108 - OS Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os, sys
import time
from datetime import datetime

# Local ----------------------------------------------------------------------------------------------------------------
from c108.os import file_touch
from c108.tools import print_method


# Tests ----------------------------------------------------------------------------------------------------------------

class TestUtils:

    def test_file_touch(self, temp_file):
        print_method()
        print(f"sys.platform: {sys.version}")
        file_path = temp_file(size=108)
        # Should use getatime() not getmtime() for Access Dates as opposed to modified dates
        file_date1 = datetime.fromtimestamp(os.path.getatime(file_path))
        time.sleep(0.001)
        file_touch(file_path)
        file_date2 = datetime.fromtimestamp(os.path.getatime(file_path))
        print("access date 1", file_date1.isoformat(sep=" ", timespec="milliseconds"))
        print("access date 2", file_date2.isoformat(sep=" ", timespec="milliseconds"))
        assert file_date1 < file_date2
