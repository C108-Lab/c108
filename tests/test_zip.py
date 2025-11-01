#
# C108 - Zip Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os

# Local ----------------------------------------------------------------------------------------------------------------
from c108.zip import untar_whitelist, zip_file_list

current_file_path = os.path.dirname(os.path.abspath(__file__))


# Tests ----------------------------------------------------------------------------------------------------------------


class TestZip:
    def test_untar_whitelist(self):
        print()
        data_directory = os.path.join(current_file_path, "test-files")
        tar_file = os.path.join(data_directory, "TEST-untar.tar.gz")
        out_dir = "/tmp/TEST-untar"
        res = untar_whitelist(
            file_name=tar_file, white_list=["README", "README-s"], out_dir=out_dir
        )
        print(f"Untar whitelist: {res}")

    def test_zip_file_list(self):
        zip_file = "/tmp/TEST-zip_file_list.zip"
        res = zip_file_list(
            file_list=[
                os.path.join(current_file_path, "test_abc.py"),
                os.path.join(current_file_path, "test_zip.py"),
            ],
            zip_file_path=zip_file,
        )
        print(f"Zip file list: {res}")
