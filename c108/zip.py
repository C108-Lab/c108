#
# C108 Zip Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
import tarfile
import zipfile
from os import PathLike
from typing import List

# Local Lib ------------------------------------------------------------------------------------------------------------
from .os import rm_dir_contents


def untar_whitelist(file_name: str | PathLike[str],
                    white_list: List[str],
                    out_dir: str | PathLike[str],
                    clean_dir: bool = False) -> List[str]:
    """Extract TAR or TAR.GZ archive to the dest directory, including only files that are white listed"""

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if clean_dir:
            rm_dir_contents(out_dir)

    extracted_list = []

    with tarfile.open(file_name, 'r:gz') as tar:
        for member in tar.getmembers():
            # looping over files and directories names in white_list
            for name in white_list:
                # case of a matching file or directory
                if name == member.name:
                    tar.extract(member, path=out_dir)
                    extracted_list.append(member.name)
                # case of a file within a directory in white_list >> extract all members in dir
                elif member.name.startswith(name):
                    tar.extract(member, path=out_dir)
                    extracted_list.append(member.name)

    return extracted_list


def zip_file_list(file_list: List[str], zip_file_path: str):
    # Remove the existing file if it exists
    # for Clarity of intentions
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)

    # Open the ZipFile object within the function.
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for file in file_list:
            zip_file.write(file)
