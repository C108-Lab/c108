#
# C108 OS and Path Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import os, re, shutil, pwd, grp
import subprocess as ps

from datetime import datetime
from pathlib import Path

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import listify


# Methods --------------------------------------------------------------------------------------------------------------

def cp_owners(source=None, dest=None,
              options: list[str] = None,
              is_sudo: bool = False,
              quiet: bool = False):
    """
    Copy ownership from ``source`` to ``dest`` with ``chown`` subprocess.
    """
    if source is None or dest is None:
        raise ValueError('Source and dest must be specified')
    # Change File or Dir ownership to match the ownership of Channel dir
    options = listify(options) if options else []
    stat_info = os.stat(source)
    uid, gid = stat_info.st_uid, stat_info.st_gid
    usr_name, grp_name = pwd.getpwuid(uid).pw_name, grp.getgrgid(gid).gr_name
    cmd = ["sudo", "chown"] if is_sudo else ["chown"]
    cmd += options + [f"{usr_name}:{grp_name}", dest]
    response = ps.run(cmd, check=True, stdout=ps.DEVNULL, stderr=ps.DEVNULL)


def backup_file(path: str | os.PathLike[str],
                time_format: str = "%Y%m%d-%H%M%S",
                raise_exception: bool = False) -> bool:
    """
    Creates a backup of the input file by copying it with a timestamp.

    Args:
        path: Path to the file to be backed up
        time_format: strftime format for timestamp
        raise_exception: Whether to raise exceptions or return False

    Returns:
        bool: True if backup was created successfully, False otherwise

    Raises:
        FileNotFoundError: If source file doesn't exist (when raise_exception=True)
        FileExistsError: If backup file already exists (when raise_exception=True)
        Exception: If backup operation fails (when raise_exception=True)
    """

    # Validate Source file exists
    if not os.path.exists(path):
        error_msg = f"Source file not found: {path}"
        if raise_exception:
            raise FileNotFoundError(error_msg)
        print(f"Warning: {error_msg}")
        return False

    # Run Backup
    file_suffix = Path(path).suffix
    path_nosuffix = path[:-len(file_suffix)] if file_suffix else path
    backup_path = f"{path_nosuffix}.{datetime.now().strftime(time_format)}{file_suffix}"
    if os.path.exists(backup_path):
        if raise_exception:
            raise FileExistsError(f"Error: Backup File already exists: {backup_path}")
        print(f"Warning: Backup File already exists: {backup_path}")
        return False
    try:
        shutil.copy2(path, backup_path)
        return True
    except Exception as e:
        if raise_exception:
            raise Exception(f"Error: Failed to create backup: {e}") from e
        print(f"Warning: Failed to create backup: {e}")
        return False


def expand_realpath(path: str) -> str:
    real_path = os.path.expandvars(path)
    real_path = os.path.expanduser(real_path)
    real_path = os.path.realpath(real_path)
    return real_path


def file_exists(path: str, quiet: bool = True) -> bool:
    if not isinstance(path, str) and not quiet:
        raise ValueError("Valid <str> required")
    p_path = Path(path)
    return p_path.is_file()


def file_touch(path):
    """Touch a file. Update Access and Modified dates"""
    with open(path, 'a'):
        os.utime(path, None)


def filter_path(path: str):
    path = path.rstrip("/") or "."
    return path


def mk_dir(path: str):
    path = filter_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def rc_file_api_key(file_name: str, api_key_name: str) -> str:
    """Reads shell rc-file and extracts the value of the specified API key variable."""
    try:
        with open(file_name, "r") as f:
            content = f.read()

        # Regular expression to match the API key variable assignment
        pattern = rf"^\s*export\s+{api_key_name}\s*=\s*\"(.*)\""
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)  # Extract the API key value
        else:
            return ""  # API key not found
    except FileNotFoundError:
        return ""  # File not found


def rm_dir_contents(path: str | os.PathLike[str]) -> None:
    """
    Removes all files and subdirectories from a directory recursively,
    but leaves the directory itself intact.
    Args:
      path: The path to the directory to clean.
    """
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def tail_file(file_path, n: int = 1):
    lines = _ps_capture_stdout(["tail", "-n", str(n), file_path])
    return lines


# Private Methods ------------------------------------------------------------------------------------------------------

def _ps_capture_stdout(cmd: list | str, shell: bool = False) -> str:
    ps_responce = ps.run(cmd, shell=shell, capture_output=True, text=True)
    return ps_responce.stdout
