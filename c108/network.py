#
# C108 OS and Path Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import os


# Methods --------------------------------------------------------------------------------------------------------------

def transfer_timeout(file_path: str | os.PathLike[str], speed: int = 300,
                     base_timeout: int = 10, safety_factor: int = 10) -> int:
    """
    Calculates timeout estimate for a file upload.

    Args:
        file_path: The path to the file to be uploaded.
        speed: The realistic file transfer speed in Mbit/sec (mbps)
        base_timeout: The minimum timeout in seconds, accounting for API latency.
        safety_factor: The multiplier for the theoretical upload time to ensure a safe margin.

    Returns:
        A timeout estimate in seconds.
    """
    # Get file size in MegaBytes
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Convert file size from MegaBytes to Megabits (MB * 8)
    file_size_mbit = file_size_mb * 8

    # Calculate the theoretical best-case upload time in seconds
    theoretical_upload_time = file_size_mbit / speed

    # The timeout is the larger of the base timeout or the calculated safe time
    timeout_seconds = max(base_timeout, int(theoretical_upload_time * safety_factor))

    return timeout_seconds
