"""
Utilities for estimating network transfer timeouts and durations.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import os
from typing import Literal


# Methods --------------------------------------------------------------------------------------------------------------

def estimate_transfer_timeout(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        speed_mbps: float = 100.0,
        base_timeout_sec: float = 5.0,
        overhead_percent: float = 15.0,
        safety_multiplier: float = 2.0,
        protocol_overhead_sec: float = 2.0,
        min_timeout_sec: float = 10.0,
        max_timeout_sec: float | None = 3600.0,
) -> int:
    """
    Estimate a safe timeout value for transferring a file over a network.

    Calculates transfer time based on file size and network conditions, accounting
    for protocol overhead, connection latency, and network variability. Designed
    for HTTP uploads, API file transfers, and similar network operations.

    Args:
        file_path: Path to the file to be transferred. Either this or file_size
            must be provided.
        file_size: Size of the file in bytes. Either this or file_path must
            be provided. If both are given, file_size takes precedence.
        speed_mbps: Expected transfer speed in megabits per second (Mbps).
            Default is 100 Mbps (~12.5 MB/s), representing a typical broadband
            connection. Consider: 10-50 Mbps (slow), 100-300 Mbps (typical),
            500+ Mbps (fast connection).
        base_timeout_sec: Base timeout added to all transfers regardless of size.
            Accounts for DNS resolution, connection establishment, and initial
            handshake.
        overhead_percent: Additional time as percentage of transfer time to account
            for network protocol overhead (TCP acknowledgments, retransmissions, etc.).
        safety_multiplier: Multiplier applied to the calculated transfer time to
            provide a safety margin for network variability.
        protocol_overhead_sec: Fixed overhead for protocol-specific operations
            like chunked encoding, multipart boundaries, or API processing time.
        min_timeout_sec: Absolute minimum timeout value to return, regardless of
            calculated time.
        max_timeout_sec: Maximum timeout value to return. Prevents unreasonably
            long timeouts for very large files. The default of 3600 is 1-hour timeout.
            Set to None for no maximum.

    Returns:
        Estimated timeout in seconds as an integer, clamped between min_timeout_secs
        and max_timeout_secs.

    Raises:
        ValueError: If neither file_path nor file_size is provided, or if
            speed_mbps is not positive.
        FileNotFoundError: If file_path is provided but the file does not exist.
        OSError: If the file size cannot be determined.

    Examples:
        >>> # Using file path - small file on typical connection
        >>> estimate_transfer_timeout("config.json")
        10  # Returns minimum timeout for tiny files

        >>> # Using file size - 100MB file on slow connection
        >>> estimate_transfer_timeout(file_size=100*1024*1024, speed_mbps=10)
        133  # ~80s transfer + overhead + safety margin

        >>> # Large file with custom parameters
        >>> estimate_transfer_timeout(
        ...     file_size=5*1024**3,  # 5GB
        ...     speed_mbps=500,
        ...     safety_multiplier=2.0,
        ...     max_timeout_sec=7200
        ... )
        177  # Calculated timeout with 2x safety margin

        >>> # Conservative estimate for unreliable network
        >>> estimate_transfer_timeout(
        ...     "large_video.mp4",
        ...     speed_mbps=50,
        ...     overhead_percent=25.0,
        ...     safety_multiplier=2.5
        ... )
        # Returns appropriately conservative timeout

    Note:
        This provides an estimate based on idealized conditions. Actual transfer
        times vary significantly based on network congestion, server load, connection
        stability, and many other factors. Always test with real-world conditions
        and adjust parameters accordingly.

        For API uploads, consider setting protocol_overhead_secs higher (5-10s) to
        account for server-side processing. For direct file transfers, lower values
        (1-2s) may be sufficient.
    """
    # Validate inputs
    if file_path is None and file_size is None:
        raise ValueError("Either file_path or file_size must be provided")

    if speed_mbps <= 0:
        raise ValueError(f"speed_mbps must be positive, got {speed_mbps}")

    # Determine file size
    if file_size is not None:
        size_bytes = file_size
    else:
        size_bytes = os.path.getsize(file_path)

    # Convert file size to megabits
    size_mbits = (size_bytes * 8) / (1024 * 1024)

    # Calculate base transfer time in seconds
    transfer_time_secs = size_mbits / speed_mbps

    # Apply overhead percentage
    transfer_with_overhead = transfer_time_secs * (1.0 + overhead_percent / 100.0)

    # Apply safety multiplier
    safe_transfer_time = transfer_with_overhead * safety_multiplier

    # Calculate total timeout
    total_timeout = base_timeout_sec + protocol_overhead_sec + safe_transfer_time

    # Clamp to min/max bounds
    timeout = max(min_timeout_sec, total_timeout)
    if max_timeout_sec is not None:
        timeout = min(timeout, max_timeout_sec)

    return int(timeout)


def estimate_transfer_duration(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        speed_mbps: float = 100.0,
        overhead_percent: float = 15.0,
        unit: Literal["seconds", "minutes", "hours"] = "seconds",
) -> float:
    """
    Estimate the expected duration for a file transfer (without safety margins).

    Calculates realistic transfer time including network overhead, but without
    the safety multipliers used for timeout estimation. Useful for progress
    indicators, ETAs, and user-facing time estimates.

    Args:
        file_path: Path to the file to be transferred. Either this or file_size
            must be provided.
        file_size: Size of the file in bytes. Either this or file_path must
            be provided. If both are given, file_size takes precedence.
        speed_mbps: Expected transfer speed in megabits per second (Mbps).
            Default is 100 Mbps (~12.5 MB/s).
        overhead_percent: Additional time as percentage of transfer time to account
            for network protocol overhead. Default is 15%.
        unit: Unit for the returned duration. Options: "seconds", "minutes", "hours".
            Default is "seconds".

    Returns:
        Estimated transfer duration in the specified unit as a float.

    Raises:
        ValueError: If neither file_path nor file_size is provided, if
            speed_mbps is not positive, or if unit is invalid.
        FileNotFoundError: If file_path is provided but the file does not exist.
        OSError: If the file size cannot be determined.

    Examples:
        >>> # Estimate transfer time for a 500MB file
        >>> estimate_transfer_duration(file_size=500*1024*1024, speed_mbps=100)
        46.0  # seconds

        >>> # Get estimate in minutes for large file
        >>> estimate_transfer_duration(
        ...     file_size=5*1024**3,  # 5GB
        ...     speed_mbps=200,
        ...     unit="minutes"
        ... )
        3.83  # minutes

        >>> # Quick estimate for progress bar
        >>> duration = estimate_transfer_duration("backup.tar.gz", speed_mbps=50)
        >>> print(f"Estimated time: {duration:.1f} seconds")

    Note:
        This estimates expected transfer time without safety margins. For setting
        timeouts, use estimate_transfer_timeout() instead which includes appropriate
        buffers for network variability.
    """
    # Validate inputs
    if file_path is None and file_size is None:
        raise ValueError("Either file_path or file_size must be provided")

    if speed_mbps <= 0:
        raise ValueError(f"speed_mbps must be positive, got {speed_mbps}")

    if unit not in ("seconds", "minutes", "hours"):
        raise ValueError(f"unit must be 'seconds', 'minutes', or 'hours', got '{unit}'")

    # Determine file size
    if file_size is not None:
        size_bytes = file_size
    else:
        size_bytes = os.path.getsize(file_path)

    # Convert file size to megabits
    size_mbits = (size_bytes * 8) / (1024 * 1024)

    # Calculate transfer time with overhead
    transfer_time_secs = size_mbits / speed_mbps
    duration_secs = transfer_time_secs * (1.0 + overhead_percent / 100.0)

    # Convert to requested unit
    if unit == "seconds":
        return duration_secs
    elif unit == "minutes":
        return duration_secs / 60.0
    else:  # hours
        return duration_secs / 3600.0
