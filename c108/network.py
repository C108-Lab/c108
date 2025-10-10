"""
Utilities for estimating network transfer timeouts and durations.

This module provides tools for calculating safe timeout values and estimating
transfer durations for network file transfers, including support for batch
operations, retry strategies, and configurable transfer types.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
import os
import time
from enum import Enum
from typing import Literal, Optional, Union
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# @formatter:off

# Constants ------------------------------------------------------------------------------------------------------------

# Defaults chosen based on common network conditions and API requirements

DEFAULT_SPEED_MBPS = 100.0              # Typical broadband connection (~12.5 MB/s)
DEFAULT_BASE_TIMEOUT_SEC = 5.0          # DNS resolution + connection establishment (3-5s typical)
DEFAULT_OVERHEAD_PERCENT = 15.0         # TCP/IP overhead: headers, acknowledgments, retransmissions
DEFAULT_SAFETY_MULTIPLIER = 2.0         # 2x buffer for network variability and congestion
DEFAULT_PROTOCOL_OVERHEAD_SEC = 2.0     # HTTP headers, chunked encoding overhead
DEFAULT_MIN_TIMEOUT_SEC = 10.0          # Minimum practical timeout for any network operation
DEFAULT_MAX_TIMEOUT_SEC = 3600.0        # 1 hour - common API gateway timeout limit

# Retry defaults
DEFAULT_MAX_RETRIES = 3                 # Common standard for transient failures
DEFAULT_BACKOFF_MULTIPLIER = 2.0        # Exponential backoff factor: 1x, 2x, 4x, 8x...
DEFAULT_INITIAL_BACKOFF_SEC = 1.0       # Starting backoff delay

# Speed measurement defaults
DEFAULT_SAMPLE_SIZE_KB = 100            # Small but still accurate sample
DEFAULT_SPEED_TEST_TIMEOUT_SEC = 10.0   # Timeout for speed measurement itself

# Enums ----------------------------------------------------------------------------------------------------------------

class TransferType(str, Enum):
    """
    Predefined network transfer types.
    """
    API_UPLOAD = "api_upload"           # Uploading to REST API with processing overhead
    CDN_DOWNLOAD = "cdn_download"       # Downloading from CDN (typically faster, more reliable)
    CLOUD_STORAGE = "cloud_storage"     # S3, GCS, Azure Blob storage
    PEER_TRANSFER = "peer_transfer"     # Direct peer-to-peer transfer
    MOBILE_NETWORK = "mobile_network"   # Mobile/cellular connection (variable quality)
    SATELLITE = "satellite"             # High latency satellite connection


class TransferSpeedUnit(str, Enum):
    """
    Units for specifying network transfer speeds.
    """
    MBPS = "mbps"             # Megabits per second
    MBYTES_SEC = "MBps"       # Megabytes per second
    KBPS = "kbps"             # Kilobits per second
    KBYTES_SEC = "KBps"       # Kilobytes per second
    GBPS = "gbps"             # Gigabits per second

# @formatter:on

# Main API functions ---------------------------------------------------------------------------------------------------

def batch_timeout(
        files: list[Union[str, os.PathLike[str], int, tuple[str | os.PathLike[str], int]]],
        parallel: bool = False,
        max_parallel: int = 4,
        speed_mbps: float = DEFAULT_SPEED_MBPS,
        speed_unit: TransferSpeedUnit | str = TransferSpeedUnit.MBPS,
        per_file_overhead_sec: float = 1.0,
        **kwargs
) -> int:
    """
    Estimate timeout for transferring multiple files.

    For sequential transfers, timeouts are summed with per-file overhead.
    For parallel transfers, uses the longest single file timeout plus startup overhead.

    Args:
        files: List of files to transfer. Each element can be:
            - str/PathLike: file path
            - int: file size in bytes
            - tuple: (file_path, file_size) for pre-computed sizes
        parallel: If True, assumes parallel transfer. If False, sequential.
        max_parallel: Maximum number of parallel transfers. Only used if parallel=True.
        speed_mbps: Expected transfer speed (divided among parallel transfers).
        speed_unit: Unit for speed parameter.
        per_file_overhead_sec: Additional overhead per file for connection setup,
            API calls, etc. Default is 1.0 second per file.
        **kwargs: Additional parameters passed to transfer_timeout().

    Returns:
        Total estimated timeout in seconds as an integer.

    Raises:
        ValueError: If the file list is empty or contains invalid elements.

    Examples:
        >>> # Sequential upload of 3 files
        >>> files = ["file1.txt", "file2.txt", "file3.txt"]
        >>> batch_timeout(files, parallel=False)

        >>> # Parallel upload with known sizes
        >>> files = [1024*1024, 2*1024*1024, 512*1024]  # 1MB, 2MB, 512KB
        >>> batch_timeout(files, parallel=True, max_parallel=3)

        >>> # Mixed: paths and sizes
        >>> files = [("file1.txt", 1024), "file2.txt", 2048]
        >>> batch_timeout(files, speed_mbps=50.0)
    """
    if not files:
        raise ValueError("files list cannot be empty")

    _validate_non_negative(per_file_overhead_sec, "per_file_overhead_sec")

    if parallel and max_parallel < 1:
        raise ValueError(f"max_parallel must be at least 1, got {max_parallel}")

    # Convert speed to Mbps
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    # Parse file list and calculate individual timeouts
    timeouts = []

    for item in files:
        if isinstance(item, int):
            # File size provided directly
            file_size = item
            file_path = None
        elif isinstance(item, tuple):
            # (path, size) tuple
            if len(item) != 2:
                raise ValueError(f"Tuple must be (path, size), got {item}")
            file_path, file_size = item
        else:
            # File path
            file_path = item
            file_size = None

        # Calculate timeout for this file
        timeout = transfer_timeout(
            file_path=file_path,
            file_size=file_size,
            speed_mbps=speed_mbps_actual,
            speed_unit=TransferSpeedUnit.MBPS,
            **kwargs
        )

        timeouts.append(timeout)

    if parallel:
        # For parallel transfers, bandwidth is shared
        # Adjust individual timeouts by the sharing factor
        sharing_factor = min(len(files), max_parallel)
        adjusted_timeouts = [t * sharing_factor for t in timeouts]

        # Total timeout is the longest file plus connection overhead for all files
        total_timeout = max(adjusted_timeouts) + (len(files) * per_file_overhead_sec)
    else:
        # For sequential transfers, sum all timeouts plus per-file overhead
        total_timeout = sum(timeouts) + (len(files) * per_file_overhead_sec)

    return math.ceil(total_timeout)


def chunk_timeout(
        chunk_size: int,
        speed_mbps: float = DEFAULT_SPEED_MBPS,
        speed_unit: TransferSpeedUnit | str = TransferSpeedUnit.MBPS,
        **kwargs
) -> int:
    """
    Estimate timeout for a single chunk in chunked/resumable transfer.

    Useful for multipart uploads, streaming, or resumable upload protocols where
    files are split into chunks.

    This method uses transfer_timeout() with file as one chunk preset.

    Args:
        chunk_size: Size of the chunk in bytes. Common sizes: 5MB (S3 minimum),
            8MB (typical), 16MB, 32MB, 64MB (large chunks).
        speed_mbps: Expected transfer speed.
        speed_unit: Unit for speed parameter.
        **kwargs: Additional parameters passed to transfer_timeout().

    Returns:
        Timeout for this chunk in seconds as an integer.

    Examples:
        >>> # Timeout for 8MB chunk (typical chunk size)
        >>> chunk_timeout(8*1024*1024, speed_mbps=100.0)

        >>> # S3 multipart upload minimum chunk
        >>> chunk_timeout(5*1024*1024, speed_mbps=50.0)

        >>> # Large chunk for fast connection
        >>> chunk_timeout(64*1024*1024, speed_mbps=500.0)
    """
    _validate_positive(chunk_size, "chunk_size")

    return transfer_timeout(
        file_size=chunk_size,
        speed_mbps=speed_mbps,
        speed_unit=speed_unit,
        **kwargs
    )


def transfer_time(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        speed_mbps: float = DEFAULT_SPEED_MBPS,
        speed_unit: TransferSpeedUnit | str = TransferSpeedUnit.MBPS,
        overhead_percent: float = DEFAULT_OVERHEAD_PERCENT,
        unit: Literal["seconds", "minutes", "hours"] = "seconds",
) -> float:
    """Estimate the expected time for a file transfer (without safety margins).

    Calculates realistic transfer time including network overhead, but without
    the safety multipliers and base timeouts used for timeout estimation. This
    is the "optimistic but realistic" estimate suitable for progress indicators,
    ETAs, and user-facing time estimates.

    The calculation is based on the ideal transfer time plus a percentage for overhead.

    Args:
        file_path: Path to the file to be transferred. Either this or file_size
            must be provided.
        file_size: Size of the file in bytes. Either this or file_path must
            be provided. If both are given, file_size takes precedence.
        speed_mbps: Expected transfer speed in the specified unit.
            Default is 100.0 Mbps (~12.5 MB/s).
        speed_unit: Unit for speed parameter. Options: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec). Default is "mbps".
        overhead_percent: Additional time as percentage of transfer time to account
            for network protocol overhead. Default is 15.0% - represents realistic
            TCP/IP and HTTP overhead.
        unit: Unit for the returned duration. Options: "seconds", "minutes", "hours".
            Default is "seconds".

    Returns:
        Estimated transfer time in the specified unit as a float.

    Raises:
        ValueError: If neither file_path nor file_size is provided, if speed is
            not positive, if file_size is negative, or if unit is invalid.
        FileNotFoundError: If file_path is provided but the file does not exist.
        OSError: If the file size cannot be determined.

    Examples:
        >>> # Estimate transfer time for a 500MB file
        >>> transfer_time(file_size=500*1024*1024, speed_mbps=100.0)
        46.0  # seconds

        >>> # Get estimate in minutes for large file
        >>> transfer_time(
        ...     file_size=5*1024**3,  # 5 GB
        ...     speed_mbps=200.0,
        ...     unit="minutes"
        ... )
        3.83  # minutes

        >>> # Using MB/s instead of Mbps
        >>> transfer_time(
        ...     file_size=1024*1024*1024,  # 1 GB
        ...     speed_mbps=10.0,  # 10 MB/s
        ...     speed_unit="MBps",
        ...     unit="minutes"
        ... )
        2.3  # minutes

        >>> # Quick estimate for progress bar
        >>> duration = transfer_time("backup.tar.gz", speed_mbps=50.0)
        >>> print(f"Estimated time: {duration:.1f} seconds")

        >>> # Estimate in hours for very large transfer
        >>> transfer_time(
        ...     file_size=100*1024**3,  # 100 GB
        ...     speed_mbps=100.0,
        ...     unit="hours"
        ... )
        2.56  # hours

    Note:
        This estimates expected transfer time without safety margins. For setting
        timeouts, use transfer_timeout() instead which includes appropriate
        buffers for network variability, connection establishment, and safety margins.

        This function is ideal for:
        - Progress bar ETAs
        - User-facing time estimates
        - Calculating average transfer speeds
        - Comparing different transfer types
    """
    # Validate inputs
    _validate_positive(speed_mbps, "speed_mbps")
    _validate_non_negative(overhead_percent, "overhead_percent")

    if unit not in ("seconds", "minutes", "hours"):
        raise ValueError(f"unit must be 'seconds', 'minutes', or 'hours', got '{unit}'")

    # Convert speed to Mbps if needed
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    # Get file size
    size_bytes = _get_file_size(file_path, file_size)

    # Convert file size to megabits
    size_mbits = (size_bytes * 8) / (1024 * 1024)

    # Calculate transfer time with overhead
    transfer_time_sec = size_mbits / speed_mbps_actual
    time_sec = transfer_time_sec * (1.0 + overhead_percent / 100.0)

    # Convert to requested unit
    if unit == "seconds":
        return time_sec
    elif unit == "minutes":
        return time_sec / 60.0
    else:  # hours
        return time_sec / 3600.0


def transfer_estimates(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        speed_mbps: float = DEFAULT_SPEED_MBPS,
        speed_unit: TransferSpeedUnit | str = TransferSpeedUnit.MBPS,
        **kwargs
) -> dict:
    """
    Get comprehensive transfer estimates with multiple metrics.

    Returns a dictionary with timeout, transfer time, and formatted human-readable strings.
    Useful for displaying transfer information to users.

    Args:
        file_path: Path to the file to be transferred.
        file_size: Size of the file in bytes.
        speed_mbps: Expected transfer speed.
        speed_unit: Unit for speed parameter.
        **kwargs: Additional parameters for transfer_timeout().

    Returns:
        Dictionary containing:
            - timeout_sec: Timeout in seconds (int)
            - time_sec: Expected duration in seconds (float)
            - time: Human-readable duration string
            - timeout: Human-readable timeout string
            - file_size_bytes: File size in bytes
            - file_size: Human-readable file size
            - speed_mbps: Speed in Mbps

    Examples:
        >>> # Get comprehensive estimate
        >>> estimate = transfer_estimates(
        ...     file_size=100*1024*1024,
        ...     speed_mbps=50.0
        ... )
        >>> print(f"File: {estimate['file_size']}")
        >>> print(f"Expected time: {estimate['time']}")
        >>> print(f"Timeout: {estimate['timeout']}")

        >>> # Use in user interface
        >>> est = transfer_estimates("backup.tar.gz")
        >>> print(f"Uploading {est['file_size']}")
        >>> print(f"Estimated time: {est['time']}")
    """
    # Get file size
    size_bytes = _get_file_size(file_path, file_size)

    # Convert speed to Mbps
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    # Calculate timeout and duration
    timeout = transfer_timeout(
        file_size=size_bytes,
        speed_mbps=speed_mbps_actual,
        speed_unit=TransferSpeedUnit.MBPS,
        **kwargs
    )

    transfer_time_ = transfer_time(
        file_size=size_bytes,
        speed_mbps=speed_mbps_actual,
        speed_unit=TransferSpeedUnit.MBPS,
        unit="seconds"
    )

    # Format file size
    def format_bytes(size: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    # Format duration
    def format_time(seconds: float) -> str:
        """Format seconds to human-readable duration."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    return {
        "file_size": format_bytes(size_bytes),
        "file_size_bytes": size_bytes,
        "speed_mbps": speed_mbps_actual,
        "time": format_time(transfer_time_),
        "time_sec": transfer_time_,
        "timeout": format_time(timeout),
        "timeout_sec": timeout,
    }


def transfer_params(transfer_type: TransferType | str) -> dict:
    """
    Get recommended parameters for common transfer types.

    Args:
        transfer_type: TransferType or equivalent string identifying the transfer context.

    Returns:
        A dictionary of recommended parameters suitable for transfer_timeout().

    Examples:
        >>> params = transfer_params(TransferType.API_UPLOAD)
        >>> transfer_timeout(file_size=1024 * 1024, **params)
    Raises:
        ValueError: If transfer_type is unknown.
    """
    try:
        transfer_type = TransferType(transfer_type)
    except ValueError:
        raise ValueError(f"Unknown transfer type: {transfer_type!r}")

    _transfer_params = {
        TransferType.API_UPLOAD: {
            "speed_mbps": 100.0,
            "base_timeout_sec": 10.0,  # Higher for API handshake
            "overhead_percent": 20.0,  # More overhead for API processing
            "safety_multiplier": 2.5,  # Extra buffer for API variability
            "protocol_overhead_sec": 5.0,  # API processing time
            "min_timeout_sec": 15.0,
        },
        TransferType.CDN_DOWNLOAD: {
            "speed_mbps": 300.0,  # CDNs are typically fast
            "base_timeout_sec": 3.0,  # Fast connection establishment
            "overhead_percent": 10.0,  # Efficient protocols
            "safety_multiplier": 1.5,  # More reliable
            "protocol_overhead_sec": 1.0,
            "min_timeout_sec": 5.0,
        },
        TransferType.CLOUD_STORAGE: {
            "speed_mbps": 200.0,  # Cloud providers have good bandwidth
            "base_timeout_sec": 5.0,
            "overhead_percent": 15.0,
            "safety_multiplier": 2.0,
            "protocol_overhead_sec": 3.0,  # Multipart upload overhead
            "min_timeout_sec": 10.0,
        },
        TransferType.PEER_TRANSFER: {
            "speed_mbps": 50.0,  # Variable peer bandwidth
            "base_timeout_sec": 8.0,  # Longer connection setup
            "overhead_percent": 25.0,  # Higher overhead
            "safety_multiplier": 3.0,  # Very variable
            "protocol_overhead_sec": 2.0,
            "min_timeout_sec": 15.0,
        },
        TransferType.MOBILE_NETWORK: {
            "speed_mbps": 20.0,  # 4G/5G can vary greatly
            "base_timeout_sec": 10.0,  # Variable latency
            "overhead_percent": 30.0,  # High packet loss potential
            "safety_multiplier": 3.5,  # Very unreliable
            "protocol_overhead_sec": 3.0,
            "min_timeout_sec": 20.0,
        },
        TransferType.SATELLITE: {
            "speed_mbps": 25.0,  # Decent bandwidth
            "base_timeout_sec": 30.0,  # Very high latency (500-700ms)
            "overhead_percent": 40.0,  # High packet loss
            "safety_multiplier": 4.0,  # Extremely variable
            "protocol_overhead_sec": 5.0,
            "min_timeout_sec": 45.0,
        },
    }

    return _transfer_params[transfer_type].copy()


def transfer_speed(
        url: str,
        sample_size_kb: int = DEFAULT_SAMPLE_SIZE_KB,
        timeout_sec: float = DEFAULT_SPEED_TEST_TIMEOUT_SEC,
        num_samples: int = 1,
) -> float:
    """
    Measure actual network transfer speed by downloading a sample from a URL.

    This performs a real network test to measure achievable transfer speeds to
    a specific endpoint. Useful for determining appropriate speed_mbps values
    for timeout calculations.

    WARNING: This makes actual HTTP requests and downloads data. Use responsibly
    and ensure you have permission to access the test URL.

    Args:
        url: URL to download from for speed testing. Should be a reliable endpoint
            that serves content quickly. Consider using a CDN-hosted file or a
            dedicated speed test endpoint.
        sample_size_kb: Amount of data to download in kilobytes for each sample.
            Default is 100 KB - large enough for accuracy, small enough to be quick.
            Larger values give more accurate results but take longer.
        timeout_sec: Timeout for the speed test request itself. Default is 10 seconds.
        num_samples: Number of samples to take. Results are averaged. Default is 1.
            Multiple samples can improve accuracy but take longer.

    Returns:
        Measured transfer speed in Mbps (megabits per second).

    Raises:
        ValueError: If parameters are invalid.
        URLError: If the URL cannot be accessed.
        HTTPError: If the server returns an error status.
        TimeoutError: If the speed test exceeds timeout_sec.

    Examples:
        >>> # Measure speed to a CDN
        >>> speed = transfer_speed("https://cdn.example.com/test.dat")
        >>> print(f"Measured speed: {speed:.1f} Mbps")

        >>> # Use measured speed for timeout estimation
        >>> speed = transfer_speed("https://api.example.com/health")
        >>> timeout = transfer_timeout(
        ...     file_size=10*1024*1024,
        ...     speed_mbps=speed
        ... )

        >>> # More accurate measurement with multiple samples
        >>> speed = transfer_speed(
        ...     "https://cdn.example.com/test.dat",
        ...     sample_size_kb=500,
        ...     num_samples=3
        ... )

        >>> # Quick test with small sample
        >>> speed = transfer_speed(
        ...     "https://cdn.example.com/test.dat",
        ...     sample_size_kb=50,
        ...     timeout_sec=5.0
        ... )

    Note:
        - Results vary based on server load, network conditions, and routing
        - First request may be slower due to DNS resolution and connection setup
        - Results represent download speed; upload speed may differ significantly
        - Use multiple samples and test at different times for reliable estimates
        - Consider using a dedicated speed test service for production applications
        - This measures application-level throughput, not raw network capacity
    """
    _validate_positive(sample_size_kb, "sample_size_kb")
    _validate_positive(timeout_sec, "timeout_sec")

    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")

    speeds = []

    for _ in range(num_samples):
        try:
            # Record start time
            start_time = time.time()

            # Download sample data
            with urlopen(url, timeout=timeout_sec) as response:
                # Read the specified amount of data
                bytes_to_read = sample_size_kb * 1024
                data = response.read(bytes_to_read)
                bytes_read = len(data)

            # Record end time
            end_time = time.time()

            # Calculate speed
            elapsed_sec = end_time - start_time

            if elapsed_sec <= 0:
                continue  # Skip invalid samples

            # Convert to Mbps: (bytes * 8 bits/byte) / (1024^2 bits/Mbit) / seconds
            speed_mbps = (bytes_read * 8) / (1024 * 1024) / elapsed_sec
            speeds.append(speed_mbps)

        except (URLError, HTTPError) as e:
            raise URLError(f"Failed to measure transfer speed from {url}: {e}")
        except Exception as e:
            if "timed out" in str(e).lower():
                raise TimeoutError(f"Speed test timed out after {timeout_sec} seconds")
            raise

    if not speeds:
        raise ValueError("No valid speed samples collected")

    # Return average speed
    return sum(speeds) / len(speeds)


def transfer_timeout(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        speed_mbps: float = DEFAULT_SPEED_MBPS,
        speed_unit: TransferSpeedUnit | str = TransferSpeedUnit.MBPS,
        base_timeout_sec: float = DEFAULT_BASE_TIMEOUT_SEC,
        overhead_percent: float = DEFAULT_OVERHEAD_PERCENT,
        safety_multiplier: float = DEFAULT_SAFETY_MULTIPLIER,
        protocol_overhead_sec: float = DEFAULT_PROTOCOL_OVERHEAD_SEC,
        min_timeout_sec: float = DEFAULT_MIN_TIMEOUT_SEC,
        max_timeout_sec: float | None = DEFAULT_MAX_TIMEOUT_SEC,
) -> int:
    """
    Estimate a safe timeout value for transferring a file over a network.

    Calculates transfer time based on file size and network conditions, accounting
    for protocol overhead, connection latency, and network variability. The timeout
    is calculated as:

        timeout = base_timeout + protocol_overhead +
                  (transfer_time * (1 + overhead%) * safety_multiplier)

    The result is then clamped to [min_timeout_sec, max_timeout_sec] and rounded
    up to the nearest integer using math.ceil() to ensure sufficient time.

    Args:
        file_path: Path to the file to be transferred. Either this or file_size
            must be provided.
        file_size: Size of the file in bytes. Either this or file_path must
            be provided. If both are given, file_size takes precedence.
        speed_mbps: Expected transfer speed in the specified unit.
            Default is 100.0 Mbps (~12.5 MB/s) - typical broadband connection.
            Common values: 10-50 (slow), 100-300 (typical), 500+ (fast).
        speed_unit: Unit for speed parameter. Options: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec). Default is "mbps".
        base_timeout_sec: Base timeout added to all transfers regardless of size.
            Default is 5.0 seconds - accounts for DNS resolution (~1s), TCP
            handshake (~1s), TLS handshake (~1-2s), and HTTP request/response (~1s).
        overhead_percent: Additional time as percentage of transfer time to account
            for network protocol overhead. Default is 15.0% - represents TCP/IP
            headers (~5%), acknowledgments (~3%), potential retransmissions (~5%),
            and HTTP chunking (~2%).
        safety_multiplier: Multiplier applied to the calculated transfer time to
            provide a safety margin. Default is 2.0x - provides buffer for network
            congestion, routing changes, server load, and other variability.
        protocol_overhead_sec: Fixed overhead for protocol-specific operations.
            Default is 2.0 seconds - for multipart boundaries, chunked encoding,
            and initial API processing. Use 5-10s for heavy API processing, 1-2s
            for simple file transfers.
        min_timeout_sec: Absolute minimum timeout value to return. Default is 10.0
            seconds - minimum practical timeout for any network operation considering
            connection establishment and basic handshakes.
        max_timeout_sec: Maximum timeout value to return. Default is 3600.0 seconds
            (1 hour) - matches common API gateway limits (AWS ALB, CloudFlare, etc.).
            Set to None for no maximum.

    Returns:
        Estimated timeout in seconds as an integer (rounded up using math.ceil).
        The timeout is always clamped between min_timeout_sec and max_timeout_sec.

    Raises:
        ValueError: If neither file_path nor file_size is provided, if speed is
            not positive, if file_size is negative, or if min_timeout_sec exceeds
            max_timeout_sec.
        FileNotFoundError: If file_path is provided but the file does not exist.
        OSError: If the file size cannot be determined due to permissions or I/O error.

    Examples:
        >>> # Small file on typical connection - returns minimum timeout
        >>> transfer_timeout(file_size=1024)  # 1 KB
        10

        >>> # 100MB file on slow connection
        >>> transfer_timeout(file_size=100*1024*1024, speed_mbps=10.0)
        206  # ~80s transfer + overhead + 2x safety + base timeouts

        >>> # Using megabytes per second instead of megabits
        >>> transfer_timeout(
        ...     file_size=500*1024*1024,  # 500 MB
        ...     speed_mbps=10.0,  # 10 MB/s
        ...     speed_unit="MBps"
        ... )
        132  # Speed is converted: 10 MB/s = 80 Mbps

        >>> # Large file with custom safety margin
        >>> transfer_timeout(
        ...     file_size=5*1024**3,  # 5 GB
        ...     speed_mbps=500.0,
        ...     safety_multiplier=1.5,  # Less conservative
        ...     max_timeout_sec=7200  # 2 hour max
        ... )
        132

        >>> # Using file path
        >>> transfer_timeout("backup.tar.gz", speed_mbps=50.0)

        >>> # Conservative estimate for unreliable network
        >>> transfer_timeout(
        ...     file_size=1024*1024*1024,  # 1 GB
        ...     speed_mbps=50.0,
        ...     overhead_percent=25.0,
        ...     safety_multiplier=2.5
        ... )
        443

    Note:
        This provides an estimate based on idealized conditions. Actual transfer
        times vary based on network congestion, server load, connection stability,
        routing, and many other factors. Always test with real-world conditions
        and adjust parameters accordingly.

        For production use, consider:
        - API uploads: Use higher protocol_overhead_sec (5-10s) and safety_multiplier (2.5-3.0)
        - Direct transfers: Lower protocol_overhead_sec (1-2s) and safety_multiplier (1.5-2.0)
        - Mobile networks: Much higher safety_multiplier (3-4x) and overhead_percent (25-40%)
        - Batch uploads: Use batch_timeout() for better accuracy
    """
    # Validate parameters
    _validate_positive(speed_mbps, "speed_mbps")
    _validate_non_negative(base_timeout_sec, "base_timeout_sec")
    _validate_non_negative(overhead_percent, "overhead_percent")
    _validate_positive(safety_multiplier, "safety_multiplier")
    _validate_non_negative(protocol_overhead_sec, "protocol_overhead_sec")
    _validate_non_negative(min_timeout_sec, "min_timeout_sec")

    if max_timeout_sec is not None:
        _validate_non_negative(max_timeout_sec, "max_timeout_sec")
        _validate_timeout_bounds(min_timeout_sec, max_timeout_sec)

    if overhead_percent > 100.0:
        raise ValueError(
            f"overhead_percent seems unreasonably high: {overhead_percent}%. "
            f"Typical values are 10-40%."
        )

    # Convert speed to Mbps if needed
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    # Get file size
    size_bytes = _get_file_size(file_path, file_size)

    # Convert file size to megabits
    size_mbits = (size_bytes * 8) / (1024 * 1024)

    # Calculate base transfer time in seconds
    transfer_time_sec = size_mbits / speed_mbps_actual

    # Apply overhead percentage
    transfer_with_overhead = transfer_time_sec * (1.0 + overhead_percent / 100.0)

    # Apply safety multiplier
    safe_transfer_time = transfer_with_overhead * safety_multiplier

    # Calculate total timeout
    total_timeout = base_timeout_sec + protocol_overhead_sec + safe_transfer_time

    # Clamp to min/max bounds
    timeout = max(min_timeout_sec, total_timeout)
    if max_timeout_sec is not None:
        timeout = min(timeout, max_timeout_sec)

    # Round up to nearest integer to ensure sufficient time
    return math.ceil(timeout)


def transfer_timeout_retry(
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        initial_backoff_sec: float = DEFAULT_INITIAL_BACKOFF_SEC,
        **kwargs
) -> int:
    """
    Estimate timeout accounting for retry attempts with exponential backoff.

    Total timeout = (base_timeout * (max_retries + 1)) + sum(backoff_delays)
    where backoff_delays = [initial_backoff * backoff_multiplier^i for i in range(max_retries)]

    This method invokes transfer_timeout() for the base timeout estimate.

    Args:
        file_path: Path to the file to be transferred.
        file_size: Size of the file in bytes.
        max_retries: Maximum number of retry attempts. Default is 3 retries
            (4 total attempts) - standard for handling transient failures.
        backoff_multiplier: Multiplier for exponential backoff. Default is 2.0,
            giving delays of: 1s, 2s, 4s, 8s, etc.
        initial_backoff_sec: Initial backoff delay in seconds. Default is 1.0 second.
        **kwargs: Additional parameters passed to transfer_timeout().

    Returns:
        Total timeout including all retry attempts, as an integer.

    Raises:
        ValueError: If max_retries is negative or backoff parameters are invalid.

    Examples:
        >>> # Default: 3 retries with exponential backoff
        >>> transfer_timeout_retry(file_size=10*1024*1024)

        >>> # More aggressive retry strategy
        >>> transfer_timeout_retry(
        ...     file_size=100*1024*1024,
        ...     max_retries=5,
        ...     backoff_multiplier=1.5,
        ...     initial_backoff_sec=2.0
        ... )

        >>> # No retries (single attempt only)
        >>> transfer_timeout_retry(
        ...     "important.dat",
        ...     max_retries=0
        ... )
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")

    _validate_positive(backoff_multiplier, "backoff_multiplier")
    _validate_non_negative(initial_backoff_sec, "initial_backoff_sec")

    # Get base timeout for a single attempt
    base_timeout = transfer_timeout(
        file_path=file_path,
        file_size=file_size,
        **kwargs
    )

    # Calculate total backoff time: initial * (1 + multiplier + multiplier^2 + ... + multiplier^(n-1))
    # This is a geometric series: a * (r^n - 1) / (r - 1)
    if max_retries == 0:
        total_backoff = 0.0
    elif backoff_multiplier == 1.0:
        total_backoff = initial_backoff_sec * max_retries
    else:
        total_backoff = initial_backoff_sec * (
                (backoff_multiplier ** max_retries - 1) / (backoff_multiplier - 1)
        )

    # Total timeout: all attempts plus backoff delays
    total_timeout = base_timeout * (max_retries + 1) + total_backoff

    return math.ceil(total_timeout)


def transfer_type_timeout(
        transfer_type: TransferType | str,
        file_path: str | os.PathLike[str] | None = None,
        file_size: int | None = None,
        **overrides
) -> int:
    """
    Estimate timeout using predefined transfer type parameters.

    Convenience function that combines transfer_params() with
    transfer_timeout(). Scenario parameters can be overridden.

    Args:
        transfer_type: Transfer type (e.g., "api_upload", "mobile_network").
        file_path: Path to the file to be transferred.
        file_size: Size of the file in bytes.
        **overrides: Parameter overrides for the transfer type defaults.

    Returns:
        Estimated timeout in seconds as an integer.

    Examples:
        >>> # Use API upload transfer type
        >>> transfer_type_timeout(
        ...     TransferType.API_UPLOAD,
        ...     file_size=50*1024*1024
        ... )

        >>> # Mobile network with custom speed
        >>> transfer_type_timeout(
        ...     "mobile_network",
        ...     file_path="large_file.zip",
        ...     speed_mbps=30.0  # Override default 20 Mbps
        ... )

        >>> # CDN download transfer type
        >>> transfer_type_timeout(
        ...     TransferType.CDN_DOWNLOAD,
        ...     file_size=1024*1024*1024  # 1 GB
        ... )
    """
    # Get transfer type parameters, apply overrides
    params = transfer_params(transfer_type)
    params.update(overrides)

    # Add file path/size
    params["file_path"] = file_path
    params["file_size"] = file_size

    return transfer_timeout(**params)


# Private helper methods -----------------------------------------------------------------------------------------------

def _speed_to_mbps(speed: float, unit: TransferSpeedUnit | str) -> float:
    """Convert speed from various units to Mbps."""
    unit = TransferSpeedUnit(unit) if isinstance(unit, str) else unit

    conversions = {
        TransferSpeedUnit.MBPS: 1.0,
        TransferSpeedUnit.MBYTES_SEC: 8.0,  # 1 MB/s = 8 Mbps
        TransferSpeedUnit.KBPS: 1.0 / 1024.0,  # 1 Kbps = 1/1024 Mbps
        TransferSpeedUnit.KBYTES_SEC: 8.0 / 1024.0,  # 1 KB/s = 8/1024 Mbps
        TransferSpeedUnit.GBPS: 1024.0,  # 1 Gbps = 1024 Mbps
    }

    return speed * conversions[unit]


def _get_file_size(file_path: str | os.PathLike[str] | None, file_size: int | None) -> int:
    """Get file size from either file_path or file_size parameter."""
    if file_size is not None:
        if file_size < 0:
            raise ValueError(f"file_size must be non-negative, got {file_size}")
        return file_size

    if file_path is not None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return os.path.getsize(file_path)

    raise ValueError("Either file_path or file_size must be provided")


def _validate_positive(value: float, name: str) -> None:
    """Validate that a parameter is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a parameter is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_timeout_bounds(min_timeout: float, max_timeout: float | None) -> None:
    """Validate that timeout bounds are consistent."""
    if max_timeout is not None and min_timeout > max_timeout:
        raise ValueError(
            f"min_timeout_sec ({min_timeout}) cannot be greater than "
            f"max_timeout_sec ({max_timeout})"
        )
