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
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Union

from c108.dataclasses import mergeable
from typing_extensions import Self
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from .abc import classgetter
from .formatters import fmt_any

# Local ----------------------------------------------------------------------------------------------------------------
from .sentinels import ifnotnone

# Constants ------------------------------------------------------------------------------------------------------------

# Defaults chosen based on common network conditions and API requirements

BASE_TIMEOUT_SEC = 5.0  # DNS resolution + connection establishment (3-5s typical)
MAX_TIMEOUT_SEC = 3600.0  # 1 hour - common API gateway timeout limit
MIN_TIMEOUT_SEC = 10.0  # Minimum practical timeout for any network operation
OVERHEAD_PERCENT = 15.0  # TCP/IP overhead: headers, acknowledgments, retransmissions
PROTOCOL_OVERHEAD_SEC = 2.0  # HTTP headers, chunked encoding overhead
SAFETY_MULTIPLIER = 2.0  # 2x buffer for network variability and congestion
SPEED_MBPS = 100.0  # Typical broadband connection (~12.5 MB/s)


# Enums ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferOptions:
    """
    Configuration options for data transfer timing and performance factors.

    Attributes:
        base_timeout (float): Time limit for DNS resolution and connection
            establishment in seconds.
        max_retries: Maximum number of retry attempts. Default is 3 retries
            (4 total attempts) - standard for handling transient failures.
        max_timeout (float): Maximum allowed timeout duration in seconds.
        min_timeout (float): Minimum allowable timeout duration in seconds
            to ensure practical network operations .
        overhead_percent (float): Percentage of additional time for TCP/IP
            overhead such as headers and retransmissions.
        protocol_overhead (float): Time overhead in seconds introduced by
            HTTP headers and chunked encoding.
        retry_delay: Base retry delay in seconds.
        retry_multiplier: Multiplier for exponential backoff. Default 2.0 multiplier,
            results in delays of: 1s, 2s, 4s, 8s, etc.
        safety_multiplier (float): Multiplier to account for network variability
            and congestion buffer.
        speed (float): Network speed in megabits per second.
    """

    base_timeout: int | float = 5.0
    max_retries: int = 0
    max_timeout: int | float = 3600.0
    min_timeout: int | float = 10.0
    overhead_percent: int | float = 15.0
    protocol_overhead: int | float = 2.0
    retry_delay: int | float = 1.0
    retry_multiplier: int | float = 2.0
    safety_multiplier: int | float = 2.0
    speed: int | float = 100.0

    @classmethod
    def api_upload(cls) -> Self:
        """API upload transfers - datacenter to cloud service.

        Optimized for uploading to REST APIs, ML model registries (HuggingFace,
        MLflow), cloud storage APIs. Assumes asymmetric connection where upload
        is typically 1/5 to 1/10 of download speed.

        Use for: Model uploads, dataset publishing, CI/CD artifact uploads.
        """
        return TransferOptions(
            base_timeout=10.0,
            max_timeout=3600.0,
            min_timeout=15.0,
            overhead_percent=20.0,
            protocol_overhead=5.0,  # API processing time
            safety_multiplier=2.5,
            speed=50.0,  # Realistic asymmetric upload (not 100)
            max_retries=2,
            retry_delay=2.0,
        )

    @classmethod
    def cdn_download(cls) -> Self:
        """CDN download transfers - highly optimized delivery.

        For downloading from CDNs (CloudFront, Cloudflare, Fastly). Use when
        downloading Python packages, pre-trained models, datasets from
        public repositories.

        CDNs are geographically distributed and highly optimized.
        """
        return TransferOptions(
            base_timeout=3.0,  # CDNs have excellent DNS/connection
            max_timeout=3600.0,
            min_timeout=5.0,
            overhead_percent=10.0,  # CDNs minimize overhead
            protocol_overhead=1.0,  # Minimal processing
            safety_multiplier=1.5,  # Very reliable
            speed=300.0,  # Good CDN performance
            max_retries=1,
            retry_delay=1.0,
        )

    @classmethod
    def cloud_storage(cls) -> Self:
        """Cloud storage transfers - AWS S3, GCP Cloud Storage, Azure Blob.

        Regional cloud storage within same region/zone. Use for ML training
        data loading, checkpoint storage, artifact storage.

        Assumes same-region transfer for best performance.
        """
        return TransferOptions(
            base_timeout=5.0,
            max_timeout=3600.0,
            min_timeout=10.0,
            overhead_percent=15.0,
            protocol_overhead=3.0,  # Multipart upload overhead
            safety_multiplier=2.0,  # Cloud providers are reliable
            speed=200.0,  # Same-region cloud speeds
            max_retries=2,  # Cloud APIs handle some retries
            retry_delay=1.5,
        )

    @classmethod
    def fiber_symmetric(cls, speed: float = 800.0) -> Self:
        """Symmetric fiber with configurable bandwidth tier.

        Args:
            speed: Provisioned fiber speed in Mbps (accounts for ~20% overhead).
                Common tiers:
                - 80: 100 Mbps fiber
                - 400: 500 Mbps fiber
                - 800: 1 Gbps fiber (default)
                - 4000: 5 Gbps fiber
                - 9000: 10 Gbps fiber

        Example:
            >>> # Standard gigabit fiber
            >>> opts = TransferOptions.fiber_symmetric()

            >>> # Enterprise 10 Gbps
            >>> opts = TransferOptions.fiber_symmetric(speed=9000.0)
        """
        # Better infrastructure at higher tiers
        overhead = 12.0 - (min(speed, 10000) / 2000)  # 12% at 100, 7% at 10000
        safety = 1.5 - (min(speed, 10000) / 50000)  # 1.5 at 100, 1.3 at 10000

        return TransferOptions(
            base_timeout=2.0,
            max_timeout=7200.0,
            min_timeout=5.0,
            overhead_percent=max(7.0, overhead),
            protocol_overhead=1.0,
            safety_multiplier=max(1.2, safety),
            speed=speed,
        )

    @classmethod
    def ipfs_gateway(cls) -> Self:
        """IPFS gateway transfers - content-addressed distributed storage.

        For accessing datasets and models via IPFS gateways. IPFS is increasingly
        used in ML reproducibility, decentralized datasets (e.g., HuggingFace
        datasets on IPFS), and blockchain ML applications.

        Gateway performance varies significantly by provider and content popularity.
        This preset assumes public gateways (pinata.cloud, ipfs.io, etc).

        Use for: Decentralized datasets, immutable model versioning, Web3 ML.
        """
        return TransferOptions(
            base_timeout=20.0,  # DHT lookup can be slow
            max_timeout=7200.0,  # Large files on slow gateways
            min_timeout=30.0,  # Content discovery takes time
            overhead_percent=40.0,  # High DHT/routing overhead
            protocol_overhead=8.0,  # Gateway processing, content routing
            safety_multiplier=3.5,  # Highly variable gateway performance
            speed=30.0,  # Conservative public gateway speed
            max_retries=3,  # Gateways often timeout, benefit from retries
            retry_delay=5.0,
        )

    @classmethod
    def lan_sync(cls, speed: float = 600.0) -> Self:
        """Local network sync - Resilio Sync, Syncthing, LAN transfers.

        Args:
            speed: LAN speed in Mbps based on your network hardware.
                Common values:
                - 100: Fast Ethernet (100BASE-T)
                - 600: Gigabit Ethernet with realistic overhead (default)
                - 2500: 2.5 GbE
                - 5000: 5 GbE
                - 9000: 10 GbE with overhead

        Example:
            >>> # Default gigabit LAN
            >>> opts = TransferOptions.lan_sync()

            >>> # 10 GbE datacenter LAN
            >>> opts = TransferOptions.lan_sync(speed=9000.0)

            >>> # Legacy 100 Mbps network
            >>> opts = TransferOptions.lan_sync(speed=100.0)
        """
        # Scale safety inversely with speed (faster = more predictable)
        safety = 1.8 - (min(speed, 10000) / 20000)  # 1.8 at 100, 1.3 at 10000

        return TransferOptions(
            base_timeout=2.0,
            min_timeout=5.0,
            overhead_percent=12.0,
            protocol_overhead=1.5,
            safety_multiplier=max(1.3, safety),
            speed=speed,
            max_retries=1,
        )

    @classmethod
    def mobile_4g(cls) -> Self:
        """4G/LTE mobile networks - still dominant globally through 2030s.

        Conservative settings for typical 4G performance across urban and
        suburban areas worldwide. Suitable for mobile app development,
        field data collection, and IoT applications.

        Typical speeds: 15-50 Mbps in real-world conditions; use .merge(speed=X)
        if you measured your connection.
        """
        return TransferOptions(
            base_timeout=8.0,  # Mobile DNS can be slower
            max_timeout=3600.0,
            min_timeout=15.0,
            overhead_percent=25.0,  # Mobile networks have NAT/carrier overhead
            protocol_overhead=3.0,
            safety_multiplier=3.0,  # High variance: signal strength, congestion
            speed=30.0,  # Conservative 4G estimate
            max_retries=2,  # Mobile connections often need retries
            retry_delay=2.0,  # Longer delays for mobile
        )

    @classmethod
    def mobile_5g(cls) -> Self:
        """5G mobile networks - rapidly expanding in major cities globally.

        Optimized for modern 5G in urban areas across US, Europe, and major
        Asian cities (Seoul, Singapore, Tokyo, etc.). Suitable for ML model
        downloads, real-time data streaming, and high-bandwidth mobile apps.

        Typical speeds: 100-350 Mbps in real-world conditions; use .merge(speed=X)
        if you measured your connection.
        """
        return TransferOptions(
            base_timeout=6.0,  # 5G connects faster
            max_timeout=3600.0,
            min_timeout=12.0,
            overhead_percent=18.0,  # Better protocol efficiency than 4G
            protocol_overhead=2.5,
            safety_multiplier=2.5,  # More reliable than 4G but still mobile
            speed=150.0,  # Mid-range 5G (conservative)
            max_retries=1,  # 5G more reliable
            retry_delay=1.5,
        )

    @classmethod
    def peer_transfer(cls) -> Self:
        """Peer-to-peer direct transfers - assumes slower peer bottleneck.

        Direct transfers between development machines, lab computers, or
        distributed training nodes. Assumes residential/office networks where
        one peer is the bottleneck.

        Use for: Git LFS, local model sharing, distributed training setup.
        """
        return TransferOptions(
            base_timeout=8.0,  # NAT traversal takes time
            max_timeout=3600.0,
            min_timeout=15.0,
            overhead_percent=25.0,  # NAT, firewall, relay overhead
            protocol_overhead=2.0,
            safety_multiplier=3.0,  # Highly variable peer quality
            speed=50.0,  # Assume bottleneck scenario
            max_retries=2,
            retry_delay=2.0,
        )

    @classmethod
    def satellite_geo(cls) -> Self:
        """Legacy GEO satellite internet (HughesNet, Viasat traditional service).

        Geostationary satellites at 35,786 km altitude. High latency makes
        these less suitable for modern development but still common in remote
        areas. Use only when LEO satellites unavailable.

        Typical speeds: 12-100 Mbps, Latency: 600-700ms.
        """
        return TransferOptions(
            base_timeout=40.0,  # Very high latency for initial handshake
            max_timeout=3600.0,
            min_timeout=60.0,  # Even small transfers take time
            overhead_percent=40.0,  # High latency causes TCP retransmissions
            protocol_overhead=8.0,  # Protocol overhead amplified by latency
            safety_multiplier=4.0,  # Weather severely impacts GEO
            speed=25.0,  # Conservative estimate
            max_retries=3,
            retry_delay=5.0,  # Long retries due to latency
        )

    @classmethod
    def satellite_leo(cls) -> Self:
        """Modern LEO satellite internet (Starlink, OneWeb, etc.).

        Low Earth Orbit satellites provide near-broadband speeds with reasonable
        latency. Suitable for remote work, ML training in field deployments,
        and connecting edge computing in remote locations.

        Typical speeds: 50-200 Mbps, Latency: 20-50ms.
        """
        return TransferOptions(
            base_timeout=15.0,  # Initial satellite acquisition
            max_timeout=3600.0,
            min_timeout=25.0,
            overhead_percent=25.0,  # Satellite protocol overhead
            protocol_overhead=4.0,  # Additional encoding/error correction
            safety_multiplier=3.0,  # Weather, satellite handoff variance
            speed=100.0,  # Conservative Starlink estimate
            max_retries=2,
            retry_delay=3.0,  # Longer retry for satellite handoff
        )

    @classmethod
    def torrent_swarm(cls) -> Self:
        """BitTorrent/P2P swarm transfers - multiple peer sources.

        Optimized for torrent-based downloads where multiple peers contribute
        chunks simultaneously. Common for large dataset distribution, Linux ISO
        downloads, and decentralized model weight sharing.

        Speed scales with swarm health (number of seeders). This preset assumes
        a healthy swarm (10+ seeders); use .merge(speed=X) based on observed swarm health.

        Use for: Academic datasets, distro images, public model mirrors.
        """
        return TransferOptions(
            base_timeout=15.0,  # DHT/tracker discovery takes time
            max_timeout=7200.0,  # Very large files are common
            min_timeout=30.0,  # Swarm coordination overhead
            overhead_percent=35.0,  # Protocol overhead, chunk verification
            protocol_overhead=5.0,  # DHT lookup, peer handshakes
            safety_multiplier=2.0,  # Swarms are resilient but variable
            speed=80.0,  # Aggregate swarm speed (conservative)
            max_retries=3,  # P2P benefits from retries
            retry_delay=5.0,  # Wait for new peers to discover
        )

    def merge(
        self,
        *,
        base_timeout: int | float = None,
        max_retries: int = None,
        max_timeout: int | float = None,
        min_timeout: int | float = None,
        overhead_percent: int | float = None,
        protocol_overhead: int | float = None,
        retry_delay: int | float = None,
        retry_multiplier: int | float = None,
        safety_multiplier: int | float = None,
        speed: int | float = None,
    ) -> Self:
        """
        Create a new TransferOptions instance with selectively updated fields.

        If a parameter is None, no update is applied to the corresponding field.

        Private fields (starting with '_') are excluded

        Args:
            base_timeout: Base Timeout
            max_retries: Max Retries
            max_timeout: Max Timeout
            min_timeout: Min Timeout
            overhead_percent: Overhead Percent
            protocol_overhead: Protocol Overhead
            retry_delay: Retry Delay
            retry_multiplier: Retry Multiplier
            safety_multiplier: Safety Multiplier
            speed: Speed

        Returns:
            New TransferOptions instance with merged configuration
        """
        base_timeout = ifnotnone(base_timeout, default=self.base_timeout)
        max_retries = ifnotnone(max_retries, default=self.max_retries)
        max_timeout = ifnotnone(max_timeout, default=self.max_timeout)
        min_timeout = ifnotnone(min_timeout, default=self.min_timeout)
        overhead_percent = ifnotnone(overhead_percent, default=self.overhead_percent)
        protocol_overhead = ifnotnone(protocol_overhead, default=self.protocol_overhead)
        retry_delay = ifnotnone(retry_delay, default=self.retry_delay)
        retry_multiplier = ifnotnone(retry_multiplier, default=self.retry_multiplier)
        safety_multiplier = ifnotnone(safety_multiplier, default=self.safety_multiplier)
        speed = ifnotnone(speed, default=self.speed)

        return TransferOptions(
            base_timeout=base_timeout,
            max_retries=max_retries,
            max_timeout=max_timeout,
            min_timeout=min_timeout,
            overhead_percent=overhead_percent,
            protocol_overhead=protocol_overhead,
            retry_delay=retry_delay,
            retry_multiplier=retry_multiplier,
            safety_multiplier=safety_multiplier,
            speed=speed,
        )

    def __post_init__(self):
        _validate_non_negative(self.base_timeout, "base_timeout")
        _validate_non_negative(self.max_retries, "max_retries")
        _validate_non_negative(self.max_timeout, "max_timeout")
        _validate_non_negative(self.min_timeout, "min_timeout")
        _validate_timeout_bounds(self.min_timeout, self.max_timeout)
        _validate_non_negative(self.overhead_percent, "overhead_percent")
        _validate_non_negative(self.protocol_overhead, "protocol_overhead")
        _validate_non_negative(self.retry_delay, "retry_delay")
        _validate_non_negative(self.retry_multiplier, "retry_multiplier")
        _validate_positive(self.safety_multiplier, "safety_multiplier")
        _validate_positive(self.speed, "speed")

        if self.overhead_percent > 200.0:
            raise ValueError(
                f"overhead_percent seems unreasonably high: {self.overhead_percent}%. "
                f"Typical values are 10-40%."
            )


class TransferType(str, Enum):
    """
    Predefined network transfer types.
    """

    API_UPLOAD = "api_upload"  # Uploading to REST API with processing overhead
    CDN_DOWNLOAD = "cdn_download"  # Downloading from CDN (typically faster, more reliable)
    CLOUD_STORAGE = "cloud_storage"  # S3, GCS, Azure Blob storage
    PEER_TRANSFER = "peer_transfer"  # Direct peer-to-peer transfer
    MOBILE_NETWORK = "mobile_network"  # Mobile/cellular connection (variable quality)
    SATELLITE = "satellite"  # High latency satellite connection


# Main API functions ---------------------------------------------------------------------------------------------------
def batch_timeout(
    files: list[Union[str, os.PathLike[str], int, tuple[str | os.PathLike[str], int]]],
    parallel: bool = False,
    max_parallel: int = 4,
    speed: float = 100,
    speed_unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"] = "mbps",
    **kwargs,
) -> int:
    """
     Estimate timeout for transferring multiple files.

     For sequential transfers, timeouts are summed.

     For parallel transfers, bandwidth is shared equally among concurrent transfers
    (up to max_parallel), and the total timeout is determined by the longest transfer.

     Args:
         files: List of files to transfer. Each element can be:
             - str/PathLike: file path
             - int: file size in bytes
             - tuple: (file_path, file_size) for pre-computed sizes
         parallel: If True, assumes parallel transfer. If False, sequential.
         max_parallel: Maximum number of parallel transfers. Only used if parallel=True.
         speed: Expected transfer speed (divided among parallel transfers).
         speed_unit: Unit of speed: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec).
         **kwargs: Additional parameters passed to transfer_timeout() (e.g.,
             base_timeout, protocol_overhead, safety_multiplier).

     Returns:
         Total estimated timeout in seconds as an integer.

     Raises:
         ValueError: If the file list is empty or contains invalid elements.

     Examples:
         >>> # Sequential and parallel upload of 3 equal-size files, # MB total
         >>> files = [2**20, 2**20, 2**20] # 1 MB each file
         >>> # OR files = ["file1.txt", "file2.txt", "file3.txt"]
         >>> batch_timeout(files, parallel=False)
         30
         >>> batch_timeout(files, parallel=True, max_parallel=3)
         10

         >>> # Sequential and parallel upload of 3 files of different sizes,
         >>> # smaller files consume bandwidth inefficiently
         >>> files = [1024, 2**20, 2**30] # bytes
         >>> batch_timeout(files, parallel=False)
         216
         >>> batch_timeout(files, parallel=True, max_parallel=3)
         573

         >>> # Parallel upload of 3 large files estimates to same timeout
         >>> files = [2**30, 2**30, 2**30] # bytes
         >>> batch_timeout(files, parallel=True, max_parallel=3)
         573
    """
    if not files:
        raise ValueError("files list cannot be empty")

    if parallel and max_parallel < 1:
        raise ValueError(f"max_parallel must be at least 1, got {max_parallel}")

    # Convert speed to Mbps
    speed_mbps = speed
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    if parallel:
        # Bandwidth is shared among concurrent transfers
        sharing_factor = min(len(files), max_parallel)
        speed_per_transfer = speed_mbps_actual / sharing_factor
    else:
        # Sequential: each transfer gets full bandwidth
        speed_per_transfer = speed_mbps_actual

    # Calculate individual timeouts with appropriate bandwidth
    timeouts = []
    for item in files:
        if isinstance(item, int):
            file_size = item
            file_path = None
        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"Tuple must be (path, size), got {item}")
            file_path, file_size = item
        else:
            file_path = item
            file_size = None

        timeout = transfer_timeout(
            file_path=file_path,
            file_size=file_size,
            speed=speed_per_transfer,
            speed_unit="mbps",
            **kwargs,
        )
        timeouts.append(timeout)

    if parallel:
        # Total time is determined by the longest file
        total_timeout = max(timeouts)
    else:
        # Sequential: sum all individual transfer times
        total_timeout = sum(timeouts)

    return math.ceil(total_timeout)


def chunk_timeout(
    chunk_size: int,
    speed: float = 100,
    speed_unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"] = "mbps",
    **kwargs,
) -> int:
    """
    Estimate timeout for a single chunk in chunked/resumable transfer.

    Useful for multipart uploads, streaming, or resumable upload protocols where
    files are split into chunks.

    This method uses transfer_timeout() with file as one chunk preset.

    Args:
        chunk_size: Size of the chunk in bytes. Common sizes: 5MB (S3 minimum),
            8MB (typical), 16MB, 32MB, 64MB (large chunks).
        speed: Expected transfer speed.
        speed_unit: Unit of speed: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec).
        **kwargs: Additional parameters passed to transfer_timeout().

    Returns:
        Timeout for this chunk in seconds as an integer.

    Examples:
        >>> # Timeout for 8MB chunk (typical chunk size)
        >>> chunk_timeout(8*1024*1024, speed=100.0)
        10

        >>> # S3 multipart upload minimum chunk
        >>> chunk_timeout(5*1024*1024, speed=12.0)
        15

        >>> # Large chunk for fast connection
        >>> chunk_timeout(120*1024*1024, speed=500.0)
        12
    """
    _validate_positive(chunk_size, "chunk_size")

    return transfer_timeout(file_size=chunk_size, speed=speed, speed_unit=speed_unit, **kwargs)


def transfer_time(
    file_path: str | os.PathLike[str] | None = None,
    file_size: int | None = None,
    speed: float = 100,
    speed_unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"] = "mbps",
    overhead_percent: float = OVERHEAD_PERCENT,
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
        speed: Expected transfer speed in the specified unit.
            Default is 100.0 Mbps (~12.5 MB/s).
        speed_unit: Unit of speed: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec).
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
        >>> transfer_time(file_size=500*1024*1024, speed=100.0)
        46.0

        >>> # Get estimate in minutes for large file
        >>> transfer_time(
        ...     file_size=5*1024**3,  # 5 GB
        ...     speed=1024,
        ...     unit="seconds"
        ... )
        46.0

        >>> # Using MB/s instead of Mbps
        >>> transfer_time(
        ...     file_size=5*1024**3,  # 5 GB
        ...     speed=1024,
        ...     speed_unit="MBps",
        ...     unit="seconds"
        ... )
        5.75

        >>> # Estimate in hours for very large transfer
        >>> transfer_time(
        ...     file_size=100*1024**3,  # 100 GB
        ...     speed=100.0,
        ...     unit="hours"
        ... )
        2.6168888888888886

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
    _validate_positive(speed, "speed")
    _validate_non_negative(overhead_percent, "overhead_percent")

    if unit not in ("seconds", "minutes", "hours"):
        raise ValueError(f"unit must be 'seconds', 'minutes', or 'hours', got '{unit}'")

    # Convert speed to Mbps if needed
    speed_mbps = speed
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
    speed: float = 100,
    speed_unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"] = "mbps",
    **kwargs,
) -> dict:
    """
    Get comprehensive transfer estimates with multiple metrics.

    Returns a dictionary with timeout, transfer time, and formatted human-readable strings.
    Useful for displaying transfer information to users.

    Args:
        file_path: Path to the file to be transferred.
        file_size: Size of the file in bytes.
        speed: Expected transfer speed.
        speed_unit: Unit of speed: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec).
        **kwargs: Additional parameters for transfer_timeout().

    Returns:
        Dictionary containing:
            - file_size: Human-readable file size
            - file_size_bytes: File size in bytes
            - time: Human-readable duration string
            - time_sec: Expected duration in seconds (float)
            - timeout_sec: Timeout in seconds (int)
            - timeout: Human-readable timeout string
            - speed: Speed in Mbps

    Examples:
        >>> # Get comprehensive estimate
        >>> est = transfer_estimates(
        ...     file_size=100*1024*1024,
        ...     speed=50.0
        ... )
        >>> {k: v for k, v in est.items() if k in ["file_size","time", "timeout"]}
        {'file_size': '100.0 MB', 'time': '18.4 seconds', 'timeout': '44.0 seconds'}
    """
    # Get file size
    size_bytes = _get_file_size(file_path, file_size)

    # Convert speed to Mbps
    speed_mbps = speed
    speed_mbps_actual = _speed_to_mbps(speed_mbps, speed_unit)

    # Calculate timeout and duration
    timeout = transfer_timeout(
        file_size=size_bytes,
        speed=speed_mbps_actual,
        speed_unit="mbps",
        **kwargs,
    )

    transfer_time_ = transfer_time(
        file_size=size_bytes,
        speed=speed_mbps_actual,
        speed_unit="mbps",
        unit="seconds",
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
        "speed": speed_mbps_actual,
        "time": format_time(transfer_time_),
        "time_sec": transfer_time_,
        "timeout": format_time(timeout),
        "timeout_sec": timeout,
    }


def transfer_speed(
    url: str,
    sample_size_kb: int = 100,
    timeout_sec: float = 10,
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
        >>> speed = transfer_speed("https://cdn.example.com/test.dat")  # doctest: +SKIP

        >>> # Use measured speed for timeout estimation
        >>> speed = transfer_speed("https://api.example.com/health")    # doctest: +SKIP
        >>> timeout = transfer_timeout(                                 # doctest: +SKIP
        ...     file_size=10*1024*1024,
        ...     speed_mbps=speed
        ... )

        >>> # More accurate measurement with multiple samples
        >>> speed = transfer_speed(                                     # doctest: +SKIP
        ...     "https://cdn.example.com/test.dat",
        ...     sample_size_kb=500,
        ...     num_samples=3
        ... )

        >>> # Quick test with small sample
        >>> speed = transfer_speed(                                     # doctest: +SKIP
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
    speed: int | float = 100,
    speed_unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"] = "mbps",
    max_retries: int = None,
    retry_delay: int | float = None,
    opts: TransferOptions = None,
) -> int:
    """
    Estimate a safe timeout value for transferring a file over a network.

    Calculates transfer time based on file size and network conditions, accounting
    for protocol overhead, connection latency, and network variability. The timeout
    is calculated as:

        timeout = base_timeout + protocol_overhead +
                  (transfer_time * (1 + overhead%) * safety_multiplier)

    The result is then clamped to [min_timeout, max_timeout] and rounded
    up to the nearest integer using math.ceil() to ensure sufficient time.

    Args:
        file_path: Path to the file to be transferred. Either this or file_size
            must be provided.
        file_size: Size of the file in bytes. Either this or file_path must
            be provided. If both are given, file_size takes precedence.
        speed: Expected transfer speed in the specified unit.
            Default is 100.0 Mbps (~12.5 MB/s) - typical broadband connection.
            Common values: 10-50 (slow), 100-300 (typical), 500+ (fast).
        speed_unit: Unit of speed: "mbps" (megabits/sec),
            "MBps" (megabytes/sec), "kbps" (kilobits/sec), "KBps" (kilobytes/sec),
            "gbps" (gigabits/sec).
        base_timeout: Base timeout added to all transfers regardless of size.
            Default is 5.0 seconds - accounts for DNS resolution (~1s), TCP
            handshake (~1s), TLS handshake (~1-2s), and HTTP request/response (~1s).
        overhead_percent: Additional time as percentage of transfer time to account
            for network protocol overhead. Default is 15.0% - represents TCP/IP
            headers (~5%), acknowledgments (~3%), potential retransmissions (~5%),
            and HTTP chunking (~2%).
        safety_multiplier: Multiplier applied to the calculated transfer time to
            provide a safety margin. Default is 2.0x - provides buffer for network
            congestion, routing changes, server load, and other variability.
        protocol_overhead: Fixed overhead for protocol-specific operations.
            Default is 2.0 seconds - for multipart boundaries, chunked encoding,
            and initial API processing. Use 5-10s for heavy API processing, 1-2s
            for simple file transfers.
        min_timeout: Absolute minimum timeout value to return. Default is 10.0
            seconds - minimum practical timeout for any network operation considering
            connection establishment and basic handshakes.
        max_timeout: Maximum timeout value to return. Default is 3600.0 seconds
            (1 hour) - matches common API gateway limits (AWS ALB, CloudFlare, etc.).
            Set to None for no maximum.
        max_retries: Maximum number of retry attempts. Default is 3 retries
            (4 total attempts) - standard for handling transient failures.
        retry_delay: Initial backoff delay in seconds. Default is 1.0 second.
        retry_multiplier: Multiplier for exponential backoff. Default is 2.0,
            giving delays of: 1s, 2s, 4s, 8s, etc.
        opts

    Returns:
        Estimated timeout in seconds as an integer (rounded up using math.ceil).
        The timeout is always clamped between min_timeout and max_timeout.

    Raises:
        ValueError: If neither file_path nor file_size is provided, if speed is
            not positive, if file_size is negative, or if min_timeout exceeds
            max_timeout.
        FileNotFoundError: If file_path is provided but the file does not exist.
        OSError: If the file size cannot be determined due to permissions or I/O error.

    Examples:
        >>> # Small file on typical connection - returns minimum timeout
        >>> transfer_timeout(file_size=1024)  # 1 KB
        10

        >>> # 100MB file on slow connection
        >>> transfer_timeout(file_size=100*1024*1024, speed=10.0)
        191

        >>> # Using megabytes per second instead of megabits
        >>> transfer_timeout(
        ...     file_size=500*1024*1024,  # 500 MB
        ...     speed=10.0,  # 10 MB/s
        ...     speed_unit="MBps"
        ... )
        122

        >>> # Large file with custom safety margin
        >>> transfer_timeout(
        ...     file_size=5*1024**3,  # 5 GB
        ...     speed=500.0,
        ...     opts = TransferOptions(
        ...                   safety_multiplier=1.5,  # Less conservative
        ...                   max_timeout=7200  # 2 hour max
        ...                   )
        ... )
        149

        >>> # Using file path
        >>> transfer_timeout("backup.tar.gz", speed=50.0)       # doctest: +SKIP

        >>> # Conservative estimate for unreliable network
        >>> transfer_timeout(
        ...     file_size=1024*1024*1024,  # 1 GB
        ...     speed=50.0,
        ...     opts=TransferOptions(overhead_percent=25.0,
        ...                     safety_multiplier=2.5)
        ...     )
        519

    Note:
        This provides an estimate based on idealized conditions. Actual transfer
        times vary based on network congestion, server load, connection stability,
        routing, and many other factors. Always test with real-world conditions
        and adjust parameters accordingly.

        For production use, consider:
        - API uploads: Use higher protocol_overhead (5-10s) and safety_multiplier (2.5-3.0)
        - Direct transfers: Lower protocol_overhead (1-2s) and safety_multiplier (1.5-2.0)
        - Mobile networks: Much higher safety_multiplier (3-4x) and overhead_percent (25-40%)
        - Batch uploads: Use batch_timeout() for better accuracy
    """

    # Convert speed to Mbps if needed
    speed_mbps = _speed_to_mbps(speed, speed_unit)
    opts = opts or TransferOptions()
    opts = opts.merge(speed=speed_mbps, max_retries=max_retries, retry_delay=retry_delay)

    # Get file size
    file_size_bytes = _get_file_size(file_path, file_size)

    # Convert file size to megabits
    size_mbits = (file_size_bytes * 8) / (1024 * 1024)

    # Calculate base transfer time in seconds
    transfer_time_sec = size_mbits / speed_mbps

    # Apply overhead percentage
    transfer_with_overhead = transfer_time_sec * (1.0 + opts.overhead_percent / 100.0)

    # Apply safety multiplier
    safe_transfer_time = transfer_with_overhead * opts.safety_multiplier

    # Calculate total timeout
    total_timeout = opts.base_timeout + opts.protocol_overhead + safe_transfer_time

    # Clamp to min/max bounds
    timeout = max(opts.min_timeout, total_timeout)
    if opts.max_timeout is not None:
        timeout = min(timeout, opts.max_timeout)

    # Round up to nearest integer to ensure sufficient time
    if opts.max_retries == 0:
        return math.ceil(timeout)
    return _transfer_timeout_retry(
        file_size=file_size_bytes,
        opts=opts,
    )


def _transfer_timeout_retry(
    file_size: int | None = None,
    opts: TransferOptions = None,
) -> int:
    """
    Estimate timeout accounting for retry attempts with exponential backoff.

    Total timeout = (opts.base_timeout * (opts.max_retries + 1)) + sum(retry_delays)
    where retry_delays = [opts.retry_delay * opts.retry_multiplier^i for i in range(opts.max_retries)]

    This method invokes transfer_timeout() for the base timeout estimate.

    Args:
        file_size: Size of the file in bytes.
        opts: options passed to transfer_timeout().

    Returns:
        Total timeout including all retry attempts, as an integer.

    Raises:
        ValueError: If max_retries is negative or backoff parameters are invalid.
    """
    opts = opts or TransferOptions()
    max_retries = opts.max_retries or 0

    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative int, got {fmt_any(max_retries)}")

    # Get base timeout for a single attempt
    base_timeout = transfer_timeout(
        file_size=file_size,
        max_retries=0,
        opts=opts,
    )

    # Calculate total backoff/delay time: initial * (1 + multiplier + multiplier^2 + ... + multiplier^(n-1))
    # This is a geometric series: a * (r^n - 1) / (r - 1)
    if max_retries == 0:
        total_delay = 0.0
    elif opts.retry_multiplier == 1.0:
        total_delay = opts.retry_delay * max_retries
    else:
        total_delay = opts.retry_delay * (
            (opts.retry_multiplier**max_retries - 1) / (opts.retry_multiplier - 1)
        )

    # Total timeout: all attempts plus backoff delays
    total_timeout = opts.base_timeout * (max_retries + 1) + total_delay

    return math.ceil(total_timeout)


# Private helper methods -----------------------------------------------------------------------------------------------


def _speed_to_mbps(speed: float, unit: Literal["mbps", "MBps", "kbps", "KBps", "gbps"]) -> float:
    """
    Convert speed from various units to Mbps.

    Network speeds use decimal (SI) prefixes, not binary prefixes:
    - 1 kbps = 1,000 bps
    - 1 Mbps = 1,000 kbps = 1,000,000 bps
    - 1 Gbps = 1,000 Mbps = 1,000,000,000 bps

    Binary prefixes (1024) are used only for RAM and storage sizes (KiB, MiB, GiB),
    not for network transmission rates.
    """
    if unit not in ["mbps", "MBps", "kbps", "KBps", "gbps"]:
        raise ValueError(f"Invalid speed unit: {unit}")

    conversions = {
        "mbps": 1.0,
        "MBps": 8.0,  # 1 MB/s = 8 Mbps
        "kbps": 1.0 / 1000.0,  # 1 kbps = 1/1000 Mbps
        "KBps": 8.0 / 1000.0,  # 1 KB/s = 8/1000 Mbps
        "gbps": 1000.0,  # 1 Gbps = 1000 Mbps
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
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {fmt_any(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a parameter is non-negative."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {fmt_any(value)}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_timeout_bounds(min_timeout: float, max_timeout: float | None) -> None:
    """Validate that timeout bounds are consistent."""
    if max_timeout is not None and min_timeout > max_timeout:
        raise ValueError(
            f"min_timeout ({min_timeout}) cannot be greater than max_timeout ({max_timeout})"
        )
