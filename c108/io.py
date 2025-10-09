"""
File I/O utilities with progress tracking for large file operations.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import io
import os
from typing import Any, Callable, Union
from pathlib import Path

# Constants ------------------------------------------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB


# Classes --------------------------------------------------------------------------------------------------------------

class StreamingFile(io.FileIO):
    """
    A file-like object that tracks read and write progress via callbacks.

    This class extends io.FileIO to add progress tracking for file operations.
    It's designed to work with cloud storage APIs (like AWS S3 or Google Cloud Storage)
    that perform large read/write operations on file-like objects.

    The class handles large operations by breaking them into smaller chunks
    to provide frequent progress updates while maintaining full data integrity.

    Args:
        path: Path to the file to open.
        mode: File mode ('r', 'rb', 'w', 'wb', etc.). Defaults to 'r'.
        callback: Function called after each chunk is transferred.
            Signature: callback(current_bytes: int, total_bytes: int) -> None
        chunk_size: Size in bytes for each chunk. Defaults to 8MB.
            Set to 0 to use file_size (single chunk, minimal progress updates).
            This value often aligns with cloud provider defaults (e.g., AWS S3
            multipart uploads default to 8MB). Google Cloud Storage defaults
            to a larger 100MB chunk size for uploads.
        expected_size: Expected total size in bytes for write operations.
            Required for accurate progress tracking in write mode.

    Attributes:
        bytes_read: Total bytes read from the file.
        bytes_written: Total bytes written to the file.
        chunk_size: Size of chunks for operations.

    Notes:
        The progress reported by the callback reflects the amount of data
        transferred to or from the underlying client library (e.g., `boto3`,
        `google-cloud-storage`), not the actual network transfer progress.
        Cloud provider libraries often have their own internal buffering,
        chunking, and retry mechanisms that are not visible to this class.
        Therefore, the progress updates indicate how much data the library has
        consumed from this file-like object, which is a close proxy but not
        a direct measure of the upload/download to the cloud service.

    Example:
        Reading (e.g., uploading to cloud storage):

        >>> def progress(current, total):
        ...     print(f"Progress: {current}/{total} bytes ({current/total*100:.1f}%)")
        ...
        >>> with StreamingFile('large_file.mp4', 'rb', callback=progress) as f:
        ...     blob.upload_from_file(f)

        Writing (e.g., downloading from cloud storage):

        >>> # Download a 100MB file
        >>> with StreamingFile('output.mp4', 'wb', callback=progress,
        ...                     expected_size=100*1024*1024) as f:
        ...     blob.download_to_file(f)
    """

    bytes_read: int
    bytes_written: int
    callback: Callable[[int, int], None]
    chunk_size: int
    _total_size: int
    _mode: str

    def __init__(
            self,
            path: Union[str, bytes, Path, int],
            mode: str = 'r',
            callback: Union[Callable[[int, int], None], None] = None,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            expected_size: Union[int, None] = None
    ) -> None:
        """
        Initialize a StreamingFile with progress tracking.

        Args:
            path: Path to the file to open.
            mode: File mode string (e.g., 'rb', 'wb').
            callback: Optional progress callback function.
            chunk_size: Size of chunks for read/write operations in bytes.
            expected_size: Expected total size for write mode (enables progress tracking).

        Raises:
            ValueError: If path is empty or invalid parameters provided.
        """
        if not path:
            raise ValueError("StreamingFile path required")

        super().__init__(path, mode)

        self._mode = mode
        self.callback = callback or self._callback_default

        # Determine total size for progress calculations
        if 'w' in mode or 'a' in mode:
            # Write/append mode: use expected_size or 0
            self._total_size = expected_size or 0
        else:
            # Read mode: get actual file size
            self._total_size = os.fstat(self.fileno()).st_size

        # Set chunk size (use 0 to disable chunking and use file_size)
        if chunk_size == 0:
            self.chunk_size = max(self._total_size, 1)
        else:
            self.chunk_size = max(chunk_size, 1)

        # Initialize progress counters
        self.bytes_read = 0
        self.bytes_written = 0

    @property
    def file_size(self) -> int:
        """
        Get the current size of the file in bytes.

        Returns:
            Current file size in bytes.

        Raises:
            ValueError: If file is closed.
        """
        if self.closed:
            raise ValueError(f"Cannot get file size, file is closed: {self.name}")
        return os.fstat(self.fileno()).st_size

    @property
    def total_chunks(self) -> int:
        """
        Get the total number of chunks used for file progress.

        Returns:
            Total number of streaming file chunks.
        """
        return _get_chunks_number(self.chunk_size, self.total_size)

    @property
    def total_size(self) -> int:
        """
        Get the total size used for progress tracking.

        For read mode, this is the actual file size.
        For write mode, this is the expected_size provided at initialization.

        Returns:
            Total size in bytes for progress calculations.
        """
        return self._total_size

    @property
    def is_closed(self) -> bool:
        """Check if the file is closed."""
        return self.closed

    @property
    def is_open(self) -> bool:
        """Check if the file is open."""
        return not self.closed

    @property
    def progress_percent(self) -> float:
        """
        Get the current progress as a percentage.

        Returns:
            Progress percentage (0.0 to 100.0).
        """
        if self._total_size == 0:
            return 0.0

        current = self.bytes_read if 'r' in self._mode else self.bytes_written
        return (current / self._total_size) * 100.0

    def _callback_default(self, current_bytes: int, total_bytes: int) -> None:
        """
        Default progress callback that prints to stdout.

        Override by providing your own callback function to __init__.

        Args:
            current_bytes: Number of bytes transferred so far.
            total_bytes: Total bytes to transfer.
        """
        mode_str = "Read" if 'r' in self._mode else "Write"
        percent = (current_bytes / total_bytes * 100) if total_bytes > 0 else 0
        print(f"{mode_str} Progress: {current_bytes}/{total_bytes} bytes ({percent:.1f}%)")

    def read(self, size: int = -1) -> bytes:
        """
        Read up to size bytes from the file with progress tracking.

        This method breaks large reads into chunks to provide frequent
        progress updates via the callback function.

        Args:
            size: Maximum number of bytes to read. -1 means read until EOF.

        Returns:
            Bytes read from the file.
        """
        # Optimize small reads: read directly without chunking
        if size > 0 and size <= self.chunk_size:
            data = super().read(size)
            self.bytes_read += len(data)

            if self.callback:
                self.callback(self.bytes_read, self._total_size)

            return data

        # For large reads or read-all (-1), use chunked reading
        buffer = bytearray()
        bytes_remaining = size  # Tracks remaining bytes for this specific read() call

        while True:
            # Determine chunk size for this iteration
            if size == -1:
                # Read all: use full chunk_size
                chunk_to_read = self.chunk_size
            else:
                # Bounded read: read up to remaining bytes
                if bytes_remaining <= 0:
                    break
                chunk_to_read = min(self.chunk_size, bytes_remaining)

            # Read the chunk
            chunk = super().read(chunk_to_read)

            # EOF or no data
            if not chunk:
                break

            buffer.extend(chunk)
            self.bytes_read += len(chunk)

            if size != -1:
                bytes_remaining -= len(chunk)

            # Report progress
            if self.callback:
                self.callback(self.bytes_read, self._total_size)

        return bytes(buffer)

    def write(self, data: bytes) -> int:
        """
        Write bytes to the file with progress tracking.

        For large writes, data is written in chunks to provide frequent
        progress updates via the callback function.

        Args:
            data: Bytes to write to the file.

        Returns:
            Total number of bytes written.
        """
        total_bytes_to_write = len(data)
        bytes_written_this_call = 0

        # Optimize small writes: write directly without chunking
        if total_bytes_to_write <= self.chunk_size:
            result = super().write(data)
            self.bytes_written += result

            if self.callback:
                self.callback(self.bytes_written, self._total_size)

            return result

        # For large writes, write in chunks
        for i in range(0, total_bytes_to_write, self.chunk_size):
            chunk = data[i: i + self.chunk_size]
            result = super().write(chunk)
            bytes_written_this_call += result
            self.bytes_written += result

            # Report progress
            if self.callback:
                self.callback(self.bytes_written, self._total_size)

        return bytes_written_this_call

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Change the file position and update progress counters.

        When seeking, the progress counters are updated to reflect the new
        file position to maintain accurate progress tracking.

        Args:
            offset: Offset in bytes.
            whence: Reference point: 0=start, 1=current position, 2=end.

        Returns:
            New absolute file position in bytes.
        """
        new_position = super().seek(offset, whence)

        # Update progress counter to match new position
        if 'r' in self._mode:
            self.bytes_read = new_position
        else:
            # For write/append mode, also update position
            # This handles cases where seeking back and overwriting
            self.bytes_written = new_position

        return new_position


# Helper Functions -----------------------------------------------------------------------------------------------------

def _get_chunks_number(chunk_size: int, file_size: int) -> int:
    """
    Calculate the number of chunks needed for a given file size.

    Args:
        chunk_size: Size of each chunk in bytes.
        file_size: Total file size in bytes.

    Returns:
        Number of chunks required (using ceiling division).

    Raises:
        ValueError: If chunk_size or file_size is negative, or if chunk_size
            is 0 for a non-empty file.
    """
    if chunk_size < 0 or file_size < 0:
        raise ValueError("chunk_size and file_size must be >= 0")

    if chunk_size == 0:
        if file_size == 0:
            return 0
        else:
            raise ValueError("chunk_size cannot be 0 if file_size is greater than 0")

    # Ceiling division: (a + b - 1) // b
    return (file_size + chunk_size - 1) // chunk_size


def _get_chunk_size(chunk_size: int, chunks: int, file_size: int) -> int:
    """
    Calculate appropriate chunk size based on parameters.

    Priority order: chunk_size > chunks > default (file_size).

    Args:
        chunk_size: Explicit chunk size in bytes (0 means not specified).
        chunks: Desired number of chunks (0 means not specified).
        file_size: Total file size in bytes.

    Returns:
        Calculated chunk size in bytes (minimum 1).

    Raises:
        ValueError: If any parameter is negative.
    """
    if chunk_size < 0 or chunks < 0 or file_size < 0:
        raise ValueError("chunk_size, chunks, and file_size must be >= 0")

    if chunk_size:
        return chunk_size
    elif chunks:
        # Ceiling division to ensure even distribution
        return max((file_size + chunks - 1) // chunks, 1)
    else:
        # Default: use file_size (single chunk)
        return max(file_size, 1)
