#
# C108 IO Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import io, os
from typing import Any, Callable


# Classes --------------------------------------------------------------------------------------------------------------

class StreamingFile(io.FileIO):
    """
    A file-like object that tracks read and write progress and reports it via a callback.

    This class extends io.FileIO to add progress tracking for file operations.
    It's designed to work with Google Cloud Storage's upload_from_file and similar methods
    that perform large read operations on file-like objects.

    The class handles large read requests by breaking them into smaller chunks to provide
    more frequent progress updates, while still returning the full requested data.

    Example usage for GCP Storage Blob-s for binary r/w operations:

        def progress_callback(read_bytes: int = -1, write_bytes: int = -1, total_bytes: int = -1):
            print(f"Progress: {read_bytes}/{total_bytes} bytes")

        with StreamingFile('src_file.mp3', 'rb', callback=progress_callback) as f:
            blob.upload_from_file(f)

        with StreamingFile('dest_file.mp3', 'wb', callback=progress_callback) as f:
            blob.download_to_file(f_dest)
    """

    bytes_read: int
    bytes_written: int
    callback: Callable
    chunk_size: int
    chunks: int

    # file_size: int

    def __init__(self, path: Any, mode: str = 'r',
                 callback: Callable = None,
                 chunk_size: int = 0,
                 chunks: int = 0):
        """
        Initialize a new StreamingFile

        Args:
            path: Path to the file to open
            mode: File mode ('r', 'rb', 'w', 'wb', etc.)
            callback: Function to call when stream updates by a chunk
            chunk_size: Size of chunks for read/write operations (0 means = file size)
            chunks: Number of chunks for read/write; use either chunk_size or chunks, not both
        """

        if not path:
            raise ValueError("Streaming file Path required")

        super().__init__(path, mode)
        self.callback = callback or self.callback_default
        # self.file_size = os.fstat(self.fileno()).st_size
        self.chunk_size = _get_chunk_size(chunk_size=chunk_size, chunks=chunks, file_size=self.file_size)
        self.chunks = _get_chunks_number(chunk_size=self.chunk_size, file_size=self.file_size)
        self.bytes_read = 0
        self.bytes_written = 0

    @property
    def file_size(self):
        # Check if the file is closed before attempting to get its size
        if self.closed:
            raise ValueError(f"Cannot get file size, file is closed: {self.name}")
        return os.fstat(self.fileno()).st_size

    @property
    def is_closed(self):
        return self.closed

    @property
    def is_open(self):
        return not self.closed

    def callback_default(self, read_bytes: int = -1, write_bytes: int = -1, total_bytes: int = -1):
        """
        This is the default method is to be used for extra functionality called when stream updates with file chunk.

        In case of file RW modes this displays stats both on read and written bytes
        Implement your own and callback pass to StreamingFile init
        """
        print(f"File Stream read|write/total Bytes : {read_bytes}|{write_bytes}/{total_bytes}:")

    def read(self, size=-1):
        """
        Read up to size bytes from the file and track progress.

        This method handles large read requests by breaking them into smaller chunks
        to provide more frequent progress updates. This is especially important for
        cloud storage uploads where the client may request the entire file at once.

        Args:
            size: Maximum number of bytes to read. -1 means read until EOF.

        Returns:
            The bytes read from the file.
        """
        # If the requested size is a positive value and less than or equal to the chunk_size,
        # read it directly without internal chunking for efficiency.
        # Note: size=0 will also fall into the chunking loop, which correctly returns empty bytes.
        if size > 0 and size <= self.chunk_size:
            data = super().read(size)
            self.bytes_read += len(data)

            # Call the callback with current progress (always call for immediate feedback)
            if self.callback:
                self.callback(read_bytes=self.bytes_read, total_bytes=self.file_size)

            return data

        # For all other cases (size=-1, size=0, or size > chunk_size),
        # proceed with chunked reading to ensure progress updates.
        buffer = bytearray()
        bytes_to_read_this_call = size  # This variable tracks how many more bytes are requested by *this* read call
        # It only decrements if `size` is not -1

        while True:
            # Determine the size of the chunk to read in this iteration.
            # It's the minimum of:
            # 1. self.chunk_size (to ensure granular updates)
            # 2. bytes_to_read_this_call (if a specific `size` was requested and not yet fully read)

            read_current_iteration_size = self.chunk_size
            if size != -1:  # If a specific size was requested (not -1)
                if bytes_to_read_this_call <= 0:  # We've read enough for the requested `size`
                    break
                read_current_iteration_size = min(self.chunk_size, bytes_to_read_this_call)

            # Read from the underlying file
            chunk = super().read(read_current_iteration_size)

            # If no data was read, we've reached EOF or there's nothing more to read
            if not chunk:
                break

            buffer.extend(chunk)
            self.bytes_read += len(chunk)

            if size != -1:  # Only decrement `bytes_to_read_this_call` for bounded reads
                bytes_to_read_this_call -= len(chunk)

            # Call the callback with current overall progress
            if self.callback:
                self.callback(read_bytes=self.bytes_read, total_bytes=self.file_size)

        return bytes(buffer)

    def write(self, data):
        """
        Write the given bytes to the file and track progress.

        This method tracks the number of bytes written and calls the
        callback function to report progress. If the provided data is
        larger than chunk_size, it will be written in chunks to provide
        more frequent progress updates.

        Args:
            data: The bytes to write to the file

        Returns:
            The number of bytes written (total for this call)
        """
        total_bytes_to_write = len(data)
        bytes_written_this_call = 0

        # If the data is smaller than or equal to chunk_size, or if chunk_size is very small (0 or less)
        # write it directly without internal chunking.
        if total_bytes_to_write <= self.chunk_size or self.chunk_size <= 0:
            result = super().write(data)
            self.bytes_written += result
            if self.callback:
                self.callback(write_bytes=self.bytes_written, total_bytes=total_bytes_to_write)
            return result

        # For large data, write in chunks
        for i in range(0, total_bytes_to_write, self.chunk_size):
            chunk = data[i: i + self.chunk_size]
            result = super().write(chunk)
            bytes_written_this_call += result
            self.bytes_written += result

            # Call the callback with current overall progress
            if self.callback:
                self.callback(write_bytes=self.bytes_written, total_bytes=total_bytes_to_write)

        return bytes_written_this_call

    def seek(self, offset, whence=0):
        """
        Override seek to maintain proper position tracking.

        This method ensures that the bytes_read counter is reset when
        seeking back to the start of the file, which is important for
        accurate progress reporting if the file is read multiple times.

        Args:
            offset: The offset in bytes
            whence: 0=from start, 1=from current position, 2=from end

        Returns:
            The new absolute position
        """
        # Call the parent seek method first to get the new absolute position
        new_position = super().seek(offset, whence)

        # Update bytes_read to reflect the actual file pointer position
        self.bytes_read = new_position

        # Note: bytes_written is typically not affected by seek, as writes
        #       are generally considered additive from the current position.
        #       If seeking affected write progress, a similar update would be needed for bytes_written.

        return new_position


# Methods --------------------------------------------------------------------------------------------------------------

def _get_chunks_number(chunk_size: int = 0, file_size: int = 0) -> int:
    if chunk_size < 0 or file_size < 0:
        raise ValueError("chunk_size and file_size must be >= 0")

    if chunk_size == 0:
        if file_size == 0:
            return 0  # 0-length file results in 0 chunks
        else:
            # Cannot split a non-empty file into chunks of size 0
            raise ValueError("chunk_size cannot be 0 if file_size is greater than 0")

    return (file_size + chunk_size - 1) // chunk_size


def _get_chunk_size(chunk_size: int = 0, chunks: int = 0, file_size: int = 0) -> int:
    if chunk_size < 0 or chunks < 0 or file_size < 0:
        raise ValueError("chunk_size, chunks, and file_size must be >= 0")

    if chunk_size:
        return chunk_size
    elif chunks:
        # Calculate chunk_size using ceiling division to ensure
        # n-1 chunks have this size and the last chunk is smaller or equal.
        chunk_size = (file_size + chunks - 1) // chunks
    else:
        # Default to file_size if neither chunk_size nor chunks is specified,
        # treating the entire file as a single chunk.
        chunk_size = file_size

    return max(chunk_size, 1)
