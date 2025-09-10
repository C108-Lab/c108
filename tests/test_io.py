#
# C108 - IO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os

# Local ----------------------------------------------------------------------------------------------------------------
from c108.io import StreamingFile


# Tests ----------------------------------------------------------------------------------------------------------------

class TestStreamingFile:

    def custom_callback(self, read_bytes: int = None, write_bytes: int = None, total_bytes: int = None):
        total_bytes = "-" if total_bytes is None else total_bytes
        if read_bytes is not None:
            print(f"Reading: {read_bytes}/{total_bytes} bytes")
        if write_bytes is not None:
            print(f"Writing: {write_bytes}/{total_bytes} bytes")

    def test_file_read(self, temp_file):

        size_byte = 1024
        src_path = temp_file(size=size_byte, unit="B")

        with StreamingFile(src_path, 'rb', callback=self.custom_callback, chunk_size=256) as sf:
            print()
            print(f"File path : {src_path}")
            print(f"File size : {os.path.getsize(src_path)} bytes")
            print(f"Chunk size: {sf.chunk_size} bytes")
            print(f"Chunks    : {sf.chunks}")

            src_data = sf.read(size=-1)
            # After reading, check internal counters of StreamingFile objects
            assert sf.bytes_read == len(src_data)

        with StreamingFile(src_path, 'rb', callback=self.custom_callback, chunks=7) as sf:
            print()
            print(f"File path : {src_path}")
            print(f"File size : {os.path.getsize(src_path)} bytes")
            print(f"Chunk size: {sf.chunk_size} bytes")
            print(f"Chunks    : {sf.chunks}")

            src_data = sf.read(size=-1)
            # After reading, check internal counters of StreamingFile objects
            assert sf.bytes_read == len(src_data)

    def test_file_write(self, temp_file):

        size_byte = 2 * 1024
        chunk_size = 2 * 256
        file_path = temp_file(size=size_byte)

        print(f"File path : {file_path}")
        print(f"File size : {os.path.getsize(file_path)} bytes")
        print(f"Chunk size: {chunk_size} bytes")
        print()

        with StreamingFile(file_path, 'w+',
                           callback=self.custom_callback,
                           chunk_size=chunk_size) as f:
            new_data = b'1' * size_byte
            f.write(new_data)
            # After copying, check internal counters of StreamingFile objects
            assert f.bytes_written == len(new_data)
