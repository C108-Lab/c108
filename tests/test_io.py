#
# C108 - IO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
from typing import Callable

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.io import StreamingFile, _get_chunks_number

# Tests ----------------------------------------------------------------------------------------------------------------

# A reasonably large size to test chunking behavior
FILE_SIZE = 10 * 1024
CHUNK_SIZE = 2 * 1024


@pytest.fixture
def callback_tracker() -> tuple[Callable[[int, int], None], list[tuple[int, int]]]:
    """Fixture that provides a callback and a list to track its calls."""
    calls = []

    def tracker(current_bytes: int, total_bytes: int) -> None:
        calls.append((current_bytes, total_bytes))

    return tracker, calls


class TestStreamingFile:
    """Test suite for the StreamingFile class."""

    def test_init_read_mode(self, temp_file):
        """Verify initialization in read mode sets total size from file."""
        file_path = temp_file(size=FILE_SIZE)
        sf = StreamingFile(file_path, "rb", chunk_size=CHUNK_SIZE)

        assert sf.total_size == FILE_SIZE
        assert sf.bytes_read == 0
        assert sf.chunk_size == CHUNK_SIZE
        assert not sf.is_closed

    def test_init_write_mode(self, temp_file):
        """Verify initialization in write mode uses expected_size."""
        file_path = temp_file(size=0)
        expected_size = 2 * FILE_SIZE
        sf = StreamingFile(
            file_path, "wb", expected_size=expected_size, chunk_size=CHUNK_SIZE
        )

        assert sf.total_size == expected_size
        assert sf.bytes_written == 0
        assert sf.chunk_size == CHUNK_SIZE

    def test_init_with_zero_chunk_size(self, temp_file):
        """Verify chunk_size=0 sets the chunk size to the total file size."""
        file_path = temp_file(size=FILE_SIZE)
        sf = StreamingFile(file_path, "rb", chunk_size=0)

        assert sf.chunk_size == FILE_SIZE

    def test_init_raises_on_empty_path(self):
        """Ensure initialization fails if the path is empty."""
        with pytest.raises(ValueError, match=r"(?i)path required"):
            StreamingFile("", "r")

    def test_read_in_chunks(self, temp_file, callback_tracker: tuple[Callable, list]):
        """Read an entire file to verify chunked reading and callback calls."""
        file_path = temp_file(size=FILE_SIZE, content=b"a")
        tracker, calls = callback_tracker

        with StreamingFile(
                file_path, "rb", callback=tracker, chunk_size=CHUNK_SIZE
        ) as sf:
            data = sf.read()

        assert data == b"a" * FILE_SIZE
        assert sf.bytes_read == FILE_SIZE
        num_chunks = _get_chunks_number(CHUNK_SIZE, FILE_SIZE)
        assert len(calls) == num_chunks
        # Check that each call reported the correct cumulative progress
        expected_calls = [
            ((i + 1) * CHUNK_SIZE, FILE_SIZE) for i in range(num_chunks)
        ]
        assert calls == expected_calls

    def test_read_smaller_than_chunk(self, temp_file, callback_tracker: tuple[Callable, list]):
        """Read an amount smaller than chunk_size to test the optimization path."""
        file_path = temp_file(size=FILE_SIZE)
        tracker, calls = callback_tracker
        read_size = CHUNK_SIZE // 2

        with StreamingFile(
                file_path, "rb", callback=tracker, chunk_size=CHUNK_SIZE
        ) as sf:
            data = sf.read(read_size)

        assert len(data) == read_size
        assert sf.bytes_read == read_size
        assert calls == [(read_size, FILE_SIZE)]

    def test_read_from_empty_file(self, temp_file, callback_tracker: tuple[Callable, list]):
        """Ensure reading from an empty file returns empty bytes and no callbacks."""
        file_path = temp_file(size=0)
        tracker, calls = callback_tracker

        with StreamingFile(file_path, "rb", callback=tracker) as sf:
            data = sf.read()

        assert data == b""
        assert sf.bytes_read == 0
        assert not calls

    def test_write_in_chunks(self, temp_file, callback_tracker: tuple[Callable, list]):
        """Write data larger than chunk_size to verify chunked writing."""
        file_path = temp_file(size=0)
        tracker, calls = callback_tracker
        data_to_write = b"b" * FILE_SIZE

        with StreamingFile(
                file_path,
                "wb",
                callback=tracker,
                chunk_size=CHUNK_SIZE,
                expected_size=FILE_SIZE,
        ) as sf:
            sf.write(data_to_write)

        assert file_path.read_bytes() == data_to_write
        assert sf.bytes_written == FILE_SIZE
        num_chunks = _get_chunks_number(CHUNK_SIZE, FILE_SIZE)
        assert len(calls) == num_chunks
        expected_calls = [
            ((i + 1) * CHUNK_SIZE, FILE_SIZE) for i in range(num_chunks)
        ]
        assert calls == expected_calls

    def test_write_smaller_than_chunk(self, temp_file, callback_tracker: tuple[Callable, list]):
        """Write an amount smaller than chunk_size to test the optimization path."""
        file_path = temp_file(size=0)
        tracker, calls = callback_tracker
        write_size = CHUNK_SIZE // 2
        data_to_write = b"c" * write_size

        with StreamingFile(
                file_path,
                "wb",
                callback=tracker,
                chunk_size=CHUNK_SIZE,
                expected_size=FILE_SIZE,
        ) as sf:
            sf.write(data_to_write)

        assert file_path.read_bytes() == data_to_write
        assert sf.bytes_written == write_size
        assert calls == [(write_size, FILE_SIZE)]

    @pytest.mark.parametrize(
        ("mode", "attribute_to_check"),
        [
            pytest.param("rb", "bytes_read", id="read_mode"),
            pytest.param("wb+", "bytes_written", id="write_mode"),
        ],
    )
    def test_seek_updates_progress(self, temp_file, mode: str, attribute_to_check: str):
        """Verify seek updates the correct progress counter for the given mode."""
        file_path = temp_file(size=FILE_SIZE)
        seek_position = FILE_SIZE // 2

        with StreamingFile(file_path, mode) as sf:
            new_pos = sf.seek(seek_position, os.SEEK_SET)
            assert new_pos == seek_position
            assert getattr(sf, attribute_to_check) == seek_position

    def test_progress_percent_property(self, temp_file):
        """Check the progress_percent property reflects the current state."""
        file_path = temp_file(size=FILE_SIZE)
        with StreamingFile(file_path, "rb", chunk_size=CHUNK_SIZE) as sf:
            assert sf.progress_percent == 0.0
            sf.read(FILE_SIZE // 4)
            assert sf.progress_percent == 25.0
            sf.read(FILE_SIZE // 4)
            assert sf.progress_percent == 50.0
            sf.seek(0)
            assert sf.progress_percent == 0.0
            sf.read()
            assert sf.progress_percent == 100.0

    @pytest.mark.parametrize(
        ("operation", "args"),
        [
            pytest.param("read", (), id="read_on_closed"),
            pytest.param("write", (b"data",), id="write_on_closed"),
            pytest.param("seek", (0,), id="seek_on_closed"),
        ],
    )
    def test_raises_on_operation_after_close(self, temp_file, operation: str, args: tuple):
        """Ensure operations on a closed file raise a ValueError."""
        file_path = temp_file(size=10)
        # Use "rb+" to allow both read and write for parametrization
        sf = StreamingFile(file_path, "rb+")
        sf.close()

        assert sf.is_closed
        with pytest.raises(ValueError, match=r"(?i)(closed file|file not open)"):
            getattr(sf, operation)(*args)
