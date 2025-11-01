#
# C108 - IO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
import threading
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


@pytest.fixture
def temp_file(tmp_path) -> str:
    """Create a temporary file with known content and return its path."""
    p = tmp_path / "data.bin"
    # Fill with deterministic pattern
    block = b"0123456789ABCDEF" * 64  # 1024 bytes per block
    repeats = FILE_SIZE // len(block)
    remainder = FILE_SIZE % len(block)
    with open(p, "wb") as f:
        for _ in range(repeats):
            f.write(block)
        if remainder:
            f.write(block[:remainder])
    return str(p)


class TestStreamingFile:
    """
    Test suite for the StreamingFile class:
    initialization, chunking, callbacks, helpers, and thread safety.
    """

    def test_init_read_mode(self, temp_file: str) -> None:
        """Verify initialization in read mode sets total size from file."""
        tracker, _ = (lambda: (lambda *_: None, []))()
        with StreamingFile(
            temp_file, mode="rb", callback=tracker, chunk_size=CHUNK_SIZE
        ) as f:
            assert f.readable() is True
            assert f.writable() is False
            assert f.total_size == os.path.getsize(temp_file)
            assert f.file_size == os.path.getsize(temp_file)
            assert f.chunk_size == CHUNK_SIZE
            assert f.bytes_read == 0
            assert f.bytes_written == 0
            assert f.total_chunks == _get_chunks_number(
                chunk_size=CHUNK_SIZE,
                file_size=f.total_size,
            )

    def test_init_write_mode(self, temp_file: str) -> None:
        """Verify initialization in write mode uses expected_size."""
        expected = FILE_SIZE + 1024
        tracker, _ = (lambda: (lambda *_: None, []))()
        with StreamingFile(
            temp_file,
            mode="wb",
            callback=tracker,
            chunk_size=CHUNK_SIZE,
            expected_size=expected,
        ) as f:
            assert f.writable() is True
            assert f.readable() is False
            assert f.total_size == expected
            assert f.chunk_size == CHUNK_SIZE
            assert f.bytes_written == 0
            assert f.total_chunks == _get_chunks_number(
                chunk_size=CHUNK_SIZE, file_size=expected
            )

    def test_init_with_zero_chunk_size(self, temp_file: str) -> None:
        """Verify chunk_size=0 sets the chunk size to the total file size."""
        tracker, _ = (lambda: (lambda *_: None, []))()
        with StreamingFile(temp_file, mode="rb", callback=tracker, chunk_size=0) as f:
            assert f.total_size == os.path.getsize(temp_file)
            assert f.chunk_size == f.total_size
            assert f.total_chunks == 1

        expected = 7 * 1024
        with StreamingFile(
            temp_file, mode="wb", callback=tracker, chunk_size=0, expected_size=expected
        ) as f:
            assert f.total_size == expected
            assert f.chunk_size == expected
            assert f.total_chunks == 1

    def test_init_raises_on_empty_path(self) -> None:
        """Ensure initialization fails if the path is empty."""
        tracker, _ = (lambda: (lambda *_: None, []))()
        with pytest.raises(ValueError, match=r"(?i).*path.*"):
            StreamingFile("", mode="rb", callback=tracker, chunk_size=CHUNK_SIZE)

    def test_read_in_chunks(
        self, temp_file: str, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Read an entire file to verify chunked reading and callback calls."""
        callback, calls = callback_tracker
        with StreamingFile(
            temp_file, mode="rb", callback=callback, chunk_size=CHUNK_SIZE
        ) as f:
            data = f.read(size=FILE_SIZE)
            assert len(data) == FILE_SIZE
            expected_calls = _get_chunks_number(
                chunk_size=CHUNK_SIZE, file_size=FILE_SIZE
            )
            assert len(calls) == expected_calls
            # Verify monotonic progress and final completion
            for i in range(1, len(calls)):
                assert calls[i][0] >= calls[i - 1][0]
            assert calls[-1] == (FILE_SIZE, FILE_SIZE)

    def test_read_smaller_than_chunk(
        self, temp_file: str, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Read an amount smaller than chunk_size to test the optimization path."""
        callback, calls = callback_tracker
        small_size = CHUNK_SIZE // 2
        with StreamingFile(
            temp_file, mode="rb", callback=callback, chunk_size=CHUNK_SIZE
        ) as f:
            chunk = f.read(size=small_size)
            assert len(chunk) == small_size
            assert len(calls) == 1
            assert calls[0] == (small_size, os.path.getsize(temp_file))

    def test_read_from_empty_file(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Ensure reading from an empty file returns empty bytes and no callbacks."""
        callback, calls = callback_tracker
        empty_path = tmp_path / "empty.bin"
        empty_path.write_bytes(b"")
        with StreamingFile(
            str(empty_path), mode="rb", callback=callback, chunk_size=CHUNK_SIZE
        ) as f:
            out = f.read(size=1024)
            assert out == b""
            assert len(calls) == 0
            assert f.total_size == 0
            assert f.total_chunks == 0

    def test_write_empty_bytes(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Ensure writing empty bytes returns 0 and triggers no callbacks."""
        callback, calls = callback_tracker
        out_path = tmp_path / "empty_write.bin"

        with StreamingFile(
            str(out_path), mode="wb", callback=callback, chunk_size=CHUNK_SIZE
        ) as f:
            bytes_written = f.write(b"")
            assert bytes_written == 0
            assert len(calls) == 0
            assert f.total_size == 0
            assert f.total_chunks == 0

        assert out_path.exists()
        assert out_path.read_bytes() == b""

    def test_write_in_chunks(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Write data larger than chunk_size to verify chunked writing."""
        callback, calls = callback_tracker
        out_path = tmp_path / "out.bin"
        data = b"A" * (CHUNK_SIZE * 3 + 123)
        expected = len(data)
        with StreamingFile(
            str(out_path),
            mode="wb",
            callback=callback,
            chunk_size=CHUNK_SIZE,
            expected_size=expected,
        ) as f:
            written = f.write(data=data)
            f.flush()
            assert written == expected
            assert f.bytes_written == expected
            assert len(calls) == _get_chunks_number(
                chunk_size=CHUNK_SIZE, file_size=expected
            )
            assert calls[-1] == (expected, expected)
        assert os.path.getsize(out_path) == expected

    def test_write_smaller_than_chunk(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Write an amount smaller than chunk_size to test the optimization path."""
        callback, calls = callback_tracker
        out_path = tmp_path / "small.bin"
        data = b"B" * (CHUNK_SIZE // 4)
        expected = len(data)
        with StreamingFile(
            str(out_path),
            mode="wb",
            callback=callback,
            chunk_size=CHUNK_SIZE,
            expected_size=expected,
        ) as f:
            written = f.write(data=data)
            assert written == expected
            assert f.bytes_written == expected
            assert len(calls) == 1
            assert calls[0] == (expected, expected)

    @pytest.mark.parametrize(
        ("mode", "attribute_to_check"),
        [
            pytest.param("rb", "bytes_read", id="read_mode"),
            pytest.param("wb+", "bytes_written", id="write_mode"),
        ],
    )
    def test_seek_updates_progress(
        self, temp_file: str, mode: str, attribute_to_check: str
    ) -> None:
        """Verify seek updates the correct progress counter for the given mode."""
        tracker, _ = (lambda: (lambda *_: None, []))()
        expected_size = FILE_SIZE if "w" not in mode else (FILE_SIZE * 2)
        with StreamingFile(
            temp_file,
            mode=mode,
            callback=tracker,
            chunk_size=CHUNK_SIZE,
            expected_size=(expected_size if "w" in mode else None),
        ) as f:
            if "w" in mode:
                # Write some bytes to ensure a position context for write mode
                f.write(data=b"\x00" * (CHUNK_SIZE + 10))
            new_pos = FILE_SIZE // 2
            pos = f.seek(offset=new_pos, whence=0)
            assert pos == new_pos
            assert getattr(f, attribute_to_check) == new_pos

    def test_progress_percent_property(self, tmp_path) -> None:
        """Check the progress_percent property reflects the current state."""
        out_path = tmp_path / "progress.bin"
        total = CHUNK_SIZE * 2
        with StreamingFile(
            str(out_path),
            mode="wb",
            callback=lambda *_: None,
            chunk_size=CHUNK_SIZE,
            expected_size=total,
        ) as f:
            f.write(data=b"X" * CHUNK_SIZE)
            pct = f.progress_percent
            # Expect exactly 50.0
            assert abs(pct - 50.0) < 1e-6
            f.write(data=b"X" * (CHUNK_SIZE // 2))
            pct2 = f.progress_percent
            assert abs(pct2 - 75.0) < 1e-6

    @pytest.mark.parametrize(
        ("operation", "args"),
        [
            pytest.param("read", (1,), id="read_on_closed"),
            pytest.param("write", (b"data",), id="write_on_closed"),
            pytest.param("seek", (0, 0), id="seek_on_closed"),
        ],
    )
    def test_raises_on_operation_after_close(
        self, temp_file: str, operation: str, args: tuple
    ) -> None:
        """Ensure operations on a closed file raise a ValueError."""
        expected = FILE_SIZE + 4096
        with StreamingFile(
            temp_file,
            mode="wb+",
            callback=lambda *_: None,
            chunk_size=CHUNK_SIZE,
            expected_size=expected,
        ) as f:
            pass
        # Operate on closed file
        with pytest.raises(ValueError, match=r"(?i).*closed.*"):
            if operation == "read":
                StreamingFile(
                    temp_file,
                    mode="rb",
                    callback=lambda *_: None,
                    chunk_size=CHUNK_SIZE,
                ).close()
                # Freshly closed object to ensure consistent state
                g = StreamingFile(
                    temp_file,
                    mode="rb",
                    callback=lambda *_: None,
                    chunk_size=CHUNK_SIZE,
                )
                g.close()
                g.read(*args)
            elif operation == "write":
                g = StreamingFile(
                    temp_file,
                    mode="wb",
                    callback=lambda *_: None,
                    chunk_size=CHUNK_SIZE,
                    expected_size=expected,
                )
                g.close()
                g.write(*args)
            elif operation == "seek":
                g = StreamingFile(
                    temp_file,
                    mode="rb",
                    callback=lambda *_: None,
                    chunk_size=CHUNK_SIZE,
                )
                g.close()
                g.seek(*args)

    def test_get_chunks_number(self) -> None:
        """Validate _get_chunks_number handles exact, partial, and zero cases."""
        # Exact division
        assert _get_chunks_number(chunk_size=1024, file_size=4096) == 4
        # Partial last chunk
        assert _get_chunks_number(chunk_size=1024, file_size=4100) == 5
        # Zero total
        assert _get_chunks_number(chunk_size=1024, file_size=0) == 0
        # Chunk size larger than total
        assert _get_chunks_number(chunk_size=4096, file_size=1024) == 1
        # Chunk size equals zero implies single chunk equal to total (guarded by init)
        assert _get_chunks_number(chunk_size=0, file_size=8192) == 1

    # Thread-safety tests ---------------------------------
    # Run:
    #     pytest tests/integration/test_io.py
    #
