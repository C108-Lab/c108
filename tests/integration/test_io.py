#
# C108 - IO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
import threading
from typing import Callable

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

pytestmark = pytest.mark.integration

# Local ----------------------------------------------------------------------------------------------------------------
from c108.io import StreamingFile

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


# Integration Tests ----------------------------------------------------------------------------------------------------


class TestStreamingFileConcurrency:
    """
    Integration tests for StreamingFile thread safety and concurrent operations.
    """

    @pytest.mark.flaky(reruns=2)
    def test_concurrent_reads_sum_and_progress(
        self, temp_file: str, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Read concurrently and assert total bytes and monotonic progress."""
        callback, calls = callback_tracker
        readers = 4
        read_size = 256
        totals: list[int] = []

        with StreamingFile(temp_file, mode="rb", callback=callback, chunk_size=CHUNK_SIZE) as f:

            def worker() -> None:
                total = 0
                while True:
                    chunk = f.read(size=read_size)
                    if not chunk:
                        break
                    total += len(chunk)
                totals.append(total)

            threads = [threading.Thread(target=worker) for _ in range(readers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert sum(totals) == FILE_SIZE
            assert f.bytes_read == FILE_SIZE

        assert len(calls) > 0
        # Verify monotonic non-decreasing progress and final completion
        for i in range(1, len(calls)):
            assert calls[i][0] >= calls[i - 1][0]
        assert calls[-1] == (FILE_SIZE, FILE_SIZE)

    @pytest.mark.flaky(reruns=2)
    def test_concurrent_appends_accumulate_and_callbacks(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Append concurrently and validate final size and callback completion."""
        callback, calls = callback_tracker
        writers = 8
        block_len = 512  # Smaller than chunk_size to keep each write atomic at API level
        patterns = [bytes([i]) * block_len for i in range(writers)]
        expected_total = writers * block_len
        out_path = tmp_path / "append.bin"

        with StreamingFile(
            str(out_path),
            mode="wb",
            callback=callback,
            chunk_size=1024,
            expected_size=expected_total,
        ) as f:

            def worker(data: bytes) -> None:
                # Single API write call per thread to test serialization on the shared file pointer
                written = f.write(data=data)
                assert written == len(data)

            threads = [threading.Thread(target=worker, args=(p,)) for p in patterns]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert f.bytes_written == expected_total

        data = out_path.read_bytes()
        assert len(data) == expected_total
        # Validate the resulting file is a permutation of the blocks (order unspecified)
        blocks = [data[i : i + block_len] for i in range(0, len(data), block_len)]
        assert len(blocks) == writers
        # Each pattern must appear exactly once
        for p in patterns:
            assert blocks.count(p) == 1

        assert len(calls) == writers
        assert calls[-1] == (expected_total, expected_total)

    @pytest.mark.flaky(reruns=2)
    def test_concurrent_seek_writes_disjoint_regions(
        self, tmp_path, callback_tracker: tuple[Callable, list]
    ) -> None:
        """Write to disjoint regions concurrently and verify exact layout."""
        callback, calls = callback_tracker
        writers = 4
        seg_size = 2048  # Larger than chunk size to exercise internal chunking with lock
        chunk_size = 1024
        out_path = tmp_path / "regions.bin"
        expected_total = writers * seg_size

        patterns = [bytes([i + 1]) * seg_size for i in range(writers)]

        with StreamingFile(
            str(out_path),
            mode="wb+",
            callback=callback,
            chunk_size=chunk_size,
            expected_size=expected_total,
        ) as f:

            def worker(idx: int) -> None:
                offset = idx * seg_size
                # Seek to region and write entire segment in one API call
                f.seek(offset=offset, whence=0)
                w = f.write(data=patterns[idx])
                assert w == seg_size

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(writers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            f.flush()
            assert f.bytes_written == expected_total

        # Verify file size and exact regions content
        assert os.path.getsize(out_path) == expected_total
        result = out_path.read_bytes()
        for i in range(writers):
            start = i * seg_size
            end = start + seg_size
            assert result[start:end] == patterns[i]

        # Progress should complete
        assert calls[-1] == (expected_total, expected_total)

    @pytest.mark.flaky(reruns=2)
    def test_progress_polling_during_concurrent_writes(self, tmp_path) -> None:
        """Poll progress concurrently and ensure it is non-decreasing and completes."""
        out_path = tmp_path / "progress_concurrent.bin"
        writers = 5
        block_len = 4 * 1024
        total = writers * block_len
        chunk_size = 1024

        percents: list[float] = []
        lock = threading.Lock()

        with StreamingFile(
            str(out_path),
            mode="wb",
            callback=lambda _, __: None,
            chunk_size=chunk_size,
            expected_size=total,
        ) as f:

            def write_worker(b: bytes) -> None:
                # Write large block in one call to exercise internal chunk splitting
                f.write(data=b)

            def poll_worker() -> None:
                # Poll frequently while writes are happening
                while f.bytes_written < total:
                    with lock:
                        percents.append(f.progress_percent)
                # Capture final percent
                with lock:
                    percents.append(f.progress_percent)

            threads = [
                threading.Thread(target=write_worker, args=(bytes([i + 2]) * block_len,))
                for i in range(writers)
            ]
            poller = threading.Thread(target=poll_worker)

            poller.start()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            poller.join()

            assert f.bytes_written == total
            assert abs(f.progress_percent - 100.0) < 1e-6

        # Ensure polled percents are non-decreasing and end at 100
        if percents:
            for i in range(1, len(percents)):
                assert percents[i] >= percents[i - 1]
            assert abs(percents[-1] - 100.0) < 1e-6
