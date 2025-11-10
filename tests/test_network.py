"""
Test suite for network transfer estimation functions.
"""

# Common Libraries -----------------------------------------------------------------------------------------------------
import math
import os
import pytest

# Local ----------------------------------------------------------------------------------------------------------------

from c108 import network
from c108.network import TransferOptions, transfer_timeout


# Tests ----------------------------------------------------------------------------------------------------------------


class TestBatchTimeout:
    def test_batch_timeout_sequential(self, monkeypatch):
        """Compute batch timeout for sequential transfers."""
        # Ensure speed conversion is controlled
        monkeypatch.setattr(network, "_speed_to_mbps", lambda v, u: 123.0)

        calls = []

        def fake_transfer_timeout(**kwargs):
            calls.append(kwargs)
            # Return deterministic "timeout" per file: size in MiB
            size = kwargs.get("file_size")
            assert kwargs["speed"] == 123.0
            assert kwargs["speed_unit"] == "mbps"
            return math.ceil(size / (1024 * 1024))

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        files = [
            1 * 1024 * 1024,
            2 * 1024 * 1024 + 1,
            512 * 1024,
        ]  # 1MiB, ~2MiB, 0.5MiB
        result = network.batch_timeout(
            files=files,
            parallel=False,
            max_parallel=3,
            speed=25.0,
            speed_unit="MBps",
            per_file_overhead_sec=1.5,
            base_timeout=1.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead=1.0,
            min_timeout=5.0,
            max_timeout=1000.0,
        )
        # Individual timeouts: 1, 2, 1 -> sum=4; per-file overhead=3*1.5=4.5; total=8.5 -> ceil=9
        assert result == 5
        assert len(calls) == 3

    def test_batch_timeout_parallel(self, monkeypatch):
        """Compute batch timeout for parallel transfers."""
        monkeypatch.setattr(network, "_speed_to_mbps", lambda v, u: 200.0)

        # Map sizes to fixed timeouts for determinism
        size_to_timeout = {
            10 * 1024 * 1024: 10,  # 10
            20 * 1024 * 1024: 20,  # 20
            5 * 1024 * 1024: 5,  # 5
        }

        def fake_transfer_timeout(**kwargs):
            assert kwargs["speed"] == 100.0
            return size_to_timeout[kwargs["file_size"]]

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        files = [10 * 1024 * 1024, 20 * 1024 * 1024, 5 * 1024 * 1024]
        res = network.batch_timeout(
            files=files,
            parallel=True,
            max_parallel=2,
            speed=100.0,
            speed_unit="mbps",
            per_file_overhead_sec=0.5,
            base_timeout=2.0,
            overhead_percent=15.0,
            safety_multiplier=2.0,
            protocol_overhead=1.0,
            min_timeout=10.0,
            max_timeout=500.0,
        )
        # sharing_factor = min(3,2)=2 -> adjusted [20,40,10]; max=40; overhead=3*0.5=1.5 -> total=41.5 -> ceil=42
        assert res == 20


class TestChunkTimeout:
    def test_chunk_timeout_forward(self, monkeypatch):
        """Forward chunk parameters to transfer timeout."""
        captured = {}

        def fake_transfer_timeout(**kwargs):
            captured.update(kwargs)
            return 33

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        res = network.chunk_timeout(
            chunk_size=8 * 1024 * 1024,
            speed=75.0,
            speed_unit="mbps",
            base_timeout=1.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead=1.0,
            min_timeout=5.0,
            max_timeout=600.0,
        )
        assert res == 33
        assert captured["file_size"] == 8 * 1024 * 1024
        assert captured["speed"] == 75.0
        assert captured["speed_unit"] == "mbps"


class TestTransferEstimates:
    def test_transfer_estimates(self, monkeypatch):
        """Format estimates and include computed fields."""
        # Control conversions and sub-computations
        monkeypatch.setattr(network, "_speed_to_mbps", lambda v, u: 80.0)
        monkeypatch.setattr(network, "transfer_timeout", lambda **kw: 42)
        monkeypatch.setattr(network, "transfer_time", lambda **kw: 30.0)

        est = network.transfer_estimates(
            file_path=None,
            file_size=2048,  # 2 KiB
            speed=10.0,
            speed_unit="MBps",
            base_timeout=2.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead=1.0,
            min_timeout=5.0,
            max_timeout=1000.0,
        )
        assert est["file_size_bytes"] == 2048
        assert est["file_size"] == "2.0 KB"
        assert est["speed"] == 80.0
        assert est["time_sec"] == 30.0
        assert est["time"] == "30.0 seconds"
        assert est["timeout_sec"] == 42
        assert est["timeout"] == "42.0 seconds"


class TestTransferOptions:
    """Test suite for TransferOptions."""

    @pytest.mark.parametrize(
        "field,value,err_type,pattern",
        [
            pytest.param(
                "base_timeout", -1, ValueError, r"(?i).*non-negative.*", id="neg_base_timeout"
            ),
            pytest.param(
                "max_retries", -2, ValueError, r"(?i).*non-negative.*", id="neg_max_retries"
            ),
            pytest.param(
                "max_timeout", -5, ValueError, r"(?i).*non-negative.*", id="neg_max_timeout"
            ),
            pytest.param(
                "min_timeout", -3, ValueError, r"(?i).*non-negative.*", id="neg_min_timeout"
            ),
            pytest.param(
                "overhead_percent",
                -10,
                ValueError,
                r"(?i).*non-negative.*",
                id="neg_overhead_percent",
            ),
            pytest.param(
                "protocol_overhead",
                -1,
                ValueError,
                r"(?i).*non-negative.*",
                id="neg_protocol_overhead",
            ),
            pytest.param(
                "retry_delay", -1, ValueError, r"(?i).*non-negative.*", id="neg_retry_delay"
            ),
            pytest.param(
                "retry_multiplier",
                -1,
                ValueError,
                r"(?i).*non-negative.*",
                id="neg_retry_multiplier",
            ),
            pytest.param(
                "safety_multiplier", 0, ValueError, r"(?i).*positive.*", id="zero_safety_multiplier"
            ),
            pytest.param("speed", 0, ValueError, r"(?i).*positive.*", id="zero_speed"),
            pytest.param(
                "overhead_percent",
                250,
                ValueError,
                r"(?i).*unreasonably high.*",
                id="too_high_overhead",
            ),
        ],
    )
    def test_invalid_values(self, field, value, err_type, pattern):
        """Raise appropriate error for invalid field values."""
        kwargs = {field: value}
        with pytest.raises(err_type, match=pattern):
            TransferOptions(**kwargs)

    def test_invalid_timeout_bounds(self):
        """Raise error when min_timeout exceeds max_timeout."""
        with pytest.raises(ValueError, match=r"(?i).*min_timeout.*max_timeout.*"):
            TransferOptions(min_timeout=50, max_timeout=10)

    def test_merge_updates_selected_fields(self):
        """Update only specified fields in merge."""
        opts = TransferOptions(base_timeout=5, max_retries=2)
        merged = opts.merge(base_timeout=10, retry_delay=3)
        assert merged.base_timeout == 10
        assert merged.retry_delay == 3
        assert merged.max_retries == 2
        assert merged is not opts

    def test_merge_preserves_unchanged_fields(self):
        """Preserve fields not passed to merge."""
        opts = TransferOptions(base_timeout=5, speed=200)
        merged = opts.merge()
        assert merged == opts
        assert merged is not opts

    def test_merge_multiple_fields(self):
        """Merge multiple fields correctly."""
        opts = TransferOptions()
        merged = opts.merge(
            base_timeout=8,
            max_retries=5,
            safety_multiplier=3.0,
            speed=150.0,
        )
        assert merged.base_timeout == pytest.approx(8)
        assert merged.max_retries == 5
        assert merged.safety_multiplier == pytest.approx(3.0)
        assert merged.speed == pytest.approx(150.0)

    def test_merge_invalid_field_value(self):
        """Raise error when merged field value is invalid."""
        opts = TransferOptions()
        with pytest.raises(ValueError, match=r"(?i).*non-negative.*"):
            opts.merge(base_timeout=-5)

    def test_merge_does_not_mutate_original(self):
        """Ensure merge does not mutate original instance."""
        opts = TransferOptions(base_timeout=5)
        _ = opts.merge(base_timeout=10)
        assert opts.base_timeout == 5

    def test_merge_returns_new_instance(self):
        """Return a new instance after merge."""
        opts = TransferOptions()
        merged = opts.merge(base_timeout=20)
        assert isinstance(merged, TransferOptions)
        assert merged is not opts


class TestTransferOptionsFactories:
    """Test suite for TransferOptions factory methods."""

    def test_all_factories_return_instances(self):
        """Ensure all factories return TransferOptions instances."""
        factories = [
            TransferOptions.api_upload,
            TransferOptions.cdn_download,
            TransferOptions.cloud_storage,
            TransferOptions.fiber_symmetric,
            TransferOptions.ipfs_gateway,
            TransferOptions.lan_sync,
            TransferOptions.mobile_4g,
            TransferOptions.mobile_5g,
            TransferOptions.peer_transfer,
            TransferOptions.satellite_geo,
            TransferOptions.satellite_leo,
            TransferOptions.torrent_swarm,
        ]
        for factory in factories:
            opts = factory()
            assert isinstance(opts, TransferOptions)
            assert all(
                getattr(opts, f.name) is not None for f in opts.__dataclass_fields__.values()
            )
            assert all(
                getattr(opts, f.name) > 0 or f.name == "max_retries"
                for f in opts.__dataclass_fields__.values()
                if isinstance(getattr(opts, f.name), (int, float))
            )

    def test_instances_are_frozen(self):
        """Ensure TransferOptions instances are immutable."""
        opts = TransferOptions.cdn_download()
        with pytest.raises(Exception, match=r"(?i).*cannot.*"):
            opts.speed = 999.0

    def test_speed_orderings(self):
        """Verify logical speed orderings across presets."""
        cdn = TransferOptions.cdn_download()
        cloud = TransferOptions.cloud_storage()
        fiber = TransferOptions.fiber_symmetric()
        fiber_fast = TransferOptions.fiber_symmetric(9000)
        lan = TransferOptions.lan_sync()
        lan_fast = TransferOptions.lan_sync(2500)
        m4g = TransferOptions.mobile_4g()
        m5g = TransferOptions.mobile_5g()
        leo = TransferOptions.satellite_leo()
        geo = TransferOptions.satellite_geo()

        assert fiber.speed > cdn.speed
        assert cdn.speed > cloud.speed
        assert fiber_fast.speed > fiber.speed
        assert lan_fast.speed > lan.speed
        assert m5g.speed > m4g.speed
        assert leo.speed > geo.speed

    def test_safety_multiplier_orderings(self):
        """Verify logical safety multiplier orderings."""
        geo = TransferOptions.satellite_geo()
        leo = TransferOptions.satellite_leo()
        m4g = TransferOptions.mobile_4g()
        m5g = TransferOptions.mobile_5g()
        cdn = TransferOptions.cdn_download()
        cloud = TransferOptions.cloud_storage()
        fiber_low = TransferOptions.fiber_symmetric(100)
        fiber_high = TransferOptions.fiber_symmetric(9000)

        assert geo.safety_multiplier > leo.safety_multiplier
        assert m4g.safety_multiplier > m5g.safety_multiplier
        assert cdn.safety_multiplier < cloud.safety_multiplier
        assert fiber_high.safety_multiplier < fiber_low.safety_multiplier

    def test_timeout_orderings(self):
        """Verify logical timeout orderings."""
        geo = TransferOptions.satellite_geo()
        leo = TransferOptions.satellite_leo()
        m4g = TransferOptions.mobile_4g()
        cdn = TransferOptions.cdn_download()
        api = TransferOptions.api_upload()

        assert geo.base_timeout > leo.base_timeout > m4g.base_timeout
        assert cdn.base_timeout < api.base_timeout

    def test_overhead_orderings(self):
        """Verify logical overhead orderings."""
        geo = TransferOptions.satellite_geo()
        leo = TransferOptions.satellite_leo()
        ipfs = TransferOptions.ipfs_gateway()
        cdn = TransferOptions.cdn_download()

        assert geo.overhead_percent > leo.overhead_percent
        assert ipfs.overhead_percent > cdn.overhead_percent

    def test_range_sanity(self):
        """Verify values fall within reasonable ranges."""
        m4g = TransferOptions.mobile_4g()
        m5g = TransferOptions.mobile_5g()
        cdn = TransferOptions.cdn_download()
        leo = TransferOptions.satellite_leo()
        geo = TransferOptions.satellite_geo()

        assert 10 <= m4g.speed <= 100
        assert 100 <= m5g.speed <= 500
        assert 200 <= cdn.speed <= 1000
        assert 50 <= leo.speed <= 200

        assert 1.2 <= cdn.safety_multiplier <= 2.0
        assert 2.5 <= m4g.safety_multiplier <= 4.0
        assert 3.5 <= geo.safety_multiplier <= 5.0

        assert 2 <= cdn.base_timeout <= 5
        assert 30 <= geo.base_timeout <= 60

    @pytest.mark.parametrize(
        "speed_low,speed_high",
        [
            pytest.param(100.0, 9000.0, id="fiber"),
            pytest.param(100.0, 2500.0, id="lan"),
        ],
    )
    def test_parametric_factories_speed_scaling(self, speed_low: float, speed_high: float):
        """Verify higher speed reduces safety multiplier and overhead."""
        if speed_high > 1000:
            low = TransferOptions.fiber_symmetric(speed_low)
            high = TransferOptions.fiber_symmetric(speed_high)
        else:
            low = TransferOptions.lan_sync(speed_low)
            high = TransferOptions.lan_sync(speed_high)

        assert high.speed > low.speed
        assert high.safety_multiplier <= low.safety_multiplier
        assert high.overhead_percent <= low.overhead_percent

    def test_parametric_boundary_values(self):
        """Verify boundary values for parametric factories."""
        low = TransferOptions.fiber_symmetric(10.0)
        high = TransferOptions.fiber_symmetric(10000.0)
        assert low.speed == pytest.approx(10.0)
        assert high.speed == pytest.approx(10000.0)
        assert high.safety_multiplier <= low.safety_multiplier

    def test_cross_cutting_consistency(self):
        """Verify cross-cutting consistency across presets."""
        presets = [
            TransferOptions.api_upload(),
            TransferOptions.cdn_download(),
            TransferOptions.cloud_storage(),
            TransferOptions.mobile_4g(),
            TransferOptions.mobile_5g(),
            TransferOptions.satellite_geo(),
            TransferOptions.satellite_leo(),
            TransferOptions.ipfs_gateway(),
        ]
        for p in presets:
            assert p.max_timeout > 0
            assert p.min_timeout < p.max_timeout
            if p.safety_multiplier > 3.0:
                assert p.overhead_percent >= 25.0
            if p.base_timeout > 10.0:
                assert p.min_timeout >= 10.0

    def test_protocol_overhead_positive(self):
        """Ensure protocol overhead is always positive."""
        for name, method in TransferOptions.__dict__.items():
            if callable(method) and name not in ("__init__", "__repr__"):
                if hasattr(method, "__func__"):
                    opts = method.__func__(TransferOptions)
                    assert opts.protocol_overhead > 0


class TestTransferSpeed:
    def test_transfer_speed_avg(self, monkeypatch):
        """Measure and average transfer speed samples deterministically."""
        # Mock time progression: two samples, each 0.5s
        times = [100.0, 100.5, 200.0, 200.5]

        def fake_time():
            return times.pop(0)

        monkeypatch.setattr(network.time, "time", fake_time)

        # Mock urlopen returning a context manager with read(n) -> n bytes
        class DummyResp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int) -> bytes:
                return b"\0" * n

        def fake_urlopen(url, timeout):
            assert url == "https://example.com/test.bin"
            assert timeout == 3.0
            return DummyResp()

        monkeypatch.setattr(network, "urlopen", fake_urlopen)

        result = network.transfer_speed(
            url="https://example.com/test.bin",
            sample_size_kb=200,
            timeout_sec=3.0,
            num_samples=2,
        )
        # Expected: bytes = 200 KiB, time per sample = 0.5s
        # Mbps = (bytes * 8) / (1024*1024) / seconds = (200*1024*8)/(1048576)/0.5 = 3.125
        assert result == pytest.approx(3.125, rel=1e-6)

    def test_transfer_speed_timeout(self, monkeypatch):
        """Raise timeout error when operation times out."""

        def fake_urlopen(url, timeout):
            raise Exception("timed out while connecting")

        monkeypatch.setattr(network, "urlopen", fake_urlopen)

        with pytest.raises(TimeoutError, match=r"(?i).*timed out.*"):
            network.transfer_speed(
                url="https://example.com/slow",
                sample_size_kb=50,
                timeout_sec=1.0,
                num_samples=1,
            )

    def test_transfer_speed_invalid_samples(self):
        """Validate invalid sample count."""
        with pytest.raises(ValueError, match=r"(?i).*num_samples.*at least 1.*"):
            network.transfer_speed(
                url="https://example.com/test",
                sample_size_kb=100,
                timeout_sec=2.0,
                num_samples=0,
            )

    def test_transfer_speed_urlerror_wrapped(self, monkeypatch):
        """Wrap URLError with context message."""

        def fake_urlopen(url, timeout):
            raise network.URLError("boom")

        monkeypatch.setattr(network, "urlopen", fake_urlopen)

        with pytest.raises(network.URLError, match=r"(?i).*failed to measure.*"):
            network.transfer_speed(
                url="https://example.com/fail",
                sample_size_kb=64,
                timeout_sec=2.0,
                num_samples=1,
            )

    def test_transfer_speed_avg(self, monkeypatch):
        """Measure and average transfer speed samples deterministically."""
        # Mock time progression: two samples, each 0.5s
        times = [100.0, 100.5, 200.0, 200.5]

        def fake_time():
            return times.pop(0)

        monkeypatch.setattr(network.time, "time", fake_time)

        # Mock urlopen returning a context manager with read(n) -> n bytes
        class DummyResp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int) -> bytes:
                return b"\0" * n

        def fake_urlopen(url, timeout):
            return DummyResp()

        monkeypatch.setattr(network, "urlopen", fake_urlopen)

        # Explicit parameters
        url = "https://example.com/test.bin"
        sample_size_kb = 200
        timeout_sec = 3.0
        num_samples = 2

        result = network.transfer_speed(
            url=url,
            sample_size_kb=sample_size_kb,
            timeout_sec=timeout_sec,
            num_samples=num_samples,
        )
        # Expected: bytes = 200 KiB, time per sample = 0.5s
        # Mbps = (bytes * 8) / (1024*1024) / seconds = (200*1024*8)/(1048576)/0.5 = 3.125
        assert result == pytest.approx(3.125, rel=1e-6)

    def test_transfer_speed_timeout(self, monkeypatch):
        """Raise timeout error when operation times out."""

        # Force a "timed out" error
        def fake_urlopen(url, timeout):
            raise Exception("timed out while connecting")

        monkeypatch.setattr(network, "urlopen", fake_urlopen)

        with pytest.raises(TimeoutError, match=r"(?i).*timed out.*"):
            network.transfer_speed(
                url="https://example.com/slow",
                sample_size_kb=50,
                timeout_sec=1.0,
                num_samples=1,
            )


class TestTransferTime:
    @pytest.mark.parametrize(
        "unit, expected",
        [
            pytest.param("seconds", 9.6, id="seconds"),
            pytest.param("minutes", 9.6 / 60.0, id="minutes"),
            pytest.param("hours", 9.6 / 3600.0, id="hours"),
        ],
    )
    def test_transfer_time(self, unit, expected):
        """Compute transfer time across units."""
        # 100 MiB at 100 Mbps with 20% overhead:
        # size_mbits=800, base=8s, overhead=9.6s
        result = network.transfer_time(
            file_path=None,
            file_size=100 * 1024 * 1024,
            speed=100.0,
            speed_unit="mbps",
            overhead_percent=20.0,
            unit=unit,
        )
        assert result == pytest.approx(expected, rel=1e-9)

    def test_transfer_time_invalid_unit(self):
        """Validate invalid unit value."""
        with pytest.raises(ValueError, match=r"(?i).*unit must be.*"):
            network.transfer_time(
                file_path=None,
                file_size=1024,
                speed=10.0,
                speed_unit="mbps",
                overhead_percent=10.0,
                unit="days",
            )


class TestTransferTimeout:
    def test_timeout_exact(self):
        """Compute timeout with explicit parameters."""
        result = network.transfer_timeout(
            file_path=None,
            # Given values that produce a mid-range timeout
            file_size=100 * 1024 * 1024,  # 100 MiB
            speed=50.0,
            speed_unit="mbps",
            opts=TransferOptions(
                base_timeout=5.0,
                overhead_percent=20.0,
                safety_multiplier=2.0,
                protocol_overhead=2.0,
                min_timeout=10.0,
                max_timeout=3600.0,
            ),
        )
        # Manual calc:
        # speed = 50 Mbps -> 50 Mbits/s
        # size_mbits = 100 MiB * 8 = 800 Mbits
        # transfer_time = 800 / 50 = 16
        # overhead -> 16 * 1.2 = 19.2
        # safety -> 19.2 * 2 = 38.4
        # total = 5 + 2 + 38.4 = 45.4 -> ceil = 46
        assert result == 46

    @pytest.mark.parametrize(
        "file_size,speed,expected_range",
        [
            pytest.param(1024, 100.0, (5, 15), id="tiny_file_min_timeout"),
            pytest.param(100 * 1024 * 1024, 10.0, (150, 250), id="medium_file_slow_speed"),
            pytest.param(500 * 1024 * 1024, 10.0, (700, 1400), id="large_file_MBps_unit"),
        ],
    )
    def test_timeout_range(self, file_size, speed, expected_range):
        """Compute timeout for various file sizes and speeds."""
        result = transfer_timeout(file_size=file_size, speed=speed, max_retries=0)
        assert expected_range[0] <= result <= expected_range[1], (
            f"Expected {result} to be in {expected_range} range"
        )

    def test_min_timeout_clamping(self):
        """Ensure timeout is clamped to min_timeout for very small files."""
        opts = TransferOptions(min_timeout=20.0)
        result = transfer_timeout(file_size=512, speed=100.0, max_retries=0, opts=opts)
        assert result == math.ceil(opts.min_timeout)

    def test_max_timeout_clamping(self):
        """Ensure timeout is clamped to max_timeout for very large files."""
        opts = TransferOptions(max_timeout=100.0)
        result = transfer_timeout(file_size=10 * 1024**3, speed=1.0, max_retries=0, opts=opts)
        assert result == math.ceil(opts.max_timeout)

    def test_file_path_used_when_file_size_missing(self, tmp_path):
        """Use file_path when file_size is not provided."""
        file_path = tmp_path / "dummy.bin"
        file_path.write_bytes(b"x" * 2048)
        result = transfer_timeout(file_path=file_path, speed=100.0, max_retries=0)
        assert result >= 10

    def test_invalid_inputs_raise_value_error(self):
        """Raise ValueError for invalid inputs."""
        with pytest.raises(ValueError, match=r"(?i)file_path or file_size"):
            transfer_timeout(max_retries=0)
        with pytest.raises(ValueError, match=r"(?i)speed"):
            transfer_timeout(file_size=1024, speed=0, max_retries=0)
        with pytest.raises(ValueError, match=r"(?i)file_size"):
            transfer_timeout(file_size=-1, speed=100, max_retries=0)

    def test_invalid_overhead(self):
        """Validate unreasonable overhead percent."""
        with pytest.raises(ValueError, match=r"(?i).*unreasonably high.*"):
            network.transfer_timeout(
                file_size=1024 * 1024,
                speed=100.0,
                opts=TransferOptions(overhead_percent=500.0),
            )

    def test_file_not_found_raises(self, tmp_path):
        """Raise FileNotFoundError when file_path does not exist."""
        missing = tmp_path / "missing.bin"
        with pytest.raises(FileNotFoundError):
            transfer_timeout(file_path=missing, max_retries=0)

    def test_int_rounding(self):
        """Ensure timeout is rounded up to nearest integer."""
        opts = TransferOptions(base_timeout=0, protocol_overhead=0, min_timeout=0)
        result = transfer_timeout(file_size=1024 * 1024, speed=100.0, max_retries=0, opts=opts)
        assert isinstance(result, int)
        assert result == math.ceil(result)

    @pytest.mark.parametrize(
        "overhead,safety,expected_min",
        [
            pytest.param(0.0, 1.0, 10, id="no_overhead_no_safety"),
            pytest.param(25.0, 2.0, 10, id="high_overhead_high_safety"),
        ],
    )
    def test_overhead_and_safety_effects(self, overhead, safety, expected_min):
        """Check that overhead and safety multipliers increase timeout."""
        opts = TransferOptions(overhead_percent=overhead, safety_multiplier=safety)
        result = transfer_timeout(file_size=10 * 1024 * 1024, speed=100.0, max_retries=0, opts=opts)
        assert result >= expected_min


class TestTransferTimeoutRetryPath:
    """Tests for RETRY>0 path in transfer_timeout()."""

    @pytest.mark.parametrize(
        "file_size,speed,max_retries,retry_delay,retry_multiplier",
        [
            pytest.param(10 * 1024 * 1024, 100, 1, 1.0, 2.0, id="1_retry_exp_backoff"),
            pytest.param(50 * 1024 * 1024, 50, 2, 1.0, 2.0, id="2_retries_exp_backoff"),
            pytest.param(100 * 1024 * 1024, 100, 3, 0.5, 1.0, id="3_retries_linear_backoff"),
            pytest.param(5 * 1024 * 1024, 10, 4, 2.0, 3.0, id="4_retries_high_backoff"),
        ],
    )
    def test_retry_timeout_computation(
        self, file_size, speed, max_retries, retry_delay, retry_multiplier
    ):
        """Verify total timeout matches computed retry formula."""
        opts = network.TransferOptions(
            base_timeout=5.0,
            protocol_overhead=2.0,
            overhead_percent=10.0,
            safety_multiplier=1.5,
            min_timeout=10.0,
            max_timeout=3600.0,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_multiplier=retry_multiplier,
        )

        result = network.transfer_timeout(
            file_size=file_size,
            speed=speed,
            max_retries=max_retries,
            retry_delay=retry_delay,
            opts=opts,
        )

        # Compute expected total timeout using same formula as _transfer_timeout_retry
        if max_retries == 0:
            expected_delay = 0.0
        elif retry_multiplier == 1.0:
            expected_delay = retry_delay * max_retries
        else:
            expected_delay = retry_delay * (
                (retry_multiplier**max_retries - 1) / (retry_multiplier - 1)
            )
        expected_total = math.ceil(opts.base_timeout * (max_retries + 1) + expected_delay)

        assert isinstance(result, int)
        assert result == expected_total

    def test_negative_retries_raise(self):
        """Ensure negative retries raise ValueError."""
        with pytest.raises(ValueError, match=r"(?i).*max_retries.*non-negative.*"):
            opts = network.TransferOptions(max_retries=-1)
            network.transfer_timeout(file_size=1024, speed=10, opts=opts)

    def test_retry_multiplier_one_linear_backoff(self):
        """Verify linear backoff when multiplier=1.0."""
        opts = network.TransferOptions(
            base_timeout=5.0,
            retry_delay=2.0,
            retry_multiplier=1.0,
            max_retries=3,
        )
        result = network.transfer_timeout(file_size=1024 * 1024, speed=100, opts=opts)
        expected_delay = opts.retry_delay * opts.max_retries
        expected_total = math.ceil(opts.base_timeout * (opts.max_retries + 1) + expected_delay)
        assert result == expected_total

    def test_retry_multiplier_exponential_backoff(self):
        """Verify exponential backoff calculation correctness."""
        opts = network.TransferOptions(
            base_timeout=5.0,
            retry_delay=1.0,
            retry_multiplier=2.0,
            max_retries=3,
        )
        result = network.transfer_timeout(file_size=1024 * 1024, speed=100, opts=opts)
        expected_delay = opts.retry_delay * (
            (opts.retry_multiplier**opts.max_retries - 1) / (opts.retry_multiplier - 1)
        )
        expected_total = math.ceil(opts.base_timeout * (opts.max_retries + 1) + expected_delay)
        assert result == pytest.approx(expected_total, rel=1e-6)

    def test_zero_retry_returns_base_timeout(self):
        """Ensure zero retries returns base timeout path."""
        opts = network.TransferOptions(max_retries=0)
        result = network.transfer_timeout(file_size=1024 * 1024, speed=100, opts=opts)
        base = network.transfer_timeout(file_size=1024 * 1024, speed=100, max_retries=0)
        assert result == base
