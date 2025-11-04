"""
Test suite for network transfer estimation functions.
"""

# Common Libraries -----------------------------------------------------------------------------------------------------
import math
import pytest

# Local ----------------------------------------------------------------------------------------------------------------

from c108 import network

# Tests ----------------------------------------------------------------------------------------------------------------


class TestNetworkCore:
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

    def test_transfer_timeout_calc(self):
        """Compute timeout with explicit parameters."""
        # Given values that produce a mid-range timeout
        result = network.transfer_timeout(
            file_path=None,
            file_size=100 * 1024 * 1024,  # 100 MiB
            speed_mbps=50.0,
            speed_unit="mbps",
            base_timeout_sec=5.0,
            overhead_percent=20.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=2.0,
            min_timeout_sec=10.0,
            max_timeout_sec=3600.0,
        )
        # Manual calc:
        # size_mbits = 100 MiB * 8 = 800 Mbits
        # transfer_time = 800 / 50 = 16
        # overhead -> 16 * 1.2 = 19.2
        # safety -> 19.2 * 2 = 38.4
        # total = 5 + 2 + 38.4 = 45.4 -> ceil = 46
        assert result == 46

    def test_transfer_timeout_min_clamp(self):
        """Clamp to minimum timeout when total is too small."""
        result = network.transfer_timeout(
            file_path=None,
            file_size=1024,  # 1 KiB
            speed_mbps=1000.0,
            speed_unit="mbps",
            base_timeout_sec=0.0,
            overhead_percent=10.0,
            safety_multiplier=1.5,
            protocol_overhead_sec=0.0,
            min_timeout_sec=12.0,
            max_timeout_sec=3600.0,
        )
        assert result == 12

    def test_transfer_timeout_max_clamp(self):
        """Clamp to maximum timeout when total is too large."""
        result = network.transfer_timeout(
            file_path=None,
            file_size=10 * 1024**3,  # 10 GiB
            speed_mbps=1.0,
            speed_unit="mbps",
            base_timeout_sec=1.0,
            overhead_percent=30.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=10.0,
            max_timeout_sec=100.0,
        )
        assert result == 100

    def test_transfer_timeout_invalid_overhead(self):
        """Validate unreasonable overhead percent."""
        with pytest.raises(ValueError, match=r"(?i).*unreasonably high.*"):
            network.transfer_timeout(
                file_path=None,
                file_size=1024 * 1024,
                speed_mbps=100.0,
                speed_unit="mbps",
                base_timeout_sec=1.0,
                overhead_percent=150.0,  # invalid
                safety_multiplier=2.0,
                protocol_overhead_sec=1.0,
                min_timeout_sec=5.0,
                max_timeout_sec=300.0,
            )

    @pytest.mark.parametrize(
        "base, max_retries, backoff_multiplier, initial_backoff, expected",
        [
            pytest.param(10, 3, 2.0, 1.5, 51, id="geom_series_multiplier_2"),
            pytest.param(17, 3, 1.0, 2.0, 74, id="linear_multiplier_1"),
        ],
    )
    def test_transfer_timeout_retry(
        self,
        monkeypatch,
        base,
        max_retries,
        backoff_multiplier,
        initial_backoff,
        expected,
    ):
        """Compute retry total timeout with backoff."""

        # Stub transfer_timeout to return fixed base
        def fake_transfer_timeout(**kwargs):
            # Ensure kwargs are explicitly passed (sanity check)
            assert "file_size" in kwargs
            assert "speed_mbps" in kwargs
            assert "speed_unit" in kwargs
            return base

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        result = network.transfer_timeout_retry(
            file_path=None,
            file_size=5 * 1024 * 1024,
            max_retries=max_retries,
            backoff_multiplier=backoff_multiplier,
            initial_backoff_sec=initial_backoff,
            speed_mbps=50.0,
            speed_unit="mbps",
            base_timeout_sec=3.0,
            overhead_percent=15.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=2.0,
            min_timeout_sec=10.0,
            max_timeout_sec=1000.0,
        )
        assert result == expected

    def test_batch_timeout_sequential(self, monkeypatch):
        """Compute batch timeout for sequential transfers."""
        # Ensure speed conversion is controlled
        monkeypatch.setattr(network, "_speed_to_mbps", lambda v, u: 123.0)

        calls = []

        def fake_transfer_timeout(**kwargs):
            calls.append(kwargs)
            # Return deterministic "timeout" per file: size in MiB
            size = kwargs.get("file_size")
            assert kwargs["speed_mbps"] == 123.0
            assert kwargs["speed_unit"] == network.TransferSpeedUnit.MBPS
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
            speed_mbps=25.0,
            speed_unit="MBps",
            per_file_overhead_sec=1.5,
            base_timeout_sec=1.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=5.0,
            max_timeout_sec=1000.0,
        )
        # Individual timeouts: 1, 2, 1 -> sum=4; per-file overhead=3*1.5=4.5; total=8.5 -> ceil=9
        assert result == 10
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
            assert kwargs["speed_mbps"] == 200.0
            return size_to_timeout[kwargs["file_size"]]

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        files = [10 * 1024 * 1024, 20 * 1024 * 1024, 5 * 1024 * 1024]
        res = network.batch_timeout(
            files=files,
            parallel=True,
            max_parallel=2,
            speed_mbps=100.0,
            speed_unit="mbps",
            per_file_overhead_sec=0.5,
            base_timeout_sec=2.0,
            overhead_percent=15.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=10.0,
            max_timeout_sec=500.0,
        )
        # sharing_factor = min(3,2)=2 -> adjusted [20,40,10]; max=40; overhead=3*0.5=1.5 -> total=41.5 -> ceil=42
        assert res == 42

    def test_chunk_timeout_forward(self, monkeypatch):
        """Forward chunk parameters to transfer timeout."""
        captured = {}

        def fake_transfer_timeout(**kwargs):
            captured.update(kwargs)
            return 33

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        res = network.chunk_timeout(
            chunk_size=8 * 1024 * 1024,
            speed_mbps=75.0,
            speed_unit="mbps",
            base_timeout_sec=1.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=5.0,
            max_timeout_sec=600.0,
        )
        assert res == 33
        assert captured["file_size"] == 8 * 1024 * 1024
        assert captured["speed_mbps"] == 75.0
        assert captured["speed_unit"] == "mbps"

    @pytest.mark.parametrize(
        "unit, expected",
        [
            pytest.param("seconds", 9.6, id="seconds"),
            pytest.param("minutes", 9.6 / 60.0, id="minutes"),
            pytest.param("hours", 9.6 / 3600.0, id="hours"),
        ],
    )
    def test_transfer_time_units(self, unit, expected):
        """Compute transfer time across units."""
        # 100 MiB at 100 Mbps with 20% overhead:
        # size_mbits=800, base=8s, overhead=9.6s
        result = network.transfer_time(
            file_path=None,
            file_size=100 * 1024 * 1024,
            speed_mbps=100.0,
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
                speed_mbps=10.0,
                speed_unit="mbps",
                overhead_percent=10.0,
                unit="days",
            )

    def test_transfer_estimates_format(self, monkeypatch):
        """Format estimates and include computed fields."""
        # Control conversions and sub-computations
        monkeypatch.setattr(network, "_speed_to_mbps", lambda v, u: 80.0)
        monkeypatch.setattr(network, "transfer_timeout", lambda **kw: 42)
        monkeypatch.setattr(network, "transfer_time", lambda **kw: 30.0)

        est = network.transfer_estimates(
            file_path=None,
            file_size=2048,  # 2 KiB
            speed_mbps=10.0,
            speed_unit="MBps",
            base_timeout_sec=2.0,
            overhead_percent=10.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=5.0,
            max_timeout_sec=1000.0,
        )
        assert est["file_size_bytes"] == 2048
        assert est["file_size"] == "2.0 KB"
        assert est["speed_mbps"] == 80.0
        assert est["time_sec"] == 30.0
        assert est["time"] == "30.0 seconds"
        assert est["timeout_sec"] == 42
        assert est["timeout"] == "42.0 seconds"

    def test_transfer_params_unknown(self):
        """Validate unknown transfer type error."""
        with pytest.raises(ValueError, match=r"(?i).*unknown transfer type.*"):
            network.transfer_params("not_a_type")

    def test_transfer_type_merge_overrides(self, monkeypatch):
        """Merge transfer type params and apply overrides."""

        # Mock transfer_params to provide defaults
        def fake_transfer_params(t):
            assert t == "api_upload"
            return {
                "speed_mbps": 20.0,
                "speed_unit": "mbps",
                "base_timeout_sec": 3.0,
                "overhead_percent": 10.0,
                "safety_multiplier": 1.5,
                "protocol_overhead_sec": 2.0,
                "min_timeout_sec": 5.0,
                "max_timeout_sec": 50.0,
            }

        captured = {}

        def fake_transfer_timeout(**kwargs):
            captured.update(kwargs)
            return 77

        monkeypatch.setattr(network, "transfer_params", fake_transfer_params)
        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        res = network.transfer_type_timeout(
            transfer_type="api_upload",
            file_path=None,
            file_size=50 * 1024 * 1024,
            speed_mbps=30.0,  # override default 20.0
            speed_unit="mbps",
            base_timeout_sec=4.0,  # override default 3.0
            overhead_percent=12.0,  # override default 10.0
            safety_multiplier=2.0,  # override default 1.5
            protocol_overhead_sec=3.0,  # override default 2.0
            min_timeout_sec=6.0,  # override default 5.0
            max_timeout_sec=60.0,  # override default 50.0
        )

        assert res == 77
        # Verify merged and overridden params passed to transfer_timeout
        assert captured["file_path"] is None
        assert captured["file_size"] == 50 * 1024 * 1024
        assert captured["speed_mbps"] == 30.0
        assert captured["base_timeout_sec"] == 4.0
        assert captured["overhead_percent"] == 12.0
        assert captured["safety_multiplier"] == 2.0
        assert captured["protocol_overhead_sec"] == 3.0
        assert captured["min_timeout_sec"] == 6.0
        assert captured["max_timeout_sec"] == 60.0

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

    def test_transfer_timeout_calc(self):
        """Compute timeout with explicit parameters."""
        # Given values that produce a mid-range timeout
        file_size = 100 * 1024 * 1024  # 100 MiB
        speed_mbps = 50.0
        speed_unit = "mbps"
        base_timeout_sec = 5.0
        overhead_percent = 20.0
        safety_multiplier = 2.0
        protocol_overhead_sec = 2.0
        min_timeout_sec = 10.0
        max_timeout_sec = 3600.0

        result = network.transfer_timeout(
            file_path=None,
            file_size=file_size,
            speed_mbps=speed_mbps,
            speed_unit=speed_unit,
            base_timeout_sec=base_timeout_sec,
            overhead_percent=overhead_percent,
            safety_multiplier=safety_multiplier,
            protocol_overhead_sec=protocol_overhead_sec,
            min_timeout_sec=min_timeout_sec,
            max_timeout_sec=max_timeout_sec,
        )
        # Manual calc:
        # size_mbits = 100 MiB * 8 = 800 Mbits
        # transfer_time = 800 / 50 = 16
        # overhead -> 16 * 1.2 = 19.2
        # safety -> 19.2 * 2 = 38.4
        # total = 5 + 2 + 38.4 = 45.4 -> ceil = 46
        assert result == 46

    def test_transfer_timeout_min_clamp(self):
        """Clamp to minimum timeout when total is too small."""
        result = network.transfer_timeout(
            file_path=None,
            file_size=1024,  # 1 KiB
            speed_mbps=1000.0,
            speed_unit="mbps",
            base_timeout_sec=0.0,
            overhead_percent=10.0,
            safety_multiplier=1.5,
            protocol_overhead_sec=0.0,
            min_timeout_sec=12.0,
            max_timeout_sec=3600.0,
        )
        assert result == 12

    def test_transfer_timeout_max_clamp(self):
        """Clamp to maximum timeout when total is too large."""
        result = network.transfer_timeout(
            file_path=None,
            file_size=10 * 1024**3,  # 10 GiB
            speed_mbps=1.0,
            speed_unit="mbps",
            base_timeout_sec=1.0,
            overhead_percent=30.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=1.0,
            min_timeout_sec=10.0,
            max_timeout_sec=100.0,
        )
        assert result == 100

    def test_transfer_timeout_invalid_overhead(self):
        """Validate unreasonable overhead percent."""
        with pytest.raises(ValueError, match=r"(?i).*unreasonably high.*"):
            network.transfer_timeout(
                file_path=None,
                file_size=1024 * 1024,
                speed_mbps=100.0,
                speed_unit="mbps",
                base_timeout_sec=1.0,
                overhead_percent=150.0,  # invalid
                safety_multiplier=2.0,
                protocol_overhead_sec=1.0,
                min_timeout_sec=5.0,
                max_timeout_sec=300.0,
            )

    @pytest.mark.parametrize(
        "base, max_retries, backoff_multiplier, initial_backoff, expected",
        [
            pytest.param(10, 3, 2.0, 1.5, 51, id="geom_series_multiplier_2"),
            pytest.param(17, 3, 1.0, 2.0, 74, id="linear_multiplier_1"),
        ],
    )
    def test_transfer_timeout_retry(
        self,
        monkeypatch,
        base,
        max_retries,
        backoff_multiplier,
        initial_backoff,
        expected,
    ):
        """Compute retry total timeout with backoff."""

        # Stub transfer_timeout to return fixed base
        def fake_transfer_timeout(**kwargs):
            # Ensure kwargs are explicitly passed (sanity check)
            assert "file_size" in kwargs
            assert "speed_mbps" in kwargs
            assert "speed_unit" in kwargs
            return base

        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        result = network.transfer_timeout_retry(
            file_path=None,
            file_size=5 * 1024 * 1024,
            max_retries=max_retries,
            backoff_multiplier=backoff_multiplier,
            initial_backoff_sec=initial_backoff,
            speed_mbps=50.0,
            speed_unit="mbps",
            base_timeout_sec=3.0,
            overhead_percent=15.0,
            safety_multiplier=2.0,
            protocol_overhead_sec=2.0,
            min_timeout_sec=10.0,
            max_timeout_sec=1000.0,
        )
        assert result == expected

    def test_transfer_type_merge_overrides(self, monkeypatch):
        """Merge transfer type params and apply overrides."""

        # Mock transfer_params to provide defaults
        def fake_transfer_params(t):
            assert t == "api_upload"
            return {
                "speed_mbps": 20.0,
                "speed_unit": "mbps",
                "base_timeout_sec": 3.0,
                "overhead_percent": 10.0,
                "safety_multiplier": 1.5,
                "protocol_overhead_sec": 2.0,
                "min_timeout_sec": 5.0,
                "max_timeout_sec": 50.0,
            }

        captured = {}

        def fake_transfer_timeout(**kwargs):
            captured.update(kwargs)
            return 77

        monkeypatch.setattr(network, "transfer_params", fake_transfer_params)
        monkeypatch.setattr(network, "transfer_timeout", fake_transfer_timeout)

        res = network.transfer_type_timeout(
            transfer_type="api_upload",
            file_path=None,
            file_size=50 * 1024 * 1024,
            speed_mbps=30.0,  # override default 20.0
            speed_unit="mbps",
            base_timeout_sec=4.0,  # override default 3.0
            overhead_percent=12.0,  # override default 10.0
            safety_multiplier=2.0,  # override default 1.5
            protocol_overhead_sec=3.0,  # override default 2.0
            min_timeout_sec=6.0,  # override default 5.0
            max_timeout_sec=60.0,  # override default 50.0
        )

        assert res == 77
        # Verify merged and overridden params passed to transfer_timeout
        assert captured["file_path"] is None
        assert captured["file_size"] == 50 * 1024 * 1024
        assert captured["speed_mbps"] == 30.0
        assert captured["base_timeout_sec"] == 4.0
        assert captured["overhead_percent"] == 12.0
        assert captured["safety_multiplier"] == 2.0
        assert captured["protocol_overhead_sec"] == 3.0
        assert captured["min_timeout_sec"] == 6.0
        assert captured["max_timeout_sec"] == 60.0
