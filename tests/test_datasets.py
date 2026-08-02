"""Tests for the dataset loader.

Almost all of these run **offline**. A test suite that needs the network is a
test suite that fails for reasons unrelated to the code, so the machinery --
caching, digest verification, parsing, error messages -- is exercised against
locally written bytes. Only two tests actually fetch, and they are marked
``network`` and skipped unless asked for.

The licensing invariant is checked too: this package is MIT and must never
acquire a GPL dataset by the back door.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest

import rcopula as rc
from rcopula.datasets import DatasetSpec, available, cache_dir, clear_cache, load

# A miniature USGS RDB file: comments, a header, the format-specifier row that
# has to be skipped, then data.
RDB = b"""# comment
# another comment
agency_cd\tsite_no\tpeak_dt\tpeak_va\tgage_ht
5s\t15s\t10d\t8s\t8s
USGS\t01646500\t1931-05-01\t45000\t9.10
USGS\t01646500\t1931-10-15\t51000\t9.80
USGS\t01646500\t1933-03-19\t62000\t10.55
USGS\t01646500\t1934-04-01\t\t7.00
"""

GHCN_ROWS = "\n".join(
    f"USW00023174,{date},{element},{value},,,W,"
    for date, element, value in [
        ("20200101", "TMAX", 210),
        ("20200101", "TMIN", 95),
        ("20200101", "PRCP", 0),
        ("20200102", "TMAX", 195),
        ("20200102", "TMIN", 88),
        ("20200102", "PRCP", 51),
        ("20200103", "TMAX", 230),
        ("20200103", "TMIN", 110),
        ("20200103", "PRCP", 0),
    ]
)


#: Three rows of each delimited source, enough to exercise the reader offline.
ABALONE = (
    b"M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15\n"
    b"F,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9\n"
    b"I,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7\n"
)

WINE = (
    b'"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";'
    b'"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";'
    b'"alcohol";"quality"\n'
    b"7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5\n"
    b"7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5\n"
)

AIRFOIL = (
    b"800\t0.0\t0.3048\t71.3\t0.00266337\t126.201\n"
    b"1000\t0.0\t0.3048\t71.3\t0.00266337\t125.201\n"
    b"1250\t0.0\t0.3048\t71.3\t0.00266337\t125.951\n"
)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("RCOPULA_DATA", str(tmp_path))
    return tmp_path


class TestTheRegistryIsWellFormed:
    def test_every_spec_is_complete(self) -> None:
        for name, spec in available().items():
            assert isinstance(spec, DatasetSpec)
            assert spec.name == name
            assert spec.url.startswith("https://")
            assert spec.description and spec.citation
            assert spec.columns

    def test_every_source_is_permissively_licensed(self) -> None:
        """The load-bearing constraint.

        R's copula ships its datasets under GPL-3, and this package is MIT.
        Vendoring one would be a licence conflict, so nothing is vendored and
        every registered source must be public domain or permissive. A future
        addition that is not will fail here rather than in a lawyer's office.
        """
        permissive = ("public domain", "cc0", "cc-by", "cc by", "mit", "odbl")
        for name, spec in available().items():
            assert any(token in spec.licence.lower() for token in permissive), (
                f"{name} is licensed {spec.licence!r}, which is not clearly "
                "permissive. Do not add it."
            )

    def test_no_data_file_is_committed(self) -> None:
        """The repository must contain links, not datasets."""
        root = Path(__file__).resolve().parent.parent
        offenders = [
            p
            for pattern in ("*.rda", "*.RData", "*.rds")
            for p in root.rglob(pattern)
            if ".git" not in p.parts
        ]
        assert not offenders, f"R data files committed: {offenders}"

    def test_available_returns_a_copy(self) -> None:
        first = available()
        first.pop(next(iter(first)))
        assert len(available()) > len(first)


class TestCaching:
    def test_the_cache_directory_is_configurable(self, isolated_cache: Path) -> None:
        assert cache_dir() == isolated_cache

    def test_a_cached_file_is_reused_without_the_network(self, isolated_cache: Path) -> None:
        """The whole point of caching: one fetch, then offline forever."""
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        frame = load("nwis_peaks", download=False)
        assert len(frame) == 3

    def test_missing_and_download_disabled_says_what_to_do(self, isolated_cache: Path) -> None:
        with pytest.raises(OSError, match="download=False"):
            load("nwis_peaks", download=False)

    def test_clear_cache_removes_files_and_counts_them(self, isolated_cache: Path) -> None:
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        (isolated_cache / "other.raw").write_bytes(b"x")
        assert clear_cache() == 2
        assert clear_cache() == 0

    def test_an_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(KeyError, match="available"):
            load("loss_alae")


class TestDigestVerification:
    def test_a_changed_upstream_file_is_refused(self, isolated_cache: Path, monkeypatch) -> None:
        """A silently changed source turns a reproducible result into an
        irreproducible one, so it must fail loudly rather than be used."""
        import rcopula.datasets as ds

        spec = available()["nwis_peaks"]
        pinned = DatasetSpec(**{**spec.__dict__, "sha256": hashlib.sha256(b"other").hexdigest()})
        monkeypatch.setitem(ds._REGISTRY, "nwis_peaks", pinned)
        with pytest.raises(OSError, match="does not match its recorded SHA-256"):
            ds._verify(RDB, pinned)

    def test_a_matching_digest_passes(self) -> None:
        import rcopula.datasets as ds

        spec = available()["nwis_peaks"]
        pinned = DatasetSpec(**{**spec.__dict__, "sha256": hashlib.sha256(RDB).hexdigest()})
        ds._verify(RDB, pinned)  # must not raise

    def test_no_digest_means_no_check(self) -> None:
        """Live query endpoints grow a row a year and cannot be pinned."""
        import rcopula.datasets as ds

        ds._verify(b"anything at all", available()["nwis_peaks"])


class TestParsing:
    def test_usgs_rdb(self, isolated_cache: Path) -> None:
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        frame = load("nwis_peaks", download=False)
        assert list(frame.columns) == ["water_year", "peak_flow", "gauge_height"]
        assert len(frame) == 3  # the row with a missing peak is dropped
        assert frame["peak_flow"].tolist() == [45000, 51000, 62000]

    def test_the_water_year_rolls_over_in_october(self, isolated_cache: Path) -> None:
        """A water year runs October to September. Getting this wrong shifts
        every autumn flood into the previous year."""
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        frame = load("nwis_peaks", download=False)
        assert frame["water_year"].tolist() == [1931, 1932, 1933]

    def test_ghcn_csv(self, isolated_cache: Path) -> None:
        (isolated_cache / "ghcn_temperature.raw").write_bytes(gzip.compress(GHCN_ROWS.encode()))
        frame = load("ghcn_temperature", download=False)
        assert list(frame.columns) == ["date", "tmax", "tmin", "prcp"]
        assert len(frame) == 3
        # GHCN stores tenths, so the loader must divide by ten.
        assert frame["tmax"].tolist() == [21.0, 19.5, 23.0]
        assert frame["prcp"].tolist() == [0.0, 5.1, 0.0]

    def test_abalone_has_no_header_to_discard(self, isolated_cache: Path) -> None:
        (isolated_cache / "uci_abalone.raw").write_bytes(ABALONE)
        frame = load("uci_abalone", download=False)
        assert list(frame.columns) == list(available()["uci_abalone"].columns)
        assert len(frame) == 3
        assert frame["rings"].tolist() == [15, 9, 7]
        assert frame["sex"].tolist() == ["M", "F", "I"]

    def test_wine_is_semicolon_separated_and_its_header_is_replaced(
        self, isolated_cache: Path
    ) -> None:
        # Upstream writes 'fixed acidity' with a space and 'pH' with a capital;
        # the loader replaces the header so callers get stable snake_case.
        (isolated_cache / "uci_wine_quality_red.raw").write_bytes(WINE)
        frame = load("uci_wine_quality_red", download=False)
        assert list(frame.columns) == list(available()["uci_wine_quality_red"].columns)
        assert "fixed_acidity" in frame.columns
        assert "ph" in frame.columns
        assert len(frame) == 2
        assert frame["alcohol"].tolist() == [9.4, 9.8]

    def test_airfoil_is_whitespace_separated(self, isolated_cache: Path) -> None:
        (isolated_cache / "uci_airfoil.raw").write_bytes(AIRFOIL)
        frame = load("uci_airfoil", download=False)
        assert list(frame.columns) == list(available()["uci_airfoil"].columns)
        assert frame["frequency"].tolist() == [800, 1000, 1250]
        assert frame["sound_pressure_level"].tolist() == [126.201, 125.201, 125.951]

    def test_a_wrong_column_count_is_reported_rather_than_mislabelled(
        self, isolated_cache: Path
    ) -> None:
        # Silently naming seven columns with nine names would corrupt every
        # downstream result, so it raises instead.
        (isolated_cache / "uci_abalone.raw").write_bytes(b"M,0.455,0.365\n")
        with pytest.raises(OSError, match="expected 9 columns"):
            load("uci_abalone", download=False)

    def test_a_truncated_download_is_reported(self, isolated_cache: Path) -> None:
        (isolated_cache / "nwis_peaks.raw").write_bytes(b"# only comments\n")
        with pytest.raises(OSError, match="no data rows"):
            load("nwis_peaks", download=False)

    def test_provenance_travels_with_the_frame(self, isolated_cache: Path) -> None:
        """So a result can be cited without going back to the registry."""
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        frame = load("nwis_peaks", download=False)
        assert frame.attrs["dataset"] == "nwis_peaks"
        assert "public domain" in frame.attrs["licence"]
        assert "U.S. Geological Survey" in frame.attrs["citation"]

    def test_the_result_is_usable_as_a_copula_input(self, isolated_cache: Path) -> None:
        (isolated_cache / "nwis_peaks.raw").write_bytes(RDB)
        frame = load("nwis_peaks", download=False)
        u = rc.pseudo_obs(frame[["peak_flow", "gauge_height"]])
        assert u.shape == (3, 2)
        assert isinstance(frame, pd.DataFrame)


@pytest.mark.network
class TestAgainstTheRealSources:
    """Skipped unless asked for: ``pytest -m network``."""

    @pytest.mark.parametrize("name", ["uci_abalone", "uci_wine_quality_red", "uci_airfoil"])
    def test_the_pinned_sources_still_match_their_digests(
        self, name: str, isolated_cache: Path
    ) -> None:
        """These three are byte-stable, so the digest is a real check.

        If this fails, the upstream file changed and every result computed from
        it stops being reproducible -- which is exactly what the digest is for.
        """
        frame = load(name)
        assert list(frame.columns) == list(available()[name].columns)
        assert len(frame) > 1000
        assert "CC BY 4.0" in frame.attrs["licence"]

    def test_the_flood_series_downloads_and_parses(self, isolated_cache: Path) -> None:
        frame = load("nwis_peaks")
        assert len(frame) > 50
        assert frame["peak_flow"].min() > 0
        tau = rc.cor_kendall(frame[["peak_flow", "gauge_height"]])[0, 1]
        assert 0.3 < tau < 0.95  # peaks and stage are strongly but not perfectly tied

    def test_the_second_call_uses_the_cache(self, isolated_cache: Path) -> None:
        load("nwis_peaks")
        assert (isolated_cache / "nwis_peaks.raw").exists()
        assert len(load("nwis_peaks", download=False)) > 50
