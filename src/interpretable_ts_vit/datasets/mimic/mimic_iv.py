"""Shared MIMIC-IV table-reading and value-standardization utilities."""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


TEMPERATURE_FAHRENHEIT_ITEMIDS = {223761}
TEMPERATURE_CELSIUS_ITEMIDS = {223762}


class _MIMICSource:
    """Read MIMIC-IV CSV.GZ tables from either a zip archive or directory."""

    def __init__(self, path: str | Path, extraction_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == ".zip"
        self.extraction_dir = Path(extraction_dir) if extraction_dir is not None else None
        self._progress: dict[str, tuple[int | None, int | None]] = {}

    def read_table(self, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
        logger.info("Reading MIMIC table %s", table)
        if self.is_zip:
            extracted = self._extract_table(table)
            if extracted is not None:
                frame = pd.read_csv(extracted, usecols=usecols)
                logger.info("Read MIMIC table %s from extracted file %s: rows=%d", table, extracted, len(frame))
                return frame
            with ZipFile(self.path) as zf:
                entry = self._zip_entry(zf, table)
                with gzip.GzipFile(fileobj=zf.open(entry)) as fh:
                    frame = pd.read_csv(fh, usecols=usecols)
                    logger.info("Read MIMIC table %s from zip entry %s: rows=%d", table, entry, len(frame))
                    return frame
        path = self._file_path(table)
        frame = pd.read_csv(path, usecols=usecols)
        logger.info("Read MIMIC table %s from %s: rows=%d", table, path, len(frame))
        return frame

    def iter_table(
        self,
        table: str,
        usecols: list[str] | None,
        chunksize: int,
    ) -> Iterator[pd.DataFrame]:
        logger.info("Iterating MIMIC table %s with chunksize=%d", table, chunksize)
        if self.is_zip:
            extracted = self._extract_table(table)
            if extracted is not None:
                yield from self._iter_csv_gz_with_progress(extracted, table, usecols, chunksize)
                return
            with ZipFile(self.path) as zf:
                entry = self._zip_entry(zf, table)
                info = zf.getinfo(entry)
                raw = _TrackingReader(zf.open(entry), table, info.file_size, self._progress)
                with gzip.GzipFile(fileobj=raw) as fh:
                    yield from pd.read_csv(fh, usecols=usecols, chunksize=chunksize)
            return
        path = self._file_path(table)
        if path.suffix == ".gz":
            yield from self._iter_csv_gz_with_progress(path, table, usecols, chunksize)
            return
        self._progress[table] = (None, None)
        yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize)

    def progress(self, table: str) -> tuple[int | None, int | None]:
        return self._progress.get(table, (None, None))

    def _iter_csv_gz_with_progress(
        self,
        path: Path,
        table: str,
        usecols: list[str] | None,
        chunksize: int,
    ) -> Iterator[pd.DataFrame]:
        total = path.stat().st_size
        logger.info("Reading compressed MIMIC table %s from %s: bytes=%d", table, path, total)
        with path.open("rb") as raw_file:
            raw = _TrackingReader(raw_file, table, total, self._progress)
            with gzip.GzipFile(fileobj=raw) as fh:
                yield from pd.read_csv(fh, usecols=usecols, chunksize=chunksize)

    def _extract_table(self, table: str) -> Path | None:
        if self.extraction_dir is None:
            return None
        output = self.extraction_dir / table
        if output.exists():
            logger.info("Using extracted MIMIC table %s at %s", table, output)
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(self.path) as zf:
            entry = self._zip_entry(zf, table)
            logger.info("Extracting MIMIC zip entry %s to %s", entry, output)
            with zf.open(entry) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
        logger.info("Extracted MIMIC zip entry %s to %s", table, output)
        return output

    def _zip_entry(self, zf: ZipFile, table: str) -> str:
        matches = [name for name in zf.namelist() if name.endswith(table)]
        if not matches:
            raise FileNotFoundError(f"Could not find {table} inside {self.path}")
        return matches[0]

    def _file_path(self, table: str) -> Path:
        filename = Path(table).name
        matches = [path for path in self.path.rglob(filename) if path.as_posix().endswith(table)]
        if not matches:
            raise FileNotFoundError(f"Could not find {table} under {self.path}")
        return matches[0]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class _TrackingReader:
    """Track compressed bytes consumed while pandas reads through gzip."""

    def __init__(
        self,
        fileobj,
        table: str,
        total_bytes: int | None,
        progress: dict[str, tuple[int | None, int | None]],
    ) -> None:
        self.fileobj = fileobj
        self.table = table
        self.total_bytes = total_bytes
        self.progress = progress
        self.bytes_read = 0
        self.progress[self.table] = (0, self.total_bytes)

    def read(self, size: int = -1):
        data = self.fileobj.read(size)
        self.bytes_read += len(data)
        self.progress[self.table] = (self.bytes_read, self.total_bytes)
        return data

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self.fileobj.close()


def standardize_temperature_to_celsius(
    values: pd.Series,
    itemids: pd.Series | None = None,
    celsius_min: float = 25.0,
    celsius_max: float = 45.0,
) -> pd.Series:
    """Convert MIMIC temperature values to Celsius and remove implausible values."""
    standardized = values.astype(float).copy()
    if itemids is None:
        fahrenheit = standardized >= 70.0
    else:
        itemids = itemids.astype("int64")
        fahrenheit = itemids.isin(TEMPERATURE_FAHRENHEIT_ITEMIDS)
        known_temperature = itemids.isin(TEMPERATURE_FAHRENHEIT_ITEMIDS | TEMPERATURE_CELSIUS_ITEMIDS)
        standardized.loc[~known_temperature] = np.nan
    standardized.loc[fahrenheit] = (standardized.loc[fahrenheit] - 32.0) * 5.0 / 9.0
    plausible = standardized.between(celsius_min, celsius_max, inclusive="both")
    standardized.loc[~plausible] = np.nan
    return standardized
