"""Portable acquisition-file conversion; no raw data are bundled."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import MINUTE, digest, resolve, write_json


def _new_panel(directory, times, ids, nominal=True):
    directory.mkdir(parents=True, exist_ok=False)
    shape = (len(times), len(ids))
    speed = np.lib.format.open_memmap(
        directory / "speed.npy", mode="w+", dtype="float32", shape=shape
    )
    speed[:] = np.nan
    stamp = None
    if nominal:
        stamp = np.lib.format.open_memmap(
            directory / "nominal_ns.npy", mode="w+", dtype="int64", shape=shape
        )
        stamp[:] = -1
    np.save(directory / "timestamps_ns.npy", times)
    np.save(directory / "segment_ids.npy", ids)
    return speed, stamp


def ingest_chicago(config, external=False):
    name = "chicago_external" if external else "chicago"
    raw = resolve(config, "raw") / name
    files = sorted(raw.rglob("*.parquet")) + sorted(raw.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No source Parquet/CSV under {raw}; see data/README.md")
    ids = np.asarray(
        __import__("json").loads(
            (config["_root"] / "configs/chicago_segment_ids.json").read_text()
        ),
        dtype="int64",
    )
    if external:
        start = (
            pd.Timestamp(config["external"]["start"], tz="America/Chicago").value - 1440 * MINUTE
        )
        end = pd.Timestamp(config["external"]["end"], tz="America/Chicago").value
    else:
        start = pd.Timestamp("2022-09-01", tz="America/Chicago").value
        end = 1693805400000000000
    times = np.arange(start, end, 10 * MINUTE, dtype="int64")
    directory = resolve(config, "processed") / name
    speed, nominal = _new_panel(directory, times, ids)
    sources = []
    for file in files:
        frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
        canonical = "source_timestamp_raw" in frame
        raw_time = pd.DatetimeIndex(
            pd.to_datetime(frame["source_timestamp_raw"] if canonical else frame["time"])
        )
        if canonical:
            local_floor = pd.DatetimeIndex(pd.to_datetime(frame.network_snapshot_time))
            utc_floor = pd.DatetimeIndex(pd.to_datetime(frame.forecast_origin_utc, utc=True))
            offset = raw_time.as_unit("ns").asi8 - local_floor.as_unit("ns").asi8
            assert np.all((offset >= 0) & (offset < 10 * MINUTE))
            raw_ns = utc_floor.as_unit("ns").asi8 + offset
            known = np.ones(len(frame), bool)
        else:
            dates = raw_time.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
            raw_ns, known = dates.as_unit("ns").asi8, dates.notna()
        segment = pd.to_numeric(frame.segment_id, errors="coerce")
        grid = ((raw_ns + 10 * MINUTE - 1) // (10 * MINUTE)) * (10 * MINUTE)
        keep = known & np.isin(segment, ids) & (grid >= start) & (grid < end)
        frame = frame.loc[keep].copy()
        frame["nominal"] = raw_ns[keep]
        frame["ti"] = (grid[keep] - start) // (10 * MINUTE)
        frame["ni"] = np.searchsorted(ids, segment[keep])
        frame = frame.sort_values("nominal", kind="stable").drop_duplicates(
            ["ti", "ni"], keep="last"
        )
        ti, ni = frame.ti.to_numpy(), frame.ni.to_numpy()
        newer = frame.nominal.to_numpy() > nominal[ti, ni]
        values = pd.to_numeric(frame.speed_mph if canonical else frame.speed).to_numpy(
            dtype="float32"
        )
        values[values < 0] = np.nan
        speed[ti[newer], ni[newer]] = values[newer]
        nominal[ti[newer], ni[newer]] = frame.nominal.to_numpy()[newer]
        sources.append(
            dict(
                file=file.relative_to(raw).as_posix(),
                bytes=file.stat().st_size,
                sha256=digest(file),
            )
        )
        print(f"{name}: {file.name}", flush=True)
    speed.flush()
    nominal.flush()
    write_json(directory / "source_manifest.json", sources)


def ingest_city(config):
    source = resolve(config, "raw") / "city_m/city-traffic-M-raw-speed.parquet"
    if not source.exists():
        raise FileNotFoundError(f"{source}; see data/README.md")
    parquet = pq.ParquetFile(source)
    assert parquet.metadata.num_rows == 35449 and parquet.metadata.num_columns == 53531
    times = (
        pq.read_table(source, columns=["timestamp"])
        .column(0)
        .to_numpy()
        .astype("datetime64[ns]")
        .astype("int64")
    )
    assert np.all(np.diff(times) == 5 * MINUTE)
    ids = np.arange(53530)
    directory = resolve(config, "processed") / "city_m"
    speed, _ = _new_panel(directory, times, ids, nominal=False)
    for first in range(0, len(ids), 128):
        names = [f"node_{j}" for j in range(first, min(first + 128, len(ids)))]
        table = pq.read_table(source, columns=names)
        speed[:, first : first + len(names)] = np.column_stack(
            [table.column(i).to_numpy() for i in range(len(names))]
        ).astype("float32")
        print(f"City-M columns {first + len(names)}/{len(ids)}", flush=True)
    speed.flush()
    write_json(
        directory / "source_manifest.json",
        dict(file=source.name, bytes=source.stat().st_size, sha256=digest(source)),
    )


def run(config, dataset):
    if dataset == "City-M":
        ingest_city(config)
    else:
        ingest_chicago(config, external=dataset == "Chicago-external")
