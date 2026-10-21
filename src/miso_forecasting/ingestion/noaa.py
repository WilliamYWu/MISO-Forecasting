import argparse
from datetime import date
from io import BytesIO
from pathlib import Path

import polars as pl
import requests


BASE_URL = (
    "https://www.ncei.noaa.gov/oa/"
    "global-historical-climatology-network/hourly/access/by-year"
)


def fetch_noaa_data(
        start_date: str,
        end_date: str,
        station: str
        ) -> pl.DataFrame:

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")

    frames = []

    # NOAA publishes one station file per year, in Parquet and PSV formats.
    # Prefer Parquet, and use PSV if that file is unavailable.
    with requests.Session() as session:
        for year in range(start.year, end.year + 1):
            errors = []
            df = None

            for file_format in ("parquet", "psv"):
                filename = f"GHCNh_{station}_{year}.{file_format}"
                url = f"{BASE_URL}/{year}/{file_format}/{filename}"

                print(f"Downloading: {url}", flush=True)

                try:
                    response = session.get(
                        url,
                        timeout=(15, 120),
                    )
                    response.raise_for_status()

                except requests.RequestException as exc:
                    errors.append(f"{file_format}: {exc}")
                    continue

                # Do not try to read an HTML error page as a data file.
                if response.content.lstrip().lower().startswith(
                        (b"<!doctype html", b"<html")
                        ):
                    errors.append(f"{file_format}: received HTML instead of data")
                    continue

                try:
                    if file_format == "parquet":
                        df = pl.read_parquet(BytesIO(response.content))
                    else:
                        df = pl.read_csv(
                            BytesIO(response.content),
                            separator="|",
                            infer_schema=False,
                        )
                except Exception as exc:
                    errors.append(f"{file_format}: failed to parse: {exc}")
                    continue

                if "DATE" not in df.columns:
                    errors.append(
                        f"{file_format}: DATE column missing; columns={df.columns[:12]}"
                    )
                    df = None
                    continue

                print(f"Downloaded {df.height} rows for {year}", flush=True)
                break

            if df is None:
                raise RuntimeError(
                    f"Unable to download NOAA GHCNh station {station} "
                    f"for {year}.\n" + "\n".join(errors)
                )

            # DATE is UTC. Date bounds are inclusive, even for multi-year pulls.
            df = (
                df.with_columns(pl.all().cast(pl.String))
                .filter(
                    pl.col("DATE")
                    .str.slice(0, 10)
                    .is_between(pl.lit(start_date), pl.lit(end_date))
                )
            )

            frames.append(df)

    return pl.concat(frames, how="diagonal")


def to_raw_dataframe(
        records: pl.DataFrame,
        ) -> pl.DataFrame:

    if records.is_empty():
        raise ValueError("No NOAA records for the requested station/date range")

    required = {"DATE"}
    missing = required - set(records.columns)

    if missing:
        raise ValueError(f"Missing NOAA fields: {sorted(missing)}")

    df = records.with_columns(pl.all().cast(pl.String))

    return df.sort("DATE")


def ingest_noaa(
        start_date: str,
        end_date: str,
        output_path: str | Path,
        station: str
        ) -> pl.DataFrame:
    """Download NOAA station observations and save raw Parquet."""

    records = fetch_noaa_data(
        start_date=start_date,
        end_date=end_date,
        station=station,
    )

    df = to_raw_dataframe(records)

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        output,
        compression="zstd",
    )

    print(f"Saved {df.height} observations to {output}", flush=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hourly NOAA GHCNh weather observations"
    )

    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    df = ingest_noaa(
        start_date=args.start_date,
        end_date=args.end_date,
        station=args.station,
        output_path=args.output,
    )

    print(f"Columns: {df.columns}")
    print(df.head(5))


if __name__ == "__main__":
    main()
