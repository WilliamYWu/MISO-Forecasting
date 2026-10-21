import argparse
from datetime import date, datetime, time, timezone
from io import BytesIO
from pathlib import Path

import polars as pl
import requests


BASE_URL = "https://docs.misoenergy.org/marketreports"


def fetch_miso_data(
        market_date: str,
        node: str = "MINN.HUB",
        ) -> pl.DataFrame:
    """Download one day's MISO report and return hourly prices for a node."""

    day = date.fromisoformat(market_date)
    date_str = day.strftime("%Y%m%d")
    url = f"{BASE_URL}/{date_str}_da_expost_lmp.csv"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw = pl.read_csv(
        BytesIO(response.content),
        skip_rows=4,
    )

    raw = raw.filter(pl.col("Node") == node)

    if raw.is_empty():
        raise ValueError(f"Node {node} not found in report for {market_date}")

    hour_cols = [f"HE {hour}" for hour in range(1, 25)]

    long = (
        raw
        .unpivot(
            on=hour_cols,
            index=["Node", "Type", "Value"],
            variable_name="HE",
            value_name="Price",
        )
        .with_columns(
            pl.col("HE")
            .str.extract(r"(\d+)$", 1)
            .cast(pl.Int8)
            .alias("hour_ending")
        )
    )

    hourly = long.pivot(
        on="Value",
        index=["Node", "Type", "hour_ending"],
        values="Price",
    )

    # Preserve the original script's fixed-EST hour interpretation.
    # HE 1 starts at midnight EST, corresponding to 05:00 UTC.
    base_utc = datetime.combine(
        day, time(5, 0), tzinfo=timezone.utc
    )

    hourly = (
        hourly
        .with_columns(
            pl.lit(day).alias("market_date"),
            (
                pl.lit(base_utc)
                + pl.duration(hours=pl.col("hour_ending") - 1)
            ).alias("interval_start_utc"),
            (
                pl.col("LMP")
                - pl.col("MCC")
                - pl.col("MLC")
            ).alias("energy"),
        )
        .sort("hour_ending")
    )

    return hourly


def ingest_miso(
        market_date: str,
        output_path: str | Path,
        node: str = "MINN.HUB",
        ) -> pl.DataFrame:
    """Download and transform MISO LMPs, then write processed Parquet."""

    df = fetch_miso_data(
        market_date=market_date,
        node=node,
    )

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.write_parquet(
        output,
        compression="zstd",
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hourly MISO day-ahead LMP data"
    )

    parser.add_argument("--market_date", required=True)
    parser.add_argument("--node", default="MINN.HUB")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    df = ingest_miso(
        market_date=args.market_date,
        node=args.node,
        output_path=args.output,
    )

    print(f"Saved {df.height} observations")
    print(f"Columns: {df.columns}")
    print(df.head(5))


if __name__ == "__main__":
    main()
