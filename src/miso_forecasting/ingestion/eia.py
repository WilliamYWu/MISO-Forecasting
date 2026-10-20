import argparse                 # Command Line parser
import os                       # Read ENV
from pathlib import Path        

import polars as pl
import requests
from dotenv import load_dotenv  # Load ENV from .env

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

PAGE_SIZE = 5000

def fetch_eia_data(
        start_date: str, 
        end_date: str,
        respondent: str = "MISO",
        series_type: str = "D",
        api_key: str | None = None, # Union type between str and None, If None we retrieve the credential from .env
        ) -> list[dict]: # We expect a list of dictionaries from the API.

    load_dotenv()  # Load ENV from .env
    key = api_key or os.getenv("EIA_API_KEY")

    if not key:
        raise ValueError("EIA_API_KEY not set.")

    # YYYY-MM-DDTHH format makes it so that we can compare as chronological order.
    # But I don't think we check the dates are actual dates. 
    if start_date > end_date:
        raise ValueError("start_date must be before end_date.")

    params = {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": series_type,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": PAGE_SIZE,
    }
    
    records = []

    # A Session is an object that manages HTTP requests and connection pooling.
    # It allows requests to reuse underlying TCP connections. 
    # Useful when multiple calls target the same API server.
    with requests.Session() as session:
        while True:
            response = session.get(
                BASE_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()
            payload = response.json()

            if "error" in payload:
                raise RuntimeError(
                    f"EIA API error: {payload['error']}"
                )

            result = payload["response"]
            batch = result["data"]
            total = int(result["total"])

            if not batch and len(records) < total:
                raise RuntimeError(
                    "EIA returned an incomplete response"
                )

            # append would create nested lists, while extend adds to existing list. 
            records.extend(batch)

            if len(records) >= total:
                if len(records) != total:
                    raise RuntimeError(
                        "Retrieved count exceeds API total"
                    )
                break

            params["offset"] += len(batch)

    return records

    
def to_raw_dataframe(
        records: list[dict],
        ) -> pl.DataFrame:

    if not records:
        raise ValueError("No EIA records returned")

    df = pl.from_dicts(records,infer_schema_length=None)

    required = {"period", "respondent", "type", "value"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing API fields: {sorted(missing)}"
        )

    # Replace all columns with strings to avoid type issues. 
    # with_columns returns a new Dataframe with the selected columns replaced.
    df = df.with_columns(pl.all().cast(pl.String))

    return df.sort("period")


def ingest_eia(
        start_date: str,
        end_date: str,
        output_path: str | Path, # str path name or a Path() object.
        respondent: str = "MISO",
        series_type: str = "D",
        api_key: str | None = None
        ) -> pl.DataFrame:
    """
    Download an EIA series and write raw Parquet. 
    Wrapper around fetch_eia_data and to_raw_dataframe.
    """
    records = fetch_eia_data(
        start_date=start_date,
        end_date=end_date,
        respondent=respondent,
        series_type=series_type,
        api_key=api_key,
    )

    df = to_raw_dataframe(records)

    output = Path(output_path)

    # Will create the parent directory if they dont exist and will not raise error if already exists. 
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
    parser = argparse.ArgumentParser(description="Download hourly EIA-930 data")

    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--respondent", default="MISO")
    parser.add_argument("--series-type", default="D")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    df = ingest_eia(
        start_date=args.start_date,
        end_date=args.end_date,
        respondent=args.respondent,
        series_type=args.series_type,
        output_path=args.output,
    )

    print(f"Saved {df.height} observations")
    print(f"Columns: {df.columns}")
    print(df.head(5))


if __name__ == "__main__":
    main()

