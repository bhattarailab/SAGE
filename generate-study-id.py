import argparse
from pathlib import Path
import pandas as pd
from uuid import uuid4

VALID_TIMESTAMP_PATTERN = (
    r"^(0[1-9]|[12]\d|3[01])\."
    r"(0[1-9]|1[0-2])\."
    r"\d{4}\s+"
    r"([01]\d|2[0-3])[:.]"
    r"([0-5]\d)[:.]"
    r"([0-5]\d)$"
)


def main(csv_path: Path, output_csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    # normalize the timestamp and if not possible raise an error with information
    df["timestamp"] = df["timestamp"].astype(str).str.strip().str.replace("/", ".", regex=False)
    valid_mask = df["timestamp"].astype(str).str.match(VALID_TIMESTAMP_PATTERN)

    if not valid_mask.all():
      bad_indices = df.index[~valid_mask].tolist()
      raise ValueError(f"Invalid timestamp format at row index: {bad_indices}. ")
    
    # change the timestamp from str to datetime for comparison
    date_part = df["timestamp"].str.split(" ", n=1).str[0]
    time_part = df["timestamp"].str.split(" ", n=1).str[1].str.replace(".", ":", regex=False)

    parsed_timestamp = pd.to_datetime(
        date_part + " " + time_part,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )
    df["timestamp"] = parsed_timestamp

    # sort the patient by patient id, then by filename as they are already sorted in hexadecimal format
    df["filename_sort_key"] = df["filename"].map(
        lambda x: int(Path(x).stem[1:], 16)
    )
    df = (
        df.sort_values(["patient_id", "filename_sort_key"])
          .drop(columns="filename_sort_key")
    )

    # for debuging only these error were unsolvable
    debug_df = df[~df["patient_id"].isin(["1854", "1899", "21", "3092", "3112-ery-gas", "3188"])].reset_index(drop=True)

    # outliers have different hexadecimal sorting and date sorting
    sorted_by_filenames_df = debug_df.copy().reset_index(drop=True)
    sorted_by_timetstamp_df = debug_df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    diff =  sorted_by_filenames_df.compare(sorted_by_timetstamp_df, result_names=("filename_sort", "timestamp_sort"))

    if diff.empty:
        print("No outlier from sorting logic")
    else:
        print(diff)
        raise Exception("Outlier from sorting found")

    sorted_df = df.sort_values(["patient_uuid", "timestamp"]).reset_index(drop=True)

    sorted_df["timestamp_dt"] = pd.to_datetime(sorted_df["timestamp"], errors="raise")
    sorted_df["visit_date"] = sorted_df["timestamp_dt"].dt.date

    visit_df = (
        sorted_df[["patient_uuid", "visit_date"]]
        .drop_duplicates()
        .sort_values(["patient_uuid", "visit_date"])
        .reset_index(drop=True)
    )

    visit_df["visit_uuid"] = [str(uuid4()) for _ in range(len(visit_df))]

    sorted_df = sorted_df.merge(
        visit_df[["patient_uuid", "visit_date", "visit_uuid"]],
        on=["patient_uuid", "visit_date"],
        how="left",
    )

    sorted_df = sorted_df.drop(columns=["timestamp_dt", "visit_date"])
    sorted_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)

    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    output_csv_path = Path(args.output_csv)

    if not csv_path.exists() or not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise ValueError("csv_path must be an existing CSV file.")

    if output_csv_path.exists() or output_csv_path.suffix.lower() != ".csv":
        raise ValueError("output_csv_path must be a non-existing CSV file path.")

    main(csv_path, output_csv_path)
