"""
Match North Island New Zealand birds against Perch 2.0 labels.

Instructions
------------
Run this script from the repository root or from any working directory:

    python scripts/match_north_island_perch_labels.py

The script reads:

    scripts/perch_label.csv
    scripts/north_island_nz_bird_list.csv

It updates north_island_nz_bird_list.csv by adding or replacing a
"Perch_2.0" column. Birds whose scientific_name appears in perch_label.csv
are marked with "Y"; birds not found in the Perch labels are left blank.

It also writes a separate match list:

    scripts/north_island_nz_perch_lablel.csv

That output file contains three columns:

    perch_label_number, common_name, scientific_name

The perch_label_number is the zero-based row number from perch_label.csv,
which corresponds to the usual model class index convention.
"""

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PERCH_LABELS_PATH = SCRIPT_DIR / "perch_label.csv"
NORTH_ISLAND_BIRDS_PATH = SCRIPT_DIR / "north_island_nz_bird_list.csv"
MATCH_OUTPUT_PATH = SCRIPT_DIR / "north_island_nz_perch_lablel.csv"

PERCH_COLUMN = "Perch_2.0"
SCIENTIFIC_NAME_COLUMN = "scientific_name"
COMMON_NAME_COLUMN = "common_name"
CSV_ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def clean_scientific_name(value):
    """Return a normalized scientific name for exact matching."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_csv_with_encoding_fallback(csv_path):
    """
    Read a CSV using common encodings and return both the DataFrame and encoding.

    The North Island list may contain non-UTF-8 characters from Excel or a web
    source. Trying a short encoding list makes the script easier to reuse.
    """
    last_error = None

    for encoding in CSV_ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(csv_path, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Could not read {csv_path} with {CSV_ENCODINGS_TO_TRY}",
    )


def load_perch_label_lookup(perch_labels_path):
    """
    Load Perch labels and return a mapping of scientific_name to label number.

    perch_label.csv currently has one column, so this uses the first column
    rather than relying on a specific header name.
    """
    perch_df, _encoding = read_csv_with_encoding_fallback(perch_labels_path)
    if perch_df.empty:
        raise ValueError(f"No labels found in {perch_labels_path}")

    label_column = perch_df.columns[0]
    label_lookup = {}

    for label_number, raw_label in perch_df[label_column].items():
        scientific_name = clean_scientific_name(raw_label)

        # Keep the first label number if a label appears more than once.
        if scientific_name and scientific_name not in label_lookup:
            label_lookup[scientific_name] = int(label_number)

    return label_lookup


def mark_north_island_birds(
    north_island_birds_path,
    perch_label_lookup,
    match_output_path,
):
    """
    Add the Perch_2.0 marker column and write the separate matched-label CSV.
    """
    birds_df, birds_encoding = read_csv_with_encoding_fallback(north_island_birds_path)

    required_columns = {COMMON_NAME_COLUMN, SCIENTIFIC_NAME_COLUMN}
    missing_columns = required_columns.difference(birds_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    # Use a temporary normalized column so whitespace differences do not block
    # otherwise exact scientific-name matches.
    birds_df["_scientific_name_match_key"] = birds_df[SCIENTIFIC_NAME_COLUMN].map(
        clean_scientific_name
    )

    birds_df[PERCH_COLUMN] = birds_df["_scientific_name_match_key"].map(
        lambda name: "Y" if name in perch_label_lookup else ""
    )

    matched_df = birds_df[birds_df[PERCH_COLUMN] == "Y"].copy()
    matched_df["perch_label_number"] = matched_df["_scientific_name_match_key"].map(
        perch_label_lookup
    )

    # Keep the requested three columns in the separate output file.
    matched_df = matched_df[
        ["perch_label_number", COMMON_NAME_COLUMN, SCIENTIFIC_NAME_COLUMN]
    ].sort_values("perch_label_number")

    birds_df = birds_df.drop(columns=["_scientific_name_match_key"])

    # Write both files. index=False prevents pandas from adding a row-number
    # column that is not part of the source data. The match output is written
    # first so it is still created if the source bird list is open elsewhere.
    matched_df.to_csv(match_output_path, index=False)

    try:
        birds_df.to_csv(north_island_birds_path, index=False, encoding=birds_encoding)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not update {north_island_birds_path}. Close the CSV if it is "
            "open in Excel or another program, then rerun this script. The "
            f"matched-label output was still written to {match_output_path}."
        ) from exc

    return len(matched_df), len(birds_df)


def main():
    perch_label_lookup = load_perch_label_lookup(PERCH_LABELS_PATH)
    matched_count, total_count = mark_north_island_birds(
        NORTH_ISLAND_BIRDS_PATH,
        perch_label_lookup,
        MATCH_OUTPUT_PATH,
    )

    print(
        f"Marked {matched_count} of {total_count} North Island birds as present "
        f"in Perch 2.0 labels."
    )
    print(f"Updated: {NORTH_ISLAND_BIRDS_PATH}")
    print(f"Wrote:   {MATCH_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
