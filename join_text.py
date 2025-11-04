"""
Join entity extraction results with original section data from parquet file.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
from tqdm import tqdm

DEFAULT_PARQUET_PATH = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"


def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Load JSON file containing entity extraction results."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(file_path: Path, data: list[dict[str, Any]]) -> None:
    """Save augmented JSON data to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_parquet_sections(
    parquet_path: Path, city_name: str
) -> dict[str, dict[str, Any]]:
    """
    Load all sections for a given city from parquet file.
    Returns a dictionary mapping section_id to section data.
    """
    conn = duckdb.connect()

    try:
        query = f"""
        SELECT 
            section_id,
            section_heading,
            citation,
            hierarchy_path,
            content
        FROM '{parquet_path}'
        WHERE city = ?
        """

        df = conn.execute(query, [city_name]).df()

        # Convert to dictionary mapping section_id -> section data
        sections_dict = {}
        for _, row in df.iterrows():
            section_id = row["section_id"]
            sections_dict[section_id] = {
                "section_heading": row["section_heading"],
                "citation": row["citation"],
                "hierarchy_path": row["hierarchy_path"],
                "content": row["content"],
            }

        return sections_dict

    finally:
        conn.close()


def augment_entities_with_section_data(
    entities_data: list[dict[str, Any]], sections_dict: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Augment each entity record with section metadata.
    """
    augmented = []
    missing_sections = []

    for record in tqdm(entities_data, desc="Augmenting records"):
        section_id = record.get("section_id")

        if not section_id:
            print(f"Warning: Record missing section_id: {record}")
            augmented.append(record)
            continue

        if section_id not in sections_dict:
            if section_id not in missing_sections:
                missing_sections.append(section_id)
            augmented.append(record)
            continue

        # Add section data to the record
        section_data = sections_dict[section_id]
        augmented_record = {
            **record,  # Keep existing fields (entities, section_id)
            **section_data,  # Add new fields (section_heading, citation, etc.)
        }

        augmented.append(augmented_record)

    if missing_sections:
        print(
            f"\nWarning: {len(missing_sections)} section_id(s) not found in parquet file:"
        )
        for sid in missing_sections[:10]:  # Show first 10
            print(f"  - {sid}")
        if len(missing_sections) > 10:
            print(f"  ... and {len(missing_sections) - 10} more")

    return augmented


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Join entity extraction results with section data from parquet file"
    )
    parser.add_argument(
        "file_name",
        type=Path,
        help="Path to JSON file containing entity extraction results",
    )
    parser.add_argument("city_name", help="City name to match in parquet file")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path(DEFAULT_PARQUET_PATH),
        help=f"Path to parquet file (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file path (default: overwrites input file)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.file_name.exists():
        print(f"Error: Input file not found: {args.file_name}")
        return

    # Validate parquet file exists
    if not args.parquet_path.exists():
        print(f"Error: Parquet file not found: {args.parquet_path}")
        return

    print(f"\n{'='*80}")
    print(f"Joining entity data with section metadata")
    print("=" * 80)
    print(f"Input file: {args.file_name}")
    print(f"City: {args.city_name}")
    print(f"Parquet: {args.parquet_path}")

    # Load entity extraction results
    print("\nLoading entity extraction results...")
    entities_data = load_json_file(args.file_name)
    print(f"Loaded {len(entities_data)} records")

    # Load parquet sections for the city
    print(f"\nLoading section data for {args.city_name} from parquet...")
    sections_dict = load_parquet_sections(args.parquet_path, args.city_name)
    print(f"Loaded {len(sections_dict)} sections")

    # Augment entity data with section metadata
    print("\nAugmenting records with section data...")
    augmented_data = augment_entities_with_section_data(entities_data, sections_dict)

    # Determine output path
    output_path = args.output if args.output else args.file_name

    # Save augmented data
    print(f"\nSaving augmented data to: {output_path}")
    save_json_file(output_path, augmented_data)

    print(f"\n✓ Successfully augmented {len(augmented_data)} records")
    print("=" * 80)


if __name__ == "__main__":
    main()
