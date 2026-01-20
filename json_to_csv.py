"""
Convert analyzed entities JSON to CSV with full content.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, List


def load_data(file_path: Path) -> List[dict[str, Any]]:
    """Load the analyzed entities JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_csv(data: List[dict[str, Any]], output_path: Path) -> None:
    """Convert JSON data to CSV format with full content."""
    rows = []

    for section in data:
        section_id = section.get("section_id", "")
        section_heading = section.get("section_heading", "")
        citation = section.get("citation", "")
        hierarchy_path = section.get("hierarchy_path", "")
        content = section.get("content", "")

        entities = section.get("entities", [])

        for entity in entities:
            # Format citations as a single string
            citations = entity.get("citations", [])
            citations_str = " | ".join(
                [f"{c.get('url', '')}: {c.get('quote', '')}" for c in citations]
            )

            row = {
                "section_id": section_id,
                "section_heading": section_heading,
                "citation": citation,
                "hierarchy_path": hierarchy_path,
                "entity_name": entity.get("name", ""),
                "entity_type": entity.get("type", ""),
                "processed": entity.get("processed", ""),
                "exists": entity.get("exists", ""),
                "nonexistence_status": entity.get("nonexistence_status", ""),
                "reasoning": entity.get("reasoning", ""),
                "citations": citations_str,
                "content": content,  # Full content, not excerpt
            }
            rows.append(row)

    # Write to CSV
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert analyzed entities JSON to CSV with full content"
    )
    parser.add_argument(
        "input_file", type=Path, help="Input JSON file with analyzed entities"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path (default: same name as input with .csv extension)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Determine output path
    if args.output_csv:
        csv_path = args.output_csv
    else:
        csv_path = args.input_file.with_suffix(".csv")

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} sections")

    # Convert to CSV
    print(f"Converting to CSV: {csv_path}")
    row_count = convert_to_csv(data, csv_path)
    print(f"✓ CSV file created successfully")
    print(f"  Total rows: {row_count:,}")
    print(f"  Columns: section_id, section_heading, citation, hierarchy_path,")
    print(f"           entity_name, entity_type, processed, exists,")
    print(f"           nonexistence_status, reasoning, citations, content")


if __name__ == "__main__":
    main()


