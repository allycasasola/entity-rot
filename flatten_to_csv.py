"""
Quick script to flatten fuzzy matched entities JSON to CSV.
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Flatten fuzzy matched entities to CSV")
    parser.add_argument("input_file", type=Path, help="Input JSON file")
    parser.add_argument(
        "--output", type=Path, help="Output CSV file (default: input with .csv extension)"
    )
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return
    
    output_path = args.output or args.input_file.with_suffix(".csv")
    
    print(f"Loading {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entities")
    print(f"Flattening to CSV...")
    
    rows = []
    for entity in data:
        row = {
            "representative_name": entity.get("representative_name", ""),
            "entity_name_varieties": " | ".join(entity.get("entity_name_varieties", [])),
            "types": ", ".join(entity.get("types", [])),
            "num_sections": len(entity.get("sections", [])),
        }
        rows.append(row)
    
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["representative_name", "entity_name_varieties", "types", "num_sections"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
