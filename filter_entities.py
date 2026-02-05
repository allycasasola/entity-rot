"""
Filter entities based on a list of words to exclude.
Removes entities whose representative_name matches any word in the filter list (case-insensitive).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, List, Set


def load_filter_words(csv_path: Path) -> Set[str]:
    """Load words to filter from CSV file and normalize to lowercase."""
    filter_words = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("words_to_filter", "").strip().lower()
            if word:
                filter_words.add(word)

    return filter_words


def load_entities(file_path: Path) -> List[dict[str, Any]]:
    """Load entities JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_entities(
    entities: List[dict[str, Any]], filter_words: Set[str]
) -> tuple[List[dict[str, Any]], List[dict[str, Any]]]:
    """
    Filter entities based on representative_name matching filter words.

    Returns:
        - filtered_entities: Entities that passed the filter
        - removed_entities: Entities that were filtered out
    """
    filtered_entities = []
    removed_entities = []

    for entity in entities:
        representative_name = entity.get("representative_name", "").strip().lower()

        # Check if the representative_name matches any filter word
        if representative_name in filter_words:
            removed_entities.append(entity)
        else:
            filtered_entities.append(entity)

    return filtered_entities, removed_entities


def save_entities(entities: List[dict[str, Any]], output_path: Path) -> None:
    """Save entities to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Filter entities based on a list of words to exclude"
    )
    parser.add_argument(
        "input_file", type=Path, help="Input JSON file with entities to filter"
    )
    parser.add_argument(
        "--filter-csv",
        type=Path,
        default=Path("entities_to_filter.csv"),
        help="CSV file with words to filter (default: entities_to_filter.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: input_file with '_filtered' suffix)",
    )
    parser.add_argument(
        "--save-removed",
        action="store_true",
        help="Save removed entities to a separate file",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    if not args.filter_csv.exists():
        print(f"Error: Filter CSV not found: {args.filter_csv}")
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_file.parent / f"{args.input_file.stem}_filtered.json"

    # Load filter words
    print(f"Loading filter words from {args.filter_csv}...")
    filter_words = load_filter_words(args.filter_csv)
    print(f"Loaded {len(filter_words)} filter words: {', '.join(sorted(filter_words))}")
    print()

    # Load entities
    print(f"Loading entities from {args.input_file}...")
    entities = load_entities(args.input_file)
    print(f"Loaded {len(entities)} entities")
    print()

    # Filter entities
    print("Filtering entities...")
    filtered_entities, removed_entities = filter_entities(entities, filter_words)

    # Print statistics
    print("=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print(f"Original entities: {len(entities):,}")
    print(f"Kept entities: {len(filtered_entities):,}")
    print(f"Removed entities: {len(removed_entities):,}")
    print(f"Removal rate: {len(removed_entities)/len(entities)*100:.1f}%")
    print()

    # Show some examples of removed entities
    if removed_entities:
        print("Sample of removed entities:")
        for entity in removed_entities[:10]:
            name = entity.get("representative_name", "")
            types = ", ".join(entity.get("types", []))
            num_sections = len(entity.get("sections", []))
            print(f"  - {name} (types: {types}, sections: {num_sections})")
        if len(removed_entities) > 10:
            print(f"  ... and {len(removed_entities) - 10} more")
        print()

    # Save filtered entities
    print(f"Saving filtered entities to {output_path}...")
    save_entities(filtered_entities, output_path)
    print(f"✓ Saved {len(filtered_entities)} entities")

    # Optionally save removed entities
    if args.save_removed and removed_entities:
        removed_path = args.input_file.parent / f"{args.input_file.stem}_removed.json"
        print(f"\nSaving removed entities to {removed_path}...")
        save_entities(removed_entities, removed_path)
        print(f"✓ Saved {len(removed_entities)} removed entities")

    print("\n" + "=" * 80)
    print("✓ Filtering complete!")
    print(f"  Input: {args.input_file}")
    print(f"  Output: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
