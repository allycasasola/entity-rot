"""
Explore extracted entity data from JSON files.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict


def load_data(file_path: Path) -> list[dict[str, Any]]:
    """Load JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze the extracted entity data."""
    total_sections = len(data)
    total_entities = 0
    entity_type_counts = Counter()
    entities_by_type = defaultdict(list)

    for section in data:
        entities = section.get("entities", [])
        total_entities += len(entities)

        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "unknown")

            entity_type_counts[entity_type] += 1

            # Store entity with its section metadata for sampling
            entities_by_type[entity_type].append(
                {
                    "name": entity_name,
                    "citation": section.get("citation"),
                    "hierarchy_path": section.get("hierarchy_path"),
                    "content": section.get("content"),
                    "section_id": section.get("section_id"),
                }
            )

    return {
        "total_sections": total_sections,
        "total_entities": total_entities,
        "entity_type_counts": entity_type_counts,
        "entities_by_type": entities_by_type,
    }


def print_statistics(analysis: dict[str, Any]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print(f"\nTotal sections: {analysis['total_sections']:,}")
    print(f"Total entities extracted: {analysis['total_entities']:,}")

    print("\nEntities by type:")
    for entity_type, count in sorted(
        analysis["entity_type_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / analysis["total_entities"]) * 100
        print(f"  {entity_type:15s}: {count:6,} ({percentage:5.1f}%)")


def sample_entities(
    entities_by_type: dict[str, list[dict[str, Any]]], sample_size: int = 10
) -> None:
    """Sample and display entities of each type."""
    print("\n" + "=" * 80)
    print("SAMPLE ENTITIES BY TYPE")
    print("=" * 80)

    for entity_type in sorted(entities_by_type.keys()):
        entities = entities_by_type[entity_type]

        print(f"\n{'-' * 80}")
        print(f"TYPE: {entity_type.upper()}")
        print(f"Total count: {len(entities):,}")
        print(f"{'-' * 80}")

        # Sample up to sample_size entities
        sample = random.sample(entities, min(sample_size, len(entities)))

        for i, entity in enumerate(sample, 1):
            print(f"\n[{i}] Entity: {entity['name']}")
            print(f"    Section ID: {entity['section_id']}")

            if entity["citation"]:
                print(f"    Citation: {entity['citation']}")

            if entity["hierarchy_path"]:
                print(f"    Hierarchy: {entity['hierarchy_path']}")

            if entity["content"]:
                # Show first 200 characters of content
                content_preview = entity["content"][:200]
                if len(entity["content"]) > 200:
                    content_preview += "..."
                print(f"    Content: {content_preview}")


def print_entity_name_samples(
    entities_by_type: dict[str, list[dict[str, Any]]],
) -> None:
    """Print just entity names for a quick overview."""
    print("\n" + "=" * 80)
    print("ENTITY NAME SAMPLES (Quick Overview)")
    print("=" * 80)

    for entity_type in sorted(entities_by_type.keys()):
        entities = entities_by_type[entity_type]
        sample = random.sample(entities, min(100, len(entities)))

        print(f"\n{entity_type.upper()} ({len(entities):,} total):")
        entity_names = [e["name"] for e in sample]
        print("  " + ", ".join(entity_names))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore extracted entity data from JSON files"
    )
    parser.add_argument(
        "file_path", type=Path, help="Path to JSON file containing extracted entities"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of entities to sample per type (default: 10)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed samples with full metadata",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible sampling"
    )

    args = parser.parse_args()

    # Validate file exists
    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        return

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading data from: {args.file_path}")
    data = load_data(args.file_path)

    print("Analyzing data...")
    analysis = analyze_data(data)

    # Print statistics
    print_statistics(analysis)

    # Print entity name samples (always)
    print_entity_name_samples(analysis["entities_by_type"])

    # Print detailed samples if requested
    if args.detailed:
        sample_entities(analysis["entities_by_type"], args.sample_size)
    else:
        print("\n" + "=" * 80)
        print("Tip: Use --detailed flag to see entity samples with full context")
        print("=" * 80)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
