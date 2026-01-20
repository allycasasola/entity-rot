"""
Explore analyzed entities data with statistics and deduplication.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List


def load_data(file_path: Path) -> List[dict[str, Any]]:
    """Load the analyzed entities JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_predictions(data: List[dict[str, Any]]) -> dict:
    """Analyze predictions and deduplicate entities."""
    # Lists to store all entity names by category
    positive_entities = []  # exists=True
    negative_entities = []  # exists=False
    unknown_entities = []  # exists=None/null

    # Count total predictions
    total_predictions = 0

    for section in data:
        entities = section.get("entities", [])
        for entity in entities:
            total_predictions += 1
            entity_name = entity.get("name", "").strip()
            exists = entity.get("exists")

            if exists is True:
                positive_entities.append(entity_name)
            elif exists is False:
                negative_entities.append(entity_name)
            else:  # None/null
                unknown_entities.append(entity_name)

    # Deduplicate (case-insensitive)
    def deduplicate(entity_list):
        # Normalize to lowercase for deduplication
        normalized = [name.lower() for name in entity_list if name]
        unique = set(normalized)
        # Create a counter for frequency
        counter = Counter([name.lower() for name in entity_list if name])
        return unique, counter

    positive_unique, positive_counter = deduplicate(positive_entities)
    negative_unique, negative_counter = deduplicate(negative_entities)
    unknown_unique, unknown_counter = deduplicate(unknown_entities)

    return {
        "total_predictions": total_predictions,
        "positive": {
            "count": len(positive_entities),
            "unique_count": len(positive_unique),
            "entities": positive_entities,
            "unique_entities": positive_unique,
            "counter": positive_counter,
        },
        "negative": {
            "count": len(negative_entities),
            "unique_count": len(negative_unique),
            "entities": negative_entities,
            "unique_entities": negative_unique,
            "counter": negative_counter,
        },
        "unknown": {
            "count": len(unknown_entities),
            "unique_count": len(unknown_unique),
            "entities": unknown_entities,
            "unique_entities": unknown_unique,
            "counter": unknown_counter,
        },
    }


def analyze_nonexistent_entities(data: List[dict[str, Any]]) -> dict:
    """
    Analyze non-existent entities, checking for conflicts and categorizing by type.

    Returns a dict with:
    - conflicts: entities with conflicting labels
    - nonexistence_status_counts: count by status for non-conflicting entities
    - all_entities_by_name: all occurrences grouped by entity name
    """
    # Group all entity occurrences by normalized name
    entities_by_name = defaultdict(list)

    for section in data:
        section_id = section.get("section_id", "")
        entities = section.get("entities", [])
        for entity in entities:
            entity_name = entity.get("name", "").strip()
            if not entity_name:
                continue

            # Normalize name (case-insensitive)
            normalized_name = entity_name.lower()

            # Store the entity with section context
            entities_by_name[normalized_name].append(
                {
                    "name": entity_name,  # Original case
                    "exists": entity.get("exists"),
                    "nonexistence_status": entity.get("nonexistence_status"),
                    "section_id": section_id,
                    "section_heading": section.get("section_heading", ""),
                    "type": entity.get("type", ""),
                }
            )

    # Analyze for conflicts
    conflicts = []
    nonexistence_status_counts = Counter()
    non_conflicting_entities = []

    for normalized_name, occurrences in entities_by_name.items():
        # Get all unique 'exists' values for this entity
        exists_values = set(occ["exists"] for occ in occurrences)

        # Get all unique 'nonexistence_status' values for entities with exists=False
        nonexistent_occurrences = [occ for occ in occurrences if occ["exists"] is False]

        if not nonexistent_occurrences:
            # This entity is never labeled as non-existent, skip it
            continue

        nonexistence_statuses = set(
            occ["nonexistence_status"] for occ in nonexistent_occurrences
        )

        # Check for conflicts
        has_conflict = False
        conflict_type = []

        # Conflict 1: Entity is labeled both existent and non-existent
        if len(exists_values) > 1:
            has_conflict = True
            conflict_type.append(
                f"exists field varies: {', '.join(str(v) for v in sorted(exists_values, key=lambda x: (x is None, x)))}"
            )

        # Conflict 2: Multiple different non-existence statuses
        # (excluding None, since None is valid if the status wasn't specified)
        non_none_statuses = {s for s in nonexistence_statuses if s is not None}
        if len(non_none_statuses) > 1:
            has_conflict = True
            conflict_type.append(
                f"nonexistence_status varies: {', '.join(sorted(non_none_statuses))}"
            )

        if has_conflict:
            conflicts.append(
                {
                    "entity_name": occurrences[0]["name"],  # Use original case
                    "normalized_name": normalized_name,
                    "conflict_types": conflict_type,
                    "occurrences": occurrences,
                    "exists_values": exists_values,
                    "nonexistence_statuses": nonexistence_statuses,
                }
            )
        else:
            # No conflict - count by nonexistence status
            # Use the most common nonexistence_status (should all be the same if no conflict)
            status = nonexistent_occurrences[0]["nonexistence_status"]
            if status is None:
                status = "unspecified"
            nonexistence_status_counts[status] += 1
            non_conflicting_entities.append(
                {
                    "entity_name": occurrences[0]["name"],
                    "status": status,
                    "count": len(nonexistent_occurrences),
                }
            )

    return {
        "conflicts": conflicts,
        "nonexistence_status_counts": nonexistence_status_counts,
        "non_conflicting_entities": non_conflicting_entities,
        "total_nonexistent_unique": len(conflicts) + len(non_conflicting_entities),
    }


def print_nonexistence_analysis(analysis: dict) -> None:
    """Print analysis of non-existent entities."""
    print("=" * 80)
    print("NON-EXISTENT ENTITY ANALYSIS")
    print("=" * 80)
    print()

    total = analysis["total_nonexistent_unique"]
    conflicts = analysis["conflicts"]
    status_counts = analysis["nonexistence_status_counts"]

    print(f"Total unique non-existent entities: {total}")
    print(f"  - Non-conflicting: {total - len(conflicts)}")
    print(f"  - Conflicting labels: {len(conflicts)}")
    print()

    # Show breakdown by nonexistence status for non-conflicting entities
    if status_counts:
        print("Non-existence status breakdown (non-conflicting entities only):")
        for status, count in status_counts.most_common():
            print(f"  - {status}: {count} unique entities")
        print()

    # Show conflicts
    if conflicts:
        print("=" * 80)
        print(f"CONFLICTING LABELS ({len(conflicts)} entities)")
        print("=" * 80)
        print()

        for i, conflict in enumerate(conflicts, 1):
            print(f"{i}. Entity: {conflict['entity_name']}")
            print(f"   Total occurrences: {len(conflict['occurrences'])}")
            print(f"   Conflicts: {'; '.join(conflict['conflict_types'])}")

            # Show breakdown of exists values
            exists_breakdown = Counter(occ["exists"] for occ in conflict["occurrences"])
            print(f"   Exists breakdown:", end="")
            for exists_val, count in sorted(
                exists_breakdown.items(), key=lambda x: (x[0] is None, x[0])
            ):
                print(f" {exists_val}={count},", end="")
            print()

            # Show nonexistence statuses if there are any
            nonexistent_occs = [
                occ for occ in conflict["occurrences"] if occ["exists"] is False
            ]
            if nonexistent_occs:
                status_breakdown = Counter(
                    occ["nonexistence_status"] for occ in nonexistent_occs
                )
                print(f"   Nonexistence status breakdown:", end="")
                for status, count in status_breakdown.most_common():
                    print(f" {status}={count},", end="")
                print()

            # Show a few example sections
            print(
                f"   Example sections: {', '.join(occ['section_id'] for occ in conflict['occurrences'][:3])}"
            )
            print()
    else:
        print("No conflicting labels found!")
        print()


def print_analysis(results: dict) -> None:
    """Print the analysis results."""
    total = results["total_predictions"]
    positive = results["positive"]
    negative = results["negative"]
    unknown = results["unknown"]

    print("=" * 80)
    print("ENTITY PREDICTION ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total predictions: {total:,}")
    print()

    # Positive predictions (exists=True)
    print("POSITIVE PREDICTIONS (exists=True):")
    print(
        f"  Total occurrences: {positive['count']:,} ({positive['count']/total*100:.1f}%)"
    )
    print(f"  Unique entities: {positive['unique_count']:,}")
    print(
        f"  Deduplication ratio: {positive['unique_count']/positive['count']*100:.1f}% "
        f"({positive['count'] - positive['unique_count']:,} duplicates)"
    )
    print()

    # Show top 10 most common positive entities
    if positive["counter"]:
        print("  Top 10 most common positive entities:")
        for entity, count in positive["counter"].most_common(10):
            print(f"    - {entity}: {count} occurrence(s)")
        print()

    # Negative predictions (exists=False)
    print("NEGATIVE PREDICTIONS (exists=False):")
    print(
        f"  Total occurrences: {negative['count']:,} ({negative['count']/total*100:.1f}%)"
    )
    print(f"  Unique entities: {negative['unique_count']:,}")
    if negative["count"] > 0:
        print(
            f"  Deduplication ratio: {negative['unique_count']/negative['count']*100:.1f}% "
            f"({negative['count'] - negative['unique_count']:,} duplicates)"
        )
    print()

    # Show top 10 most common negative entities
    if negative["counter"]:
        print("  Top 10 most common negative entities:")
        for entity, count in negative["counter"].most_common(10):
            print(f"    - {entity}: {count} occurrence(s)")
        print()

    # Unknown predictions (exists=None/null)
    print("UNKNOWN PREDICTIONS (exists=null):")
    print(
        f"  Total occurrences: {unknown['count']:,} ({unknown['count']/total*100:.1f}%)"
    )
    print(f"  Unique entities: {unknown['unique_count']:,}")
    if unknown["count"] > 0:
        print(
            f"  Deduplication ratio: {unknown['unique_count']/unknown['count']*100:.1f}% "
            f"({unknown['count'] - unknown['unique_count']:,} duplicates)"
        )
    print()

    # Show top 10 most common unknown entities
    if unknown["counter"]:
        print("  Top 10 most common unknown entities:")
        for entity, count in unknown["counter"].most_common(10):
            print(f"    - {entity}: {count} occurrence(s)")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(
        f"Total unique entities (deduplicated): "
        f"{positive['unique_count'] + negative['unique_count'] + unknown['unique_count']:,}"
    )
    print(f"  - Positive: {positive['unique_count']:,}")
    print(f"  - Negative: {negative['unique_count']:,}")
    print(f"  - Unknown: {unknown['unique_count']:,}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Explore analyzed entities with prediction statistics and deduplication"
    )
    parser.add_argument(
        "input_file", type=Path, help="Input JSON file with analyzed entities"
    )
    parser.add_argument(
        "--show-all-unique",
        action="store_true",
        help="Show all unique entities (can be very long)",
    )
    parser.add_argument(
        "--category",
        choices=["positive", "negative", "unknown"],
        help="Show all unique entities for a specific category",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} sections")
    print()

    # Analyze predictions
    results = analyze_predictions(data)

    # Print analysis
    print_analysis(results)

    # Analyze non-existent entities
    nonexistence_analysis = analyze_nonexistent_entities(data)
    print_nonexistence_analysis(nonexistence_analysis)

    # Show all unique entities if requested
    if args.show_all_unique:
        print("=" * 80)
        print("ALL UNIQUE ENTITIES")
        print("=" * 80)
        print()

        print(
            f"POSITIVE ENTITIES ({len(results['positive']['unique_entities'])} unique):"
        )
        for entity in sorted(results["positive"]["unique_entities"]):
            count = results["positive"]["counter"][entity]
            print(f"  - {entity} ({count} occurrence(s))")
        print()

        print(
            f"NEGATIVE ENTITIES ({len(results['negative']['unique_entities'])} unique):"
        )
        for entity in sorted(results["negative"]["unique_entities"]):
            count = results["negative"]["counter"][entity]
            print(f"  - {entity} ({count} occurrence(s))")
        print()

        print(
            f"UNKNOWN ENTITIES ({len(results['unknown']['unique_entities'])} unique):"
        )
        for entity in sorted(results["unknown"]["unique_entities"]):
            count = results["unknown"]["counter"][entity]
            print(f"  - {entity} ({count} occurrence(s))")
        print()

    # Show specific category if requested
    elif args.category:
        category = results[args.category]
        print("=" * 80)
        print(
            f"ALL UNIQUE {args.category.upper()} ENTITIES ({len(category['unique_entities'])} unique)"
        )
        print("=" * 80)
        print()
        for entity in sorted(category["unique_entities"]):
            count = category["counter"][entity]
            print(f"  - {entity} ({count} occurrence(s))")
        print()


if __name__ == "__main__":
    main()
