"""
Fuzzy match entity names from extracted entities and group similar entities together.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rapidfuzz import fuzz


def load_extracted_entities(file_path: Path) -> List[Dict[str, Any]]:
    """Load the extracted entities from a JSONL file.
    Return a list of dictionaries with the keys name, type, and node_id."""

    entities = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                section = json.loads(line)
                for entity in section.get("entities", []):
                    entities.append(
                        {
                            "name": entity.get("name"),
                            "type": entity.get("type"),
                            "node_id": section.get("node_id"),
                        }
                    )
    return entities


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for better matching."""
    normalized = name.lower().strip()
    return normalized


def dedupe_by_name(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate entities by normalized name and type.

    Input: list of dicts with keys: name, type, node_id
    Output: list of dicts with keys: name, type, node_ids (list)
    """
    deduped_entities = []
    for entity in entities:
        entity_name = entity.get("name")
        entity_type = entity.get("type")
        entity_name_normalized = normalize_entity_name(entity_name)

        # Check if the entity name and type already exist in deduped_entities
        found = False
        for existing_entity in deduped_entities:
            if (
                normalize_entity_name(existing_entity["name"]) == entity_name_normalized
                and existing_entity["type"] == entity_type
            ):
                existing_entity["node_ids"].append(entity.get("node_id"))
                found = True
                break

        if not found:
            deduped_entities.append(
                {
                    "name": entity_name,
                    "type": entity_type,
                    "node_ids": [entity.get("node_id")],
                }
            )
    return deduped_entities


def tiebreak_types(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If entities have the same name but different types, keep the most common type.

    Input/Output: list of dicts with keys: name, type, node_ids
    """
    tiebroken_entities = []

    for entity in entities:
        entity_name = entity.get("name")

        # If already in tiebroken_entities, skip
        if normalize_entity_name(entity_name) in [
            normalize_entity_name(e.get("name")) for e in tiebroken_entities
        ]:
            continue

        # Find all entities with the same normalized name
        same_name_entities = [
            e
            for e in entities
            if normalize_entity_name(e.get("name")) == normalize_entity_name(entity_name)
        ]

        if len(same_name_entities) > 1:
            # Tiebreak by the type with the most occurrences (node_ids)
            most_common_type = max(
                same_name_entities, key=lambda x: len(x.get("node_ids", []))
            )["type"]

            # Merge all node_ids from entities with the same name
            new_node_ids = []
            for e in same_name_entities:
                new_node_ids.extend(e.get("node_ids", []))

            tiebroken_entities.append(
                {
                    "name": entity_name,
                    "type": most_common_type,
                    "node_ids": new_node_ids,
                }
            )
        else:
            tiebroken_entities.append(entity)

    return tiebroken_entities


def fuzzy_match_entities(
    entities: List[Dict[str, Any]],
    similarity_threshold: int = 85,
) -> List[List[Dict[str, Any]]]:
    """
    Fuzzy match entity names and group similar entities together.

    Returns a list of groups, where each group is a list of entity dicts
    that were fuzzy matched together.

    Assumes input entities are already deduplicated by exact normalized name.
    """
    matched_groups: List[List[Dict[str, Any]]] = []

    for entity in entities:
        entity_name = entity.get("name")
        entity_name_normalized = normalize_entity_name(entity_name)

        # Try to find a matching group
        matched = False
        for group in matched_groups:
            for other_entity in group:
                similarity = fuzz.token_sort_ratio(
                    entity_name_normalized,
                    normalize_entity_name(other_entity.get("name")),
                )
                if similarity >= similarity_threshold:
                    group.append(entity)
                    matched = True
                    break
            if matched:
                break

        # If no match found, create a new group
        if not matched:
            matched_groups.append([entity])

    return matched_groups


def add_counts_to_groups(groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Add occurrence counts to each fuzzy-matched group.

    Returns a list of dicts, each with:
    - entities: the list of entity dicts in the group
    - occurrence_count: total number of node_ids across all entities
    """
    result = []
    for group in groups:
        occurrence_count = sum(len(entity.get("node_ids", [])) for entity in group)
        result.append({
            "entities": group,
            "occurrence_count": occurrence_count,
        })
    return result


def save_fuzzy_matched(result: List[List[Dict[str, Any]]], output_path: Path) -> None:
    """Save fuzzy matched entities to JSON file with counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add counts to each group
    result_with_counts = add_counts_to_groups(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_with_counts, f, indent=2, ensure_ascii=False)


DEFAULT_ROOT_DIR = Path("/Users/allisoncasasola/reglab/entity-rot")


def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy match entity names from extracted entities"
    )
    parser.add_argument(
        "city",
        help="City to process in format 'city_slug:state_code' (e.g., 'santa_clara_county:ca')",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory for inputs/outputs (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Override input file path (default: outputs/extraction/{city}/entities.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Similarity threshold (0-100) for fuzzy matching (default: 80)",
    )

    args = parser.parse_args()

    # Parse city specification
    parts = args.city.split(":")
    if len(parts) != 2:
        print(f"Error: Invalid city specification '{args.city}'")
        print("Expected format: 'city_slug:state_code'")
        return

    city_slug, state_code = parts
    city_name = f"{city_slug}_{state_code}"

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = args.root_dir / "outputs" / "extraction" / city_name / "entities.jsonl"

    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # Set output path to new directory structure
    output_dir = args.root_dir / "outputs" / "deduplication" / "fuzzy-matches" / city_name
    output_file = output_dir / "fuzzy_matched.json"

    print(f"Processing: {city_slug} ({state_code})")
    print("=" * 80)

    # Load extracted entities
    print(f"Loading extracted entities from {input_file}...")
    entities = load_extracted_entities(input_file)
    print(f"Loaded {len(entities)} entity occurrences")

    # Deduplicate entities by name
    print("Deduplicating entities by name and type...")
    deduped_entities = dedupe_by_name(entities)
    print(f"Deduplicated to {len(deduped_entities)} unique name+type combinations")

    # Tiebreak types
    print("Tiebreaking types for entities with same name...")
    tiebroken_entities = tiebreak_types(deduped_entities)
    print(f"Tiebroken to {len(tiebroken_entities)} unique entities")

    # Fuzzy match entities
    print(f"Fuzzy matching entities (threshold={args.threshold})...")
    matched_groups = fuzzy_match_entities(tiebroken_entities, args.threshold)
    print(f"Grouped into {len(matched_groups)} fuzzy matched groups")

    # Save fuzzy matched entities to JSON file
    save_fuzzy_matched(matched_groups, output_file)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
