#!/usr/bin/env python3
"""
LEGO color data loader.

This script:
1. Reads a JSON file containing a list of color objects (name, hex, legoId)
2. Converts each hex value to LAB color space
3. Generates MongoDB insert commands for the lego_colors collection

Input JSON format:
    [
        {"name": "Bright Red", "hex": "#C91A09", "legoId": 21},
        ...
    ]

Usage:
    python3 insert_colors.py colors.json
    python3 insert_colors.py colors.json -o insert_colors.js
"""

import argparse
import json
from pathlib import Path
from typing import List
import sys

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


def hex_to_lab(hex_color: str) -> List[float]:
    """Convert a hex color string to CIELAB [L, a, b]."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    rgb = sRGBColor(r, g, b)
    lab = convert_color(rgb, LabColor)
    return [round(lab.lab_l, 4), round(lab.lab_a, 4), round(lab.lab_b, 4)]


def generate_insert_commands(
    colors_file: Path,
    output_file: Path = None,
    verbose: bool = False,
) -> str:
    """Generate MongoDB insert commands for all colors.

    Args:
        colors_file: Path to JSON file containing color objects
        output_file: Optional file path to save the output
        verbose: Print progress to stderr

    Returns:
        String containing MongoDB insert commands
    """
    if not colors_file.exists():
        raise FileNotFoundError(f"Colors file not found: {colors_file}")

    with open(colors_file, encoding="utf-8") as f:
        colors = json.load(f)

    if not isinstance(colors, list):
        raise ValueError("Input JSON must be a list of color objects")

    if verbose:
        print(f"Processing {len(colors)} colors...", file=sys.stderr)

    documents = []
    for entry in colors:
        name = entry["name"]
        hex_color = entry["hex"]
        lego_color_id = entry["legoId"]

        lab = hex_to_lab(hex_color)

        documents.append({
            "name": name,
            "hex": hex_color if hex_color.startswith("#") else f"#{hex_color}",
            "legoColorId": lego_color_id,
            "lab": lab,
        })

        if verbose:
            print(f"  {name} ({hex_color}) -> LAB {lab}", file=sys.stderr)

    output_lines = [
        "// ===== LEGO COLORS =====",
        "// MongoDB Collection: lego_colors",
        "// Insert commands:\n",
        "db.lego_colors.insertMany([",
    ]

    for i, doc in enumerate(documents):
        json_str = json.dumps(doc, indent=2)
        comma = "," if i < len(documents) - 1 else ""
        indented = "\n".join("  " + line for line in json_str.split("\n"))
        output_lines.append(f"{indented}{comma}")

    output_lines.append("])\n")

    output = "\n".join(output_lines)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        if verbose:
            print(f"Output saved to: {output_file}", file=sys.stderr)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate MongoDB insert commands for LEGO colors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage, print to stdout
  python3 insert_colors.py colors.json

  # Save to file
  python3 insert_colors.py colors.json -o insert_colors.js

  # Verbose output
  python3 insert_colors.py colors.json -v
        """,
    )

    parser.add_argument(
        "colors_file",
        type=str,
        help="Path to JSON file containing color objects with name, hex, and legoId fields",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path for MongoDB commands (default: print to stdout)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information to stderr",
    )

    args = parser.parse_args()

    try:
        output = generate_insert_commands(
            colors_file=Path(args.colors_file),
            output_file=Path(args.output) if args.output else None,
            verbose=args.verbose,
        )

        if not args.output:
            print(output)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
