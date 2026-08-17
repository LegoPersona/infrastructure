#!/usr/bin/env python3
"""
Generic MongoDB data loader with embeddings.

This script:
1. Scans a templates directory for .ldr files organized by category
2. Fetches embeddings from an embedding service for each description
3. Generates MongoDB insert commands with embedded vectors
4. Each category becomes a separate MongoDB collection

Usage:
    python3 embed_and_insert.py --templates-dir /path/to/templates \\
                                --embedding-url http://localhost:8003/embed \\
                                --output insert_commands.js
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import requests
import sys

from google.cloud import storage

# LegoColor IDs to associate with every inserted module document.
DEFAULT_COLORS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10 , 11, 14, 15, 17 ,18, 19, 25 ,28 ,68, 70, 71, 72, 86, 180, 191, 216, 225, 226, 308]


def filename_to_desc(filename: str) -> str:
    """Convert filename to human-readable description.

    Example: black_french_beard.ldr -> Black French Beard
    """
    name = filename.replace(".ldr", "")
    return " ".join(word.capitalize() for word in name.split("_"))


def upload_ldr(bucket: "storage.Bucket", key: str, path: Path) -> None:
    """Upload a single .ldr file to the assets bucket under the given object key."""
    bucket.blob(key).upload_from_filename(str(path), content_type="text/plain; charset=utf-8")


def get_embeddings(texts: List[str], embedding_url: str) -> List[List[float]]:
    """Fetch embeddings for multiple texts from the embedding service."""
    try:
        response = requests.post(embedding_url, json={"texts": texts}, timeout=30)
        response.raise_for_status()
        return response.json()["embeddings"]
    except requests.exceptions.ConnectionError:
        raise requests.RequestException(
            f"Failed to connect to embedding service at {embedding_url}\n"
            f"Make sure the embedding service is running on port 8003."
        )
    except requests.exceptions.Timeout:
        raise requests.RequestException(
            f"Embedding service request timed out. Service may be slow or unresponsive."
        )


def generate_insert_commands(
    templates_dir: Path,
    embedding_url: str,
    bucket: "storage.Bucket",
    gcs_prefix: str = "templates",
    output_file: Path = None,
    verbose: bool = False
) -> str:
    """Generate MongoDB insert commands for all templates with embeddings.

    Also uploads each .ldr template to the assets bucket and records its stable object key
    (ldrKey) on the document, so the lego-service can fetch templates by key.

    Args:
        templates_dir: Path to templates directory containing category subdirectories
        embedding_url: URL of the embedding service endpoint
        bucket: GCS bucket to upload .ldr templates into
        gcs_prefix: Object-key prefix under which templates are stored (default "templates")
        output_file: Optional file path to save the output
        verbose: Print progress to stderr

    Returns:
        String containing MongoDB insert commands
    """
    templates_dir = Path(templates_dir)

    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    output_lines = []

    # Iterate through each category directory
    for category_dir in sorted(templates_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        ldr_files = sorted(category_dir.glob("*.ldr"))

        if not ldr_files:
            continue

        if verbose:
            print(f"Processing category: {category_name} ({len(ldr_files)} items)", file=sys.stderr)

        output_lines.append(f"\n// ===== {category_name.upper()} =====")
        output_lines.append(f"// MongoDB Collection: {category_name}")
        output_lines.append(f"// Insert commands:\n")

        filenames = [f.name for f in ldr_files]
        descriptions = [filename_to_desc(f) for f in filenames]
        ldr_keys = [f"{gcs_prefix}/{category_name}/{name}" for name in filenames]

        if verbose:
            print(f"  Uploading {len(ldr_files)} templates to gs://{bucket.name}/{gcs_prefix}/{category_name}/...", file=sys.stderr, flush=True)

        for ldr_file, key in zip(ldr_files, ldr_keys):
            upload_ldr(bucket, key, ldr_file)

        if verbose:
            print(f"  Embedding {len(descriptions)} descriptions...", file=sys.stderr, flush=True)

        embeddings = get_embeddings(descriptions, embedding_url)

        documents = [
            {"moduleName": filename, "desc": desc, "embedding": embedding, "colors": DEFAULT_COLORS, "ldrKey": key}
            for filename, desc, embedding, key in zip(filenames, descriptions, embeddings, ldr_keys)
        ]

        # Build insertMany command
        output_lines.append(f'db.{category_name}.insertMany([')
        for i, doc in enumerate(documents):
            json_str = json.dumps(doc, indent=2)
            comma = "," if i < len(documents) - 1 else ""
            indented = "\n".join("  " + line for line in json_str.split("\n"))
            output_lines.append(f'{indented}{comma}')
        output_lines.append('])\n')

    # Upload the shared base templates (referenced by fixed keys in the lego-service; no docs).
    for base_file in sorted(templates_dir.glob("*.ldr")):
        key = f"{gcs_prefix}/{base_file.name}"
        if verbose:
            print(f"Uploading base template gs://{bucket.name}/{key}", file=sys.stderr)
        upload_ldr(bucket, key, base_file)

    output = "\n".join(output_lines)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        if verbose:
            print(f"Output saved to: {output_file}", file=sys.stderr)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate MongoDB insert commands with embeddings for LEGO templates",
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        epilog="""
Examples:
  # Upload templates to GCS and print insert commands
  python3 embed_and_insert.py ../lego-service/templates --gcs-bucket legopersona-assets

  # Specify output file
  python3 embed_and_insert.py ../lego-service/templates --gcs-bucket legopersona-assets -o insert_commands.js

  # Custom embedding service URL and key prefix
  python3 embed_and_insert.py ../lego-service/templates --gcs-bucket legopersona-assets \\
      --embedding-url http://localhost:8003/embed --gcs-prefix templates

Credentials: set GOOGLE_APPLICATION_CREDENTIALS to a service-account key with write access
to the assets bucket.
        """
    )

    parser.add_argument(
        "templates_dir",
        type=str,
        help="Path to templates directory containing category subdirectories with .ldr files"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path for MongoDB commands (default: print to stdout)"
    )

    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8003/embed",
        help="URL of the embedding service endpoint (default: http://localhost:8003/embed)"
    )

    parser.add_argument(
        "--gcs-bucket",
        type=str,
        required=True,
        help="GCS bucket to upload .ldr templates into (the private assets bucket)"
    )

    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="templates",
        help="Object-key prefix under which templates are stored (default: templates)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information to stderr"
    )

    args = parser.parse_args()

    try:
        bucket = storage.Client().bucket(args.gcs_bucket)
        output = generate_insert_commands(
            templates_dir=args.templates_dir,
            embedding_url=args.embedding_url,
            bucket=bucket,
            gcs_prefix=args.gcs_prefix,
            output_file=Path(args.output) if args.output else None,
            verbose=args.verbose
        )

        if not args.output:
            print(output)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
