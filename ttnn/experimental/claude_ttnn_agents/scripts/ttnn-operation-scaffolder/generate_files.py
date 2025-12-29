#!/usr/bin/env python3
"""
TTNN Operation File Generator

Generates all scaffolding files from JSON config using Jinja2 templates.
This script is purely deterministic - no LLM calls.
"""

import sys
import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template


def generate_files(config_path: str, repo_root: str = None, check_existing: bool = True) -> dict:
    """
    Generate all scaffolding files from config.

    Args:
        config_path: Path to JSON config file
        repo_root: Repository root directory
        check_existing: If True, check for existing files and warn

    Returns:
        dict with 'created' and 'skipped' file lists
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine repo root
    # Priority: .git directory (most reliable), then build_metal.sh (tt-metal specific)
    # Avoid using just 'ttnn' as marker since ttnn/cpp/ttnn/ exists and causes false matches
    repo_root_explicit = repo_root is not None
    if repo_root is None:
        current = Path(config_path).resolve()
        while current != current.parent:
            # .git is the most reliable marker for repo root
            if (current / ".git").exists():
                repo_root = str(current)
                break
            # build_metal.sh is tt-metal specific and lives at repo root
            if (current / "build_metal.sh").exists():
                repo_root = str(current)
                break
            current = current.parent
        if repo_root is None:
            repo_root = os.getcwd()
            print(f"Warning: Could not find .git or build_metal.sh, using cwd: {repo_root}", file=sys.stderr)
        else:
            print(f"Note: Auto-detected repo_root: {repo_root}")
    else:
        print(f"Using explicit repo_root: {repo_root}")

    repo_root = Path(repo_root)

    # Setup Jinja2 environment
    script_dir = Path(__file__).parent
    template_dir = script_dir / "templates"
    env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

    # Determine operation directory
    operation_path = repo_root / config["operation_path"]
    device_path = operation_path / "device"

    # Create directories if they don't exist
    operation_path.mkdir(parents=True, exist_ok=True)
    device_path.mkdir(parents=True, exist_ok=True)

    # Define file mappings: (template_name, output_path)
    operation_name = config["operation_name"]
    files_to_generate = [
        ("device_operation_types.hpp.j2", device_path / f"{operation_name}_device_operation_types.hpp"),
        ("device_operation.hpp.j2", device_path / f"{operation_name}_device_operation.hpp"),
        ("device_operation.cpp.j2", device_path / f"{operation_name}_device_operation.cpp"),
        ("program_factory.hpp.j2", device_path / f"{operation_name}_program_factory.hpp"),
        ("program_factory.cpp.j2", device_path / f"{operation_name}_program_factory.cpp"),
        ("operation.hpp.j2", operation_path / f"{operation_name}.hpp"),
        ("operation.cpp.j2", operation_path / f"{operation_name}.cpp"),
        ("nanobind.hpp.j2", operation_path / f"{operation_name}_nanobind.hpp"),
        ("nanobind.cpp.j2", operation_path / f"{operation_name}_nanobind.cpp"),
    ]

    created_files = []
    skipped_files = []

    # Generate each file
    for template_name, output_path in files_to_generate:
        # Check if file exists
        if check_existing and output_path.exists():
            print(f"Warning: File already exists, skipping: {output_path}")
            skipped_files.append(str(output_path))
            continue

        # Load template
        template = env.get_template(template_name)

        # Render template
        rendered = template.render(**config)

        # Write file
        with open(output_path, "w") as f:
            f.write(rendered)

        print(f"Created: {output_path}")
        created_files.append(str(output_path))

    return {
        "created": created_files,
        "skipped": skipped_files,
        "operation_path": str(operation_path),
        "operation_name": operation_name,
        "category": config["category"],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_files.py <config.json> [repo_root] [--force]")
        print("Example: generate_files.py my_operation_scaffolding_config.json")
        print("Options:")
        print("  --force: Overwrite existing files")
        sys.exit(1)

    config_path = sys.argv[1]
    repo_root = None
    check_existing = True

    # Parse remaining args
    for arg in sys.argv[2:]:
        if arg == "--force":
            check_existing = False
        elif repo_root is None:
            repo_root = arg

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    result = generate_files(config_path, repo_root, check_existing)

    print(f"\nGeneration complete!")
    print(f"Created: {len(result['created'])} files")
    if result["skipped"]:
        print(f"Skipped: {len(result['skipped'])} files (already exist)")


if __name__ == "__main__":
    main()
