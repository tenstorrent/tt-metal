#!/usr/bin/env python3
"""
TTNN Operation Config Validator/Saver

This script validates and saves JSON configuration for TTNN operation scaffolding.
The LLM parsing is done by the Claude Code agent, which then provides the JSON to this script.

Usage:
  # From stdin (agent pipes JSON)
  echo '{...json...}' | python3 parse_spec.py --from-stdin [output_path]

  # From JSON file (agent writes JSON, then script validates)
  python3 parse_spec.py --from-json config.json [output_path]
"""

import sys
import json
from pathlib import Path


def validate_config(config: dict) -> list:
    """Validate config has all required fields. Returns list of errors."""
    errors = []

    required_fields = [
        "operation_name",
        "operation_name_pascal",
        "category",
        "namespace",
        "parameters",
        "input_tensors",
        "validations",
        "output_shape",
        "output_dtype",
        "output_layout",
        "docstring",
    ]

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate operation_name format
    if "operation_name" in config:
        name = config["operation_name"]
        if not name.islower() or " " in name:
            errors.append(f"operation_name must be snake_case: {name}")

    # Validate parameters structure
    if "parameters" in config:
        for i, param in enumerate(config["parameters"]):
            if "name" not in param:
                errors.append(f"Parameter {i} missing 'name'")
            if "cpp_type" not in param:
                errors.append(f"Parameter {i} missing 'cpp_type'")

    # Validate input_tensors structure
    if "input_tensors" in config:
        for i, tensor in enumerate(config["input_tensors"]):
            if "name" not in tensor:
                errors.append(f"Input tensor {i} missing 'name'")
            if "cpp_name" not in tensor:
                errors.append(f"Input tensor {i} missing 'cpp_name'")

    # Validate validations structure
    if "validations" in config:
        for i, val in enumerate(config["validations"]):
            if "condition" not in val:
                errors.append(f"Validation {i} missing 'condition'")
            if "error_message" not in val:
                errors.append(f"Validation {i} missing 'error_message'")

    # Validate output_shape structure
    if "output_shape" in config:
        if "cpp_code" not in config["output_shape"]:
            errors.append("output_shape missing 'cpp_code'")

    return errors


def save_config(config: dict, output_path: str = None) -> str:
    """
    Validate and save config JSON to file.

    Args:
        config: Configuration dictionary
        output_path: Optional explicit output path

    Returns:
        Path where config was saved
    """
    # Validate
    errors = validate_config(config)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Add derived fields if missing
    if "operation_path" not in config:
        config["operation_path"] = f"ttnn/cpp/ttnn/operations/{config['category']}/{config['operation_name']}"

    # Determine output path
    if output_path is None:
        output_path = f"{config['operation_name']}_scaffolding_config.json"

    # Write config
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated config: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_spec.py --from-json <config.json> [output_path]")
        print("       parse_spec.py --from-stdin [output_path]")
        print("")
        print("This script validates and saves JSON config for operation scaffolding.")
        print("The LLM parsing is done by the Claude Code agent.")
        print("")
        print("Examples:")
        print("  # Agent writes JSON to file, then script validates/saves")
        print("  python3 parse_spec.py --from-json my_op_config.json")
        print("")
        print("  # Agent pipes JSON via stdin")
        print('  echo \'{"operation_name": "my_op", ...}\' | python3 parse_spec.py --from-stdin')
        sys.exit(1)

    # Mode 1: Load from JSON file
    if sys.argv[1] == "--from-json":
        if len(sys.argv) < 3:
            print("Error: --from-json requires a JSON file path", file=sys.stderr)
            sys.exit(1)

        json_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None

        try:
            with open(json_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {json_path}: {e}", file=sys.stderr)
            sys.exit(1)

        save_config(config, output_path)
        return

    # Mode 2: Read from stdin
    if sys.argv[1] == "--from-stdin":
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        try:
            config_json = sys.stdin.read()
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)

        save_config(config, output_path)
        return

    # Unknown mode
    print(f"Error: Unknown option: {sys.argv[1]}", file=sys.stderr)
    print("Use --from-json or --from-stdin", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
