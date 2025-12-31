#!/usr/bin/env python3
"""
TTNN Operation Build Integrator

Updates CMakeLists.txt and __init__.cpp files to integrate the scaffolded operation.
This script is idempotent - safe to run multiple times.
"""

import sys
import json
import os
import re
from pathlib import Path


def add_to_nanobind_cmake(repo_root: Path, category: str, operation_name: str) -> bool:
    """Add nanobind source to ttnn/CMakeLists.txt"""
    cmake_path = repo_root / "ttnn" / "CMakeLists.txt"

    if not cmake_path.exists():
        print(f"Error: CMakeLists.txt not found at {cmake_path}", file=sys.stderr)
        return False

    with open(cmake_path, "r") as f:
        content = f.read()

    # Entry to add
    nanobind_entry = f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/cpp/ttnn/operations/{category}/{operation_name}/{operation_name}_nanobind.cpp"

    # Check if already added
    if nanobind_entry.strip() in content:
        print(f"Nanobind entry already exists in {cmake_path}")
        return True

    # Find TTNN_SRC_PYBIND section
    # Look for the list definition
    match = re.search(r"(set\s*\(\s*TTNN_SRC_PYBIND\s*\n)(.*?)(\n\s*\))", content, re.DOTALL)

    if not match:
        print(f"Error: Could not find TTNN_SRC_PYBIND in {cmake_path}", file=sys.stderr)
        return False

    # Add entry at the end of the list
    before = match.group(1) + match.group(2)
    after = match.group(3)

    # Insert new entry before the closing parenthesis
    new_content = content[: match.start()] + before + "\n" + nanobind_entry + after + content[match.end() :]

    # Write back
    with open(cmake_path, "w") as f:
        f.write(new_content)

    print(f"Added nanobind entry to {cmake_path}")
    return True


def add_to_category_cmake(repo_root: Path, category: str, operation_name: str) -> bool:
    """Add cpp sources to ttnn/cpp/ttnn/operations/{category}/CMakeLists.txt

    Supports two CMake patterns:
    1. set(SOURCES ...) - older pattern used by some categories
    2. target_sources(... PRIVATE ...) - newer pattern used by pool, etc.
    """
    cmake_path = repo_root / "ttnn" / "cpp" / "ttnn" / "operations" / category / "CMakeLists.txt"

    if not cmake_path.exists():
        print(f"Error: Category CMakeLists.txt not found at {cmake_path}", file=sys.stderr)
        return False

    with open(cmake_path, "r") as f:
        content = f.read()

    # Entries to add (with consistent 8-space indentation for target_sources pattern)
    entries = [
        f"        {operation_name}/device/{operation_name}_device_operation.cpp",
        f"        {operation_name}/device/{operation_name}_program_factory.cpp",
        f"        {operation_name}/{operation_name}.cpp",
    ]

    # Also prepare 4-space versions for set(SOURCES) pattern
    entries_4space = [
        f"    {operation_name}/device/{operation_name}_device_operation.cpp",
        f"    {operation_name}/device/{operation_name}_program_factory.cpp",
        f"    {operation_name}/{operation_name}.cpp",
    ]

    # Check if already added (check both indent patterns)
    base_entries = [
        f"{operation_name}/device/{operation_name}_device_operation.cpp",
        f"{operation_name}/device/{operation_name}_program_factory.cpp",
        f"{operation_name}/{operation_name}.cpp",
    ]
    if all(entry in content for entry in base_entries):
        print(f"All entries already exist in {cmake_path}")
        return True

    # Try Pattern 1: set(SOURCES ...)
    match = re.search(r"(set\s*\(\s*SOURCES\s*\n)(.*?)(\n\s*\))", content, re.DOTALL)
    if match:
        before = match.group(1) + match.group(2)
        after = match.group(3)

        additions = []
        for entry in entries_4space:
            if entry.strip() not in content:
                additions.append(entry)

        if additions:
            new_content = (
                content[: match.start()] + before + "\n" + "\n".join(additions) + after + content[match.end() :]
            )
            with open(cmake_path, "w") as f:
                f.write(new_content)
            print(f"Added {len(additions)} entries to {cmake_path} (set(SOURCES) pattern)")
        return True

    # Try Pattern 2: target_sources(... PRIVATE ...)
    # This pattern looks like:
    #   target_sources(
    #       ttnn_op_xxx
    #       ...
    #       PRIVATE
    #           file1.cpp
    #           file2.cpp
    #   )
    match = re.search(r"(target_sources\s*\(\s*\n\s*ttnn_op_\w+.*?PRIVATE\s*\n)(.*?)(\n\s*\))", content, re.DOTALL)
    if match:
        before = match.group(1) + match.group(2)
        after = match.group(3)

        additions = []
        for entry in entries:
            if entry.strip() not in content:
                additions.append(entry)

        if additions:
            new_content = (
                content[: match.start()] + before + "\n" + "\n".join(additions) + after + content[match.end() :]
            )
            with open(cmake_path, "w") as f:
                f.write(new_content)
            print(f"Added {len(additions)} entries to {cmake_path} (target_sources PRIVATE pattern)")
        return True

    # Neither pattern found
    print(f"Error: Could not find SOURCES or target_sources PRIVATE section in {cmake_path}", file=sys.stderr)
    print(f"You may need to manually add these entries:", file=sys.stderr)
    for entry in base_entries:
        print(f"  {entry}", file=sys.stderr)
    return False


def add_to_category_nanobind(repo_root: Path, category: str, operation_name: str) -> bool:
    """Add include and registration call to category's nanobind aggregation file.

    Some categories (like reduction, data_movement) have their own {category}_nanobind.cpp
    that aggregates all bind calls for that category. This function handles that pattern.

    Returns True if category aggregation file exists and was updated, False otherwise.
    """
    category_nanobind_path = repo_root / "ttnn" / "cpp" / "ttnn" / "operations" / category / f"{category}_nanobind.cpp"

    if not category_nanobind_path.exists():
        return False  # No category aggregation file, fall back to __init__.cpp

    with open(category_nanobind_path, "r") as f:
        content = f.read()

    # Include line - use full path from ttnn/operations
    include_line = f'#include "ttnn/operations/{category}/{operation_name}/{operation_name}_nanobind.hpp"'

    # Registration call - use fully qualified namespace since we're in a different namespace
    registration_call = f"    ::ttnn::operations::{operation_name}::bind_{operation_name}_operation(mod);"

    # Check if already added
    if f"{operation_name}_nanobind.hpp" in content and f"bind_{operation_name}_operation" in content:
        print(f"Operation already registered in {category_nanobind_path}")
        return True

    modified = False

    # Add include if not present
    if include_line not in content:
        # Find the last include in the file (before namespace declaration)
        include_pattern = r'#include\s+"[^"]+_nanobind\.hpp"'
        matches = list(re.finditer(include_pattern, content))

        if matches:
            last_match = matches[-1]
            insert_pos = last_match.end()
            content = content[:insert_pos] + "\n" + include_line + content[insert_pos:]
            print(f"Added include to {category_nanobind_path}")
            modified = True
        else:
            # Try to find any #include line
            any_include = list(re.finditer(r'#include\s+[<"][^">]+[">]', content))
            if any_include:
                last_match = any_include[-1]
                insert_pos = last_match.end()
                content = content[:insert_pos] + "\n" + include_line + content[insert_pos:]
                print(f"Added include to {category_nanobind_path}")
                modified = True
            else:
                print(f"Warning: Could not find include section in {category_nanobind_path}", file=sys.stderr)

    # Add registration call if not present
    if f"bind_{operation_name}_operation" not in content:
        # Find the closing brace of py_module function - look for pattern like:
        # void py_module(nb::module_& mod) { ... }
        # We want to insert before the final }
        py_module_match = re.search(r"void\s+py_module\s*\([^)]*\)\s*\{", content)

        if py_module_match:
            # Find the matching closing brace
            start = py_module_match.end()
            brace_count = 1
            pos = start
            while pos < len(content) and brace_count > 0:
                if content[pos] == "{":
                    brace_count += 1
                elif content[pos] == "}":
                    brace_count -= 1
                pos += 1

            if brace_count == 0:
                # pos is now right after the closing }, insert before it
                closing_brace_pos = pos - 1
                # Find the last non-whitespace line before closing brace
                content = content[:closing_brace_pos] + registration_call + "\n" + content[closing_brace_pos:]
                print(f"Added registration call to {category_nanobind_path}")
                modified = True
        else:
            print(f"Warning: Could not find py_module function in {category_nanobind_path}", file=sys.stderr)

    if modified:
        with open(category_nanobind_path, "w") as f:
            f.write(content)

    return True


def add_to_nanobind_init(repo_root: Path, category: str, operation_name: str) -> bool:
    """Add include and registration call to ttnn/cpp/ttnn-nanobind/__init__.cpp

    First tries to use category aggregation file (e.g., reduction_nanobind.cpp).
    Falls back to __init__.cpp if no aggregation file exists.
    """
    # First, try category-level aggregation file
    if add_to_category_nanobind(repo_root, category, operation_name):
        return True

    # Fall back to __init__.cpp
    init_path = repo_root / "ttnn" / "cpp" / "ttnn-nanobind" / "__init__.cpp"

    if not init_path.exists():
        print(f"Error: __init__.cpp not found at {init_path}", file=sys.stderr)
        return False

    with open(init_path, "r") as f:
        content = f.read()

    # Include line
    include_line = f'#include "ttnn/operations/{category}/{operation_name}/{operation_name}_nanobind.hpp"'

    # Registration call - use fully qualified namespace
    registration_call = f"    ::ttnn::operations::{operation_name}::bind_{operation_name}_operation(m_{category});"

    # Check if already added
    if include_line in content and f"bind_{operation_name}_operation" in content:
        print(f"Operation already registered in {init_path}")
        return True

    modified = False

    # Add include if not present
    if include_line not in content:
        # Find the last include for this category
        category_include_pattern = rf'#include "ttnn/operations/{category}/[^"]+_nanobind\.hpp"'
        matches = list(re.finditer(category_include_pattern, content))

        if matches:
            # Insert after the last category include
            last_match = matches[-1]
            insert_pos = last_match.end()
            content = content[:insert_pos] + "\n" + include_line + content[insert_pos:]
            print(f"Added include to {init_path}")
            modified = True
        else:
            print(f"Warning: Could not find existing includes for category {category}", file=sys.stderr)
            print(f"You may need to manually add: {include_line}")

    # Add registration call if not present
    if f"bind_{operation_name}_operation" not in content:
        # Find the category module initialization
        module_pattern = rf"(auto\s+m_{category}\s*=.*?;)"
        match = re.search(module_pattern, content)

        if match:
            # Find the next empty line or closing brace after this module
            insert_pos = match.end()

            # Look for existing bind_*_operation calls for this module
            next_section = content[insert_pos : insert_pos + 2000]

            # Find where to insert (after last bind call for this module)
            category_calls = list(re.finditer(rf"[a-z_:]+::bind_[a-z_]+_operation\(m_{category}\);", next_section))

            if category_calls:
                last_call = category_calls[-1]
                insert_pos = insert_pos + last_call.end()
                content = content[:insert_pos] + "\n" + registration_call + content[insert_pos:]
                print(f"Added registration call to {init_path}")
                modified = True
            else:
                print(
                    f"Warning: Could not find existing bind_*_operation calls for module m_{category}", file=sys.stderr
                )
                print(f"You may need to manually add: {registration_call}")
        else:
            print(f"Warning: Could not find module m_{category} initialization", file=sys.stderr)
            print(f"You may need to manually add: {registration_call}")

    if modified:
        with open(init_path, "w") as f:
            f.write(content)

    return True


def integrate_build(config_path: str, repo_root: str = None) -> bool:
    """Integrate operation into build system."""
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine repo root
    # Priority: .git directory (most reliable), then build_metal.sh (tt-metal specific)
    # Avoid using just 'ttnn' as marker since ttnn/cpp/ttnn/ exists and causes false matches
    if repo_root is None:
        current = Path(config_path).resolve()
        while current != current.parent:
            if (current / ".git").exists():
                repo_root = str(current)
                break
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

    operation_name = config["operation_name"]
    category = config["category"]

    print(f"Integrating {operation_name} into build system...")

    # 1. Update ttnn/CMakeLists.txt (nanobind)
    success = add_to_nanobind_cmake(repo_root, category, operation_name)
    if not success:
        return False

    # 2. Update ttnn/cpp/ttnn/operations/{category}/CMakeLists.txt
    success = add_to_category_cmake(repo_root, category, operation_name)
    if not success:
        return False

    # 3. Update ttnn/cpp/ttnn-nanobind/__init__.cpp
    success = add_to_nanobind_init(repo_root, category, operation_name)
    if not success:
        return False

    print(f"\nBuild integration complete!")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: integrate_build.py <config.json> [repo_root]")
        print("Example: integrate_build.py my_operation_scaffolding_config.json")
        sys.exit(1)

    config_path = sys.argv[1]
    repo_root = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    success = integrate_build(config_path, repo_root)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
