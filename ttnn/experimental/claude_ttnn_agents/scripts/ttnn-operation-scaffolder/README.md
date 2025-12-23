# TTNN Operation Scaffolder Scripts

This directory contains scripts for scaffolding TTNN operations using the modern device operation pattern.

## Overview

The scaffolder is a **hybrid system**: deterministic scripts + agent LLM intelligence.

### Architecture

```
Spec.md → [Agent] LLM parses → config.json → [1] generate_files.py → files
                                                      ↓
                                           [2] integrate_build.py
                                                      ↓
                                           [3] verify_scaffolding.sh
                                                      ↓
                                                  Build & Test
```

### Scripts

| Script | Type | Purpose |
|--------|------|---------|
| `parse_spec.py` | Validation | Validate and save config JSON (agent does LLM parsing) |
| `generate_files.py` | Deterministic | Render Jinja2 templates to create 9 source files |
| `integrate_build.py` | Deterministic | Update CMakeLists.txt and __init__.cpp |
| `verify_scaffolding.sh` | Deterministic | Check for banned/required patterns |

### Supporting Files

| File | Purpose |
|------|---------|
| `scaffolder_config_schema.json` | JSON schema for configuration |
| `templates/*.j2` | Jinja2 templates for all source files |

## Requirements

- Python 3.7+
- `jinja2`: `pip install jinja2`
- No API key needed (agent uses its own LLM capabilities)

## Usage

### Claude Code Agent Workflow

The agent does LLM parsing, then calls the scripts:

```bash
# From repository root
cd /path/to/tt-metal

# 1. Agent parses spec and writes JSON config directly

# 2. Agent calls generate_files.py
python3 .claude/scripts/ttnn-operation-scaffolder/generate_files.py \
  path/to/operation_scaffolding_config.json \
  /path/to/tt-metal

# 3. Agent calls integrate_build.py
python3 .claude/scripts/ttnn-operation-scaffolder/integrate_build.py \
  path/to/operation_scaffolding_config.json \
  /path/to/tt-metal

# 4. Agent calls verify_scaffolding.sh
bash .claude/scripts/ttnn-operation-scaffolder/verify_scaffolding.sh \
  ttnn/cpp/ttnn/operations/{category}/{operation} \
  {operation}

# 5. Agent runs build
./build_metal.sh -b Debug
```

### Script Details

#### parse_spec.py

**Purpose**: Validate and save config JSON.

**Two modes**:

1. **From file** - Agent writes JSON, script validates:
   ```bash
   python3 parse_spec.py --from-json config.json [output.json]
   ```

2. **From stdin** - Agent provides JSON via stdin:
   ```bash
   echo '{...json...}' | python3 parse_spec.py --from-stdin [output.json]
   ```

**Note**: The agent does LLM parsing using its built-in capabilities (no API key needed).

#### generate_files.py

**Purpose**: Generate all 9 scaffolding files from config using Jinja2 templates.

**Type**: Purely deterministic, no LLM calls.

**Usage**:
```bash
python3 generate_files.py <config.json> [repo_root] [--force]
```

**Options**:
- `--force`: Overwrite existing files (default: skip)

**Files created**:
1. `device/{op}_device_operation_types.hpp` - Type definitions
2. `device/{op}_device_operation.hpp` - Device operation header
3. `device/{op}_device_operation.cpp` - Device operation implementation
4. `device/{op}_program_factory.hpp` - Program factory header
5. `device/{op}_program_factory.cpp` - Program factory stub
6. `{op}.hpp` - Operation registration header
7. `{op}.cpp` - Operation implementation
8. `{op}_pybind.hpp` - Pybind declaration
9. `{op}_pybind.cpp` - Pybind implementation

#### integrate_build.py

**Purpose**: Update CMakeLists.txt and __init__.cpp to integrate operation.

**Type**: Deterministic, idempotent (safe to run multiple times).

**Usage**:
```bash
python3 integrate_build.py <config.json> [repo_root]
```

**What it does**:
1. Adds pybind source to `ttnn/CMakeLists.txt`
2. Adds cpp sources to `ttnn/cpp/ttnn/operations/{category}/CMakeLists.txt`
3. Adds include and registration to `ttnn/cpp/ttnn-pybind/__init__.cpp`

#### verify_scaffolding.sh

**Purpose**: Verify generated code follows modern device operation pattern.

**Usage**:
```bash
bash verify_scaffolding.sh <operation_path> <operation_name>
```

**Checks**:
1. No legacy file names (`*_op.hpp`, `*_op.cpp`)
2. All 9 required files exist
3. No banned patterns (`run_operation.hpp`, `operation::run`, etc.)
4. Required patterns present (`device_operation.hpp`, `ttnn::prim::`, etc.)
5. DeviceOperation struct has only static functions

**Exit code**: 0 if all pass, 1 if any fail

## Templates

All templates are in the `templates/` directory using Jinja2 syntax.

### Template Variables

Templates receive the config JSON as variables:
- `{{ operation_name }}` - snake_case name
- `{{ operation_name_pascal }}` - PascalCase name
- `{{ category }}` - operation category
- `{{ namespace }}` - C++ namespace
- `{{ parameters }}` - list of parameter dicts
- `{{ input_tensors }}` - list of tensor dicts
- `{{ validations }}` - list of validation dicts
- `{{ output_shape }}` - output shape dict
- `{{ output_dtype }}` - output dtype string
- `{{ output_layout }}` - output layout string
- `{{ docstring }}` - Python docstring

## Config Schema

The JSON config follows the schema in `scaffolder_config_schema.json`.

### Required Fields

- `operation_name`: snake_case name
- `operation_name_pascal`: PascalCase name
- `category`: TTNN category
- `namespace`: C++ namespace
- `operation_path`: Relative path to operation directory
- `parameters`: Array of parameter objects
- `input_tensors`: Array of tensor objects
- `validations`: Array of validation rules
- `output_shape`: Object with formula and cpp_code
- `output_dtype`: Dtype string or "same_as_input"
- `output_layout`: Layout enum string
- `docstring`: Python API documentation

### Example Config

```json
{
  "operation_name": "my_operation",
  "operation_name_pascal": "MyOperation",
  "category": "data_movement",
  "namespace": "ttnn::operations::my_operation",
  "operation_path": "ttnn/cpp/ttnn/operations/data_movement/my_operation",
  "parameters": [
    {
      "name": "kernel_size",
      "cpp_type": "uint32_t",
      "py_type": "int",
      "default": null,
      "description": "Size of the kernel"
    }
  ],
  "input_tensors": [
    {
      "name": "input",
      "cpp_name": "input",
      "required_rank": 4,
      "required_dtypes": ["DataType::BFLOAT16"],
      "required_layout": "Layout::ROW_MAJOR"
    }
  ],
  "validations": [
    {
      "condition": "input.logical_shape().rank() == 4",
      "error_message": "Input must be 4D, got rank {}",
      "error_args": ["input.logical_shape().rank()"]
    }
  ],
  "output_shape": {
    "formula": "same_as_input",
    "cpp_code": "ttnn::Shape output_shape = input.logical_shape();",
    "cpp_code_padded": "ttnn::Shape output_padded = input.padded_shape();"
  },
  "output_dtype": "same_as_input",
  "output_layout": "Layout::ROW_MAJOR",
  "docstring": "My operation description"
}
```

## Modern Device Operation Pattern

All generated code follows the modern pattern:

### Key Characteristics

1. **Static functions** - All member functions are static
2. **Nested structs** - `operation_attributes_t`, `tensor_args_t`
3. **File naming** - `{op}_device_operation.hpp/cpp` (NOT `{op}_op.hpp`)
4. **Includes** - `ttnn/device_operation.hpp` (NOT `ttnn/run_operation.hpp`)
5. **Registration** - `ttnn::prim::{op}()` (NOT `operation::run()`)
6. **Program factory** - `CachedProgram<SharedVariables>` (NOT `ProgramWithCallbacks`)

## Troubleshooting

### Missing Python packages

```bash
pip install jinja2
```

### JSON validation fails

- Check JSON syntax: `python3 -m json.tool config.json`
- Ensure all required fields exist (see schema)

### generate_files.py fails

- Check config JSON is valid
- Check Jinja2 is installed: `pip show jinja2`
- Always pass explicit repo_root as second argument

### integrate_build.py can't find insertion points

CMakeLists.txt may have unexpected format. Script will print manual instructions.

### verify_scaffolding.sh reports failures

Read the specific error messages. Common issues:
- Legacy file names exist
- Banned patterns in code
- Missing required patterns

### Build fails

The agent diagnoses and fixes:
1. Read compiler errors
2. Identify problematic files
3. Apply targeted fixes
4. Rebuild

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
