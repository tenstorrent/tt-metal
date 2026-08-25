#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict compact-lock validator and deterministic C++ emitter.

This tool is intentionally self-contained and uses only the Python standard
library. Ordinary builds consume only the checked-in lock and write generated
files below the current CMake binary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ARTIFACT_KIND = "ttnn_matmul_registry_lock"
GENERATOR_VERSION = "ttnn-matmul-lock-emitter-v1"
LOCK_SCHEMA_VERSION = 1
KEY_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 2
CODEGEN_RECIPE_ABI = 1
MAX_ENTRIES = 4096
MAX_LOCK_BYTES = 32 * 1024 * 1024
HEX_40 = frozenset("0123456789abcdef")
HEX_64 = HEX_40


class LockValidationError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LockValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_lock(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > MAX_LOCK_BYTES:
        raise LockValidationError("lock exceeds the bounded byte size")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LockValidationError("lock must be UTF-8") from error
    try:
        value = json.loads(text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise LockValidationError(f"invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise LockValidationError("lock root must be an object")
    canonical = canonical_bytes(value)
    if raw not in {canonical, canonical + b"\n"}:
        raise LockValidationError("lock bytes are not canonical JSON")
    return value


def _reject_json_constant(value: str) -> None:
    raise LockValidationError(f"invalid number: {value}")


def canonical_bytes(value: Any) -> bytes:
    _reject_noncanonical_values(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _reject_noncanonical_values(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise LockValidationError(f"{path}: floats are forbidden")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_noncanonical_values(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LockValidationError(f"{path}: object keys must be strings")
            _reject_noncanonical_values(item, f"{path}.{key}")
        return
    raise LockValidationError(f"{path}: unsupported JSON value")


def _exact_fields(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockValidationError(f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise LockValidationError(f"{path} field mismatch: missing={missing}, unknown={unknown}")
    return value


def _uint(value: Any, bits: int, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LockValidationError(f"{path} must be an unsigned integer")
    minimum = 1 if positive else 0
    if not minimum <= value < 1 << bits:
        raise LockValidationError(f"{path} is outside uint{bits}")
    return value


def _boolean(value: Any, path: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise LockValidationError(f"{path} must be boolean")
    if expected is not None and value is not expected:
        raise LockValidationError(f"{path} must be {str(expected).lower()}")
    return value


def _string(value: Any, path: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise LockValidationError(f"{path} must be a nonempty bounded string")
    if "\x00" in value or "/" in value or "\\" in value:
        raise LockValidationError(f"{path} must not contain a path")
    return value


def _hex(value: Any, length: int, path: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(character not in HEX_64 for character in value):
        raise LockValidationError(f"{path} must be lowercase {length}-hex")
    return value


def _tensor(value: Any, path: str) -> dict[str, Any]:
    item = _exact_fields(value, {"buffer_type", "dtype", "layout", "memory_layout", "tile_height", "tile_width"}, path)
    if item["dtype"] not in {"bfloat16", "float32", "bfloat8_b"}:
        raise LockValidationError(f"{path}.dtype is unknown")
    if item["layout"] not in {"tile", "row_major"}:
        raise LockValidationError(f"{path}.layout is unknown")
    if item["memory_layout"] != "interleaved":
        raise LockValidationError(f"{path}.memory_layout is unsupported")
    if item["buffer_type"] not in {"dram", "l1"}:
        raise LockValidationError(f"{path}.buffer_type is unknown")
    _uint(item["tile_height"], 16, f"{path}.tile_height", positive=True)
    _uint(item["tile_width"], 16, f"{path}.tile_width", positive=True)
    return item


def _key(value: Any, path: str, domain: str) -> dict[str, Any]:
    fields = {
        "alpha_f32_bits",
        "architecture",
        "board_capability_class",
        "bcast_batch",
        "beta_f32_bits",
        "codegen_recipe_abi",
        "compute_grid_x",
        "compute_grid_y",
        "device_count",
        "has_activation",
        "has_bias",
        "input_a",
        "input_b",
        "logical_k",
        "logical_m",
        "logical_n",
        "mesh_cols",
        "mesh_rows",
        "padded_k",
        "padded_m",
        "padded_n",
        "output",
        "run_batched",
        "schema_version",
        "topology_sha256",
        "transpose_a",
        "transpose_b",
        "untilize_out",
    }
    item = _exact_fields(value, fields, path)
    if item["schema_version"] != KEY_SCHEMA_VERSION or item["codegen_recipe_abi"] != CODEGEN_RECIPE_ABI:
        raise LockValidationError(f"{path} has unsupported schema/recipe ABI")
    for name in ("logical_m", "logical_k", "logical_n", "padded_m", "padded_k", "padded_n"):
        _uint(item[name], 64, f"{path}.{name}", positive=True)
    if (
        item["padded_m"] < item["logical_m"]
        or item["padded_k"] < item["logical_k"]
        or item["padded_n"] < item["logical_n"]
    ):
        raise LockValidationError(f"{path} padded dimensions must cover logical dimensions")
    for name in ("architecture", "board_capability_class"):
        _uint(item[name], 32, f"{path}.{name}", positive=True)
    for name in ("device_count", "mesh_rows", "mesh_cols", "compute_grid_x", "compute_grid_y"):
        _uint(item[name], 16, f"{path}.{name}", positive=True)
    _hex(item["topology_sha256"], 64, f"{path}.topology_sha256")
    for name in ("transpose_a", "transpose_b", "has_bias", "has_activation", "untilize_out", "run_batched"):
        _boolean(item[name], f"{path}.{name}", False)
    if domain == "dense.addmm":
        _uint(item["alpha_f32_bits"], 32, f"{path}.alpha_f32_bits")
        _uint(item["beta_f32_bits"], 32, f"{path}.beta_f32_bits")
        if item["alpha_f32_bits"] in {0, 0x80000000}:
            raise LockValidationError(f"{path}.alpha_f32_bits must encode a nonzero binary32 value")
        if item["beta_f32_bits"] not in {0, 0x80000000}:
            raise LockValidationError(f"{path}.beta_f32_bits must encode positive or negative zero in v1")
    elif item["alpha_f32_bits"] is not None or item["beta_f32_bits"] is not None:
        raise LockValidationError(f"{path} scalar semantics are exclusive to dense.addmm")
    if item["bcast_batch"] is not None:
        raise LockValidationError(f"{path}.bcast_batch must be null")
    input_a = _tensor(item["input_a"], f"{path}.input_a")
    input_b = _tensor(item["input_b"], f"{path}.input_b")
    if input_a["layout"] != "tile" or input_b["layout"] != "tile":
        raise LockValidationError(f"{path}.input_a and {path}.input_b must use tile layout in v1")
    output = _tensor(item["output"], f"{path}.output")
    if (
        output["layout"] != "tile"
        or output["memory_layout"] != "interleaved"
        or output["buffer_type"] != "dram"
        or output["tile_height"] != 32
        or output["tile_width"] != 32
    ):
        raise LockValidationError(f"{path}.output is outside the first dense DRAM-interleaved tile-32 envelope")
    return item


def _validate_multi_core_reuse_work_split(key: dict[str, Any], program: dict[str, Any], path: str) -> None:
    input_a = key["input_a"]
    input_b = key["input_b"]
    output = key["output"]
    if input_a["tile_height"] != output["tile_height"] or input_b["tile_width"] != output["tile_width"]:
        raise LockValidationError(f"{path} input/output tile axes are inconsistent")

    dimensions_and_tiles = (
        (key["padded_m"], input_a["tile_height"], "padded_m/input_a.tile_height"),
        (key["padded_k"], input_a["tile_width"], "padded_k/input_a.tile_width"),
        (key["padded_k"], input_b["tile_height"], "padded_k/input_b.tile_height"),
        (key["padded_n"], input_b["tile_width"], "padded_n/input_b.tile_width"),
    )
    for dimension, tile, name in dimensions_and_tiles:
        if dimension % tile != 0:
            raise LockValidationError(f"{path} {name} must divide exactly")

    m_tiles = key["padded_m"] // input_a["tile_height"]
    input_a_k_tiles = key["padded_k"] // input_a["tile_width"]
    input_b_k_tiles = key["padded_k"] // input_b["tile_height"]
    n_tiles = key["padded_n"] // input_b["tile_width"]
    if input_a_k_tiles != input_b_k_tiles:
        raise LockValidationError(f"{path} input K tile counts are inconsistent")
    if input_a_k_tiles % program["in0_block_w"] != 0:
        raise LockValidationError(f"{path}.in0_block_w must divide padded K tiles")
    if m_tiles % program["per_core_m"] != 0:
        raise LockValidationError(f"{path}.per_core_m must divide padded M tiles")
    if n_tiles % program["per_core_n"] != 0:
        raise LockValidationError(f"{path}.per_core_n must divide padded N tiles")
    if program["per_core_m"] % program["out_subblock_h"] != 0:
        raise LockValidationError(f"{path}.out_subblock_h must divide per_core_m")
    if program["per_core_n"] % program["out_subblock_w"] != 0:
        raise LockValidationError(f"{path}.out_subblock_w must divide per_core_n")
    if program["out_subblock_h"] * program["out_subblock_w"] > 8:
        raise LockValidationError(f"{path} output subblock exceeds the destination-register bound")


def _recipe(value: Any, key: dict[str, Any], path: str) -> dict[str, Any]:
    item = _exact_fields(value, {"call_state", "compute_kernel_config", "program_config", "schema_version"}, path)
    if item["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise LockValidationError(f"{path}.schema_version is unsupported")

    program = _exact_fields(
        item["program_config"],
        {
            "allowed_worker_cores",
            "compute_grid_x",
            "compute_grid_y",
            "family",
            "in0_block_w",
            "out_subblock_h",
            "out_subblock_w",
            "per_core_m",
            "per_core_n",
        },
        f"{path}.program_config",
    )
    if program["family"] != "multi_core_reuse":
        raise LockValidationError(f"{path}.program_config.family is unknown")
    for name in ("compute_grid_x", "compute_grid_y"):
        _uint(program[name], 16, f"{path}.program_config.{name}", positive=True)
    for name in ("in0_block_w", "out_subblock_h", "out_subblock_w", "per_core_m", "per_core_n"):
        _uint(program[name], 32, f"{path}.program_config.{name}", positive=True)
    if program["allowed_worker_cores"] is not None:
        raise LockValidationError(f"{path}.program_config.allowed_worker_cores must record exact null")
    if program["compute_grid_x"] > key["compute_grid_x"] or program["compute_grid_y"] > key["compute_grid_y"]:
        raise LockValidationError(f"{path}.program_config compute grid exceeds the attested device grid")
    _validate_multi_core_reuse_work_split(key, program, f"{path}.program_config")

    ckc = _exact_fields(
        item["compute_kernel_config"],
        {
            "dst_full_sync_en",
            "fp32_dest_acc_en",
            "math_approx_mode",
            "math_fidelity",
            "packer_l1_acc",
            "throttle_level",
        },
        f"{path}.compute_kernel_config",
    )
    if ckc["math_fidelity"] not in {"lofi", "hifi2", "hifi3", "hifi4"}:
        raise LockValidationError(f"{path}.compute_kernel_config.math_fidelity is unknown")
    if ckc["throttle_level"] not in {"no_throttle", "throttle_1", "throttle_2", "throttle_3"}:
        raise LockValidationError(f"{path}.compute_kernel_config.throttle_level is unknown")
    for name in ("math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        _boolean(ckc[name], f"{path}.compute_kernel_config.{name}")

    state = _exact_fields(
        item["call_state"],
        {
            "bcast_batch",
            "global_cb",
            "output",
            "output_tile",
            "sub_device_id",
            "transpose_a",
            "transpose_b",
            "untilize_out",
            "user_core_coord",
            "user_fused_activation",
            "user_run_batched",
        },
        f"{path}.call_state",
    )
    _tensor(state["output"], f"{path}.call_state.output")
    if state["output"] != key["output"]:
        raise LockValidationError(f"{path}.call_state.output must equal the key output")
    for name in (
        "bcast_batch",
        "global_cb",
        "output_tile",
        "sub_device_id",
        "user_core_coord",
        "user_fused_activation",
    ):
        if state[name] is not None:
            raise LockValidationError(f"{path}.call_state.{name} must record exact null")
    for name in ("transpose_a", "transpose_b", "untilize_out", "user_run_batched"):
        _boolean(state[name], f"{path}.call_state.{name}", False)
    return item


def _certificate(value: Any, path: str) -> None:
    fields = {
        "baseline_calls",
        "baseline_ns",
        "baseline_policy_id",
        "baseline_sessions",
        "candidate_calls",
        "candidate_ns",
        "candidate_sessions",
        "evidence_sha256",
        "operational_lower_bound_ppm",
        "pcc_min_ppb",
        "speedup_ppm",
    }
    item = _exact_fields(value, fields, path)
    for name in fields - {"baseline_policy_id", "evidence_sha256"}:
        _uint(item[name], 64, f"{path}.{name}", positive=True)
    _string(item["baseline_policy_id"], f"{path}.baseline_policy_id")
    _hex(item["evidence_sha256"], 64, f"{path}.evidence_sha256")


def content_sha256(lock: dict[str, Any]) -> str:
    unsigned = dict(lock)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def entry_id(entry: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes({"domain": entry["domain"], "key": entry["key"], "recipe": entry["recipe"]})
    ).hexdigest()


def validate_lock(value: Any) -> dict[str, Any]:
    fields = {
        "artifact_kind",
        "build_identity_sha256",
        "content_sha256",
        "entries",
        "key_schema_version",
        "lock_schema_version",
        "policy_version",
        "producer",
        "replay_schema_version",
        "runtime_capability_sha256",
        "semantic_source_sha256",
    }
    lock = _exact_fields(value, fields, "$")
    if lock["artifact_kind"] != ARTIFACT_KIND:
        raise LockValidationError("$.artifact_kind is unsupported")
    if (
        lock["lock_schema_version"] != LOCK_SCHEMA_VERSION
        or lock["key_schema_version"] != KEY_SCHEMA_VERSION
        or lock["replay_schema_version"] != REPLAY_SCHEMA_VERSION
    ):
        raise LockValidationError("lock schema version is unsupported")
    _string(lock["policy_version"], "$.policy_version")
    for name in ("semantic_source_sha256", "build_identity_sha256", "runtime_capability_sha256", "content_sha256"):
        _hex(lock[name], 64, f"$.{name}")
    if lock["content_sha256"] != content_sha256(lock):
        raise LockValidationError("$.content_sha256 mismatch")
    producer = _exact_fields(
        lock["producer"], {"codegen_commit", "generator_version", "measured_tt_metal_commit"}, "$.producer"
    )
    _hex(producer["codegen_commit"], 40, "$.producer.codegen_commit")
    _hex(producer["measured_tt_metal_commit"], 40, "$.producer.measured_tt_metal_commit")
    if producer["generator_version"] != GENERATOR_VERSION:
        raise LockValidationError("$.producer.generator_version is unsupported")
    entries = lock["entries"]
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        raise LockValidationError("$.entries must be a bounded array")
    if entries:
        compatibility_values = (
            lock["semantic_source_sha256"],
            lock["build_identity_sha256"],
            lock["runtime_capability_sha256"],
            producer["codegen_commit"],
            producer["measured_tt_metal_commit"],
        )
        if any(set(value) == {"0"} for value in compatibility_values):
            raise LockValidationError("nonempty locks require measured compatibility and provenance digests")
    prior_key: bytes | None = None
    ids: set[str] = set()
    keys: set[bytes] = set()
    for index, raw_entry in enumerate(entries):
        path = f"$.entries[{index}]"
        item = _exact_fields(raw_entry, {"certificate", "domain", "entry_id", "key", "recipe"}, path)
        if item["domain"] not in {"dense.matmul", "dense.linear", "dense.addmm"}:
            raise LockValidationError(f"{path}.domain is unsupported")
        _hex(item["entry_id"], 64, f"{path}.entry_id")
        key = _key(item["key"], f"{path}.key", item["domain"])
        recipe = _recipe(item["recipe"], key, f"{path}.recipe")
        _certificate(item["certificate"], f"{path}.certificate")
        if item["entry_id"] != entry_id(item):
            raise LockValidationError(f"{path}.entry_id mismatch")
        key_bytes = canonical_bytes({"domain": item["domain"], "key": key})
        if prior_key is not None and key_bytes < prior_key:
            raise LockValidationError("$.entries are not sorted by canonical key")
        if key_bytes in keys:
            raise LockValidationError(f"{path} duplicates an exact key")
        if item["entry_id"] in ids:
            raise LockValidationError(f"{path} duplicates an entry_id")
        prior_key = key_bytes
        keys.add(key_bytes)
        ids.add(item["entry_id"])
        item["key"] = key
        item["recipe"] = recipe
    return lock


def _bytes_cpp(value: str) -> str:
    return "{{" + ", ".join(f"0x{value[index : index + 2]}" for index in range(0, len(value), 2)) + "}}"


def _enum(prefix: str, value: str, mapping: dict[str, str]) -> str:
    return f"compact::{prefix}::{mapping[value]}"


def _tensor_cpp(value: dict[str, Any]) -> str:
    return (
        "compact::TensorDescriptor{"
        + ", ".join(
            (
                f".buffer_type = {_enum('BufferType', value['buffer_type'], {'dram': 'Dram', 'l1': 'L1'})}",
                f".dtype = {_enum('DataType', value['dtype'], {'bfloat16': 'BFloat16', 'float32': 'Float32', 'bfloat8_b': 'BFloat8B'})}",
                f".layout = {_enum('Layout', value['layout'], {'tile': 'Tile', 'row_major': 'RowMajor'})}",
                ".memory_layout = compact::MemoryLayout::Interleaved",
                f".tile_height = {value['tile_height']}",
                f".tile_width = {value['tile_width']}",
            )
        )
        + "}"
    )


def _tensor_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        {"dram": 0, "l1": 1}[value["buffer_type"]],
        {"bfloat16": 0, "bfloat8_b": 1, "float32": 2}[value["dtype"]],
        {"row_major": 0, "tile": 1}[value["layout"]],
        0,  # The first schema admits interleaved memory only.
        value["tile_height"],
        value["tile_width"],
    )


def _compact_key_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Mirror KeyDescriptor's defaulted comparison exactly."""
    key = item["key"]
    return (
        key["architecture"],
        False,
        False,  # bcast_batch is exact null in the first schema.
        key["board_capability_class"],
        key["codegen_recipe_abi"],
        key["compute_grid_x"],
        key["compute_grid_y"],
        key["device_count"],
        False,
        False,  # activation and bias are rejected in the first schema.
        _tensor_sort_key(key["input_a"]),
        _tensor_sort_key(key["input_b"]),
        key["logical_k"],
        key["logical_m"],
        key["logical_n"],
        key["mesh_cols"],
        key["mesh_rows"],
        _tensor_sort_key(key["output"]),
        key["padded_k"],
        key["padded_m"],
        key["padded_n"],
        False,  # run_batched
        key["schema_version"],
        bytes.fromhex(key["topology_sha256"]),
        False,
        False,
        False,  # transpose_a, transpose_b, untilize_out
        {"dense.matmul": 0, "dense.linear": 1, "dense.addmm": 2}[item["domain"]],
        key["alpha_f32_bits"] or 0,
        key["beta_f32_bits"] or 0,
    )


def _entry_cpp(item: dict[str, Any]) -> str:
    key = item["key"]
    recipe = item["recipe"]
    program = recipe["program_config"]
    ckc = recipe["compute_kernel_config"]
    state = recipe["call_state"]
    fidelity = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi3": "HiFi3", "hifi4": "HiFi4"}[ckc["math_fidelity"]]
    throttle = {
        "no_throttle": "NoThrottle",
        "throttle_1": "Throttle1",
        "throttle_2": "Throttle2",
        "throttle_3": "Throttle3",
    }[ckc["throttle_level"]]
    domain = {"dense.matmul": "DenseMatmul", "dense.linear": "DenseLinear", "dense.addmm": "DenseAddmm"}[item["domain"]]

    def boolean(value: bool) -> str:
        return "true" if value else "false"

    return f"""compact::EntryDescriptor{{
    .entry_id = {_bytes_cpp(item["entry_id"])},
    .key = compact::KeyDescriptor{{
        .architecture = {key["architecture"]}, .bcast_batch_present = false, .bcast_batch = false,
        .board_capability_class = {key["board_capability_class"]}, .codegen_recipe_abi = {key["codegen_recipe_abi"]},
        .compute_grid_x = {key["compute_grid_x"]}, .compute_grid_y = {key["compute_grid_y"]},
        .device_count = {key["device_count"]}, .has_activation = false, .has_bias = false,
        .input_a = {_tensor_cpp(key["input_a"])}, .input_b = {_tensor_cpp(key["input_b"])},
        .logical_k = {key["logical_k"]}ULL, .logical_m = {key["logical_m"]}ULL, .logical_n = {key["logical_n"]}ULL,
        .mesh_cols = {key["mesh_cols"]}, .mesh_rows = {key["mesh_rows"]}, .output = {_tensor_cpp(key["output"])},
        .padded_k = {key["padded_k"]}ULL, .padded_m = {key["padded_m"]}ULL, .padded_n = {key["padded_n"]}ULL,
        .run_batched = false, .schema_version = {key["schema_version"]},
        .topology_sha256 = {_bytes_cpp(key["topology_sha256"])},
        .transpose_a = false, .transpose_b = false, .untilize_out = false,
        .domain = compact::Domain::{domain},
        .alpha_f32_bits = {key["alpha_f32_bits"] or 0}, .beta_f32_bits = {key["beta_f32_bits"] or 0},
    }},
    .replay = compact::ReplayDescriptor{{
        .schema_version = {recipe["schema_version"]}, .family = compact::ProgramFamily::MultiCoreReuse,
        .program_config = compact::MultiCoreReuseDescriptor{{
            .compute_grid_x = {program["compute_grid_x"]}, .compute_grid_y = {program["compute_grid_y"]},
            .in0_block_w = {program["in0_block_w"]}, .out_subblock_h = {program["out_subblock_h"]},
            .out_subblock_w = {program["out_subblock_w"]}, .per_core_m = {program["per_core_m"]},
            .per_core_n = {program["per_core_n"]}, .allowed_worker_cores_present = false,
        }},
        .compute_kernel_config = compact::ComputeKernelDescriptor{{
            .math_fidelity = compact::MathFidelity::{fidelity}, .throttle_level = compact::ThrottleLevel::{throttle},
            .math_approx_mode = {boolean(ckc["math_approx_mode"])}, .fp32_dest_acc_en = {boolean(ckc["fp32_dest_acc_en"])},
            .packer_l1_acc = {boolean(ckc["packer_l1_acc"])}, .dst_full_sync_en = {boolean(ckc["dst_full_sync_en"])},
        }},
        .call_state = compact::CallStateDescriptor{{
            .output = {_tensor_cpp(state["output"])}, .untilize_out = false,
            .bcast_batch_is_null = true, .user_core_coord_is_null = true, .user_fused_activation_is_null = true,
            .user_run_batched_is_false = true, .transpose_a_is_false = true, .transpose_b_is_false = true,
            .output_tile_is_null = true, .global_cb_is_null = true, .sub_device_id_is_null = true,
        }},
    }},
}}"""


def emit(lock: dict[str, Any]) -> tuple[bytes, bytes]:
    checked = validate_lock(lock)
    header = """// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Generated by emit_cpp.py. Do not edit.
#pragma once
#include <span>
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"
namespace ttnn::operations::matmul::registry::generated {
const compact::TableMetadata& metadata() noexcept;
std::span<const compact::EntryDescriptor> entries() noexcept;
compact::ExactIndex index() noexcept;
}  // namespace ttnn::operations::matmul::registry::generated
"""
    # Lock review order is canonical JSON byte order. Runtime order follows the
    # POD's numeric/defaulted comparison, which is deliberately different for
    # values such as integer 2 versus 10.
    runtime_entries = sorted(checked["entries"], key=_compact_key_sort_key)
    entry_text = ",\n".join(_entry_cpp(entry) for entry in runtime_entries)
    source = f"""// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Generated by emit_cpp.py. Do not edit.
#include "matmul_registry_data.hpp"
#include <array>
namespace ttnn::operations::matmul::registry::generated {{
namespace {{
constexpr compact::TableMetadata kMetadata{{
    .lock_schema_version = {checked["lock_schema_version"]},
    .key_schema_version = {checked["key_schema_version"]},
    .replay_schema_version = {checked["replay_schema_version"]},
    .content_sha256 = {_bytes_cpp(checked["content_sha256"])},
    .semantic_source_sha256 = {_bytes_cpp(checked["semantic_source_sha256"])},
    .build_identity_sha256 = {_bytes_cpp(checked["build_identity_sha256"])},
    .runtime_capability_sha256 = {_bytes_cpp(checked["runtime_capability_sha256"])},
}};
constexpr std::array<compact::EntryDescriptor, {len(checked["entries"])}> kEntries{{{{
{entry_text}
}}}};
constexpr bool keys_are_strictly_sorted() {{
    for (std::size_t index = 1; index < kEntries.size(); ++index) {{
        if (!(kEntries[index - 1].key < kEntries[index].key)) {{
            return false;
        }}
    }}
    return true;
}}
static_assert(keys_are_strictly_sorted());
}}  // namespace
const compact::TableMetadata& metadata() noexcept {{ return kMetadata; }}
std::span<const compact::EntryDescriptor> entries() noexcept {{ return kEntries; }}
compact::ExactIndex index() noexcept {{ return compact::ExactIndex{{kEntries}}; }}
}}  // namespace ttnn::operations::matmul::registry::generated
"""
    return header.encode("utf-8"), source.encode("utf-8")


def _write_if_changed(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args(argv)
    header, source = emit(load_lock(arguments.lock))
    _write_if_changed(arguments.header, header)
    _write_if_changed(arguments.source, source)


if __name__ == "__main__":
    main()
