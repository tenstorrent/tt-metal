# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict compact-lock validator and deterministic C++ emitter.

This tool is intentionally self-contained and uses only the Python standard
library. Promotion runs it offline; ordinary builds compile the checked-in C++
snapshot without invoking Python or parsing JSON.
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
POLICY_VERSION = "matmul-direct-bank-v2"
DIRECT_BANK_EVIDENCE_POLICY_VERSION = "deterministic-matmul-bank-v2"
LOCK_SCHEMA_VERSION = 2
KEY_SCHEMA_VERSION = 1
CODEGEN_RECIPE_ABI = 2
MAX_ENTRIES = 4096
MAX_LOCK_BYTES = 32 * 1024 * 1024
MATMUL_KERNEL_EQUIVALENCE = {
    "canonical_domain": "dense.matmul",
    "eligibility_gate": "preflight_v1_eligibility",
    "eligible_alias_domains": ["dense.addmm", "dense.linear"],
    "policy_id": "eligibility-proven-dense-matmul-kernel-equivalence-v1",
    "schema_version": 1,
}
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


def _program_config(value: Any, path: str) -> dict[str, Any]:
    fields = {
        "allowed_worker_cores",
        "compute_grid_x",
        "compute_grid_y",
        "family",
        "fuse_batch",
        "in0_block_w",
        "mcast_in0",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "num_global_cb_receivers",
        "per_core_m",
        "per_core_n",
        "transpose_mcast",
        "fused_activation_present",
        "gather_in0",
        "hop_cores_present",
        "untilize_out",
        "stream_in1",
    }
    item = _exact_fields(value, fields, path)
    family = item["family"]
    if family not in {"multi_core_reuse", "multi_cast_1d", "multi_cast_2d"}:
        raise LockValidationError(f"{path}.family is unsupported")
    for name in ("compute_grid_x", "compute_grid_y"):
        _uint(item[name], 16, f"{path}.{name}", positive=True)
    for name in ("in0_block_w", "out_subblock_h", "out_subblock_w", "per_core_m", "per_core_n"):
        _uint(item[name], 32, f"{path}.{name}", positive=True)
    for name in ("out_block_h", "out_block_w", "num_global_cb_receivers"):
        _uint(item[name], 32, f"{path}.{name}")
    if item["allowed_worker_cores"] is not None:
        raise LockValidationError(f"{path}.allowed_worker_cores must be null")
    for name in (
        "fuse_batch",
        "mcast_in0",
        "transpose_mcast",
        "fused_activation_present",
        "gather_in0",
        "hop_cores_present",
        "untilize_out",
        "stream_in1",
    ):
        _boolean(item[name], f"{path}.{name}")
    for name in ("fused_activation_present", "gather_in0", "hop_cores_present", "untilize_out", "stream_in1"):
        if item[name]:
            raise LockValidationError(f"{path}.{name} is outside the attested acquisition policy")
    flags = (item["fuse_batch"], item["mcast_in0"], item["transpose_mcast"])
    if family == "multi_core_reuse" and flags != (False, False, False):
        raise LockValidationError(f"{path} basic family flags are not canonical")
    if family == "multi_core_reuse" and any(
        item[name] != 0 for name in ("out_block_h", "out_block_w", "num_global_cb_receivers")
    ):
        raise LockValidationError(f"{path} basic derived fields are not canonical")
    if family == "multi_cast_1d" and (not item["fuse_batch"] or item["transpose_mcast"]):
        raise LockValidationError(f"{path} mm1d family flags are unsupported")
    if family == "multi_cast_1d" and (
        item["out_block_h"] != item["per_core_m"]
        or item["out_block_w"] != item["per_core_n"]
        or item["num_global_cb_receivers"] != 1
    ):
        raise LockValidationError(f"{path} mm1d fields must match attested acquisition defaults")
    if family == "multi_cast_2d" and (not item["fuse_batch"] or item["mcast_in0"]):
        raise LockValidationError(f"{path} mm2d family flags are unsupported")
    if family == "multi_cast_2d" and (
        item["out_block_h"] != item["per_core_m"]
        or item["out_block_w"] != item["per_core_n"]
        or item["num_global_cb_receivers"] != 0
    ):
        raise LockValidationError(f"{path} mm2d fields must match attested acquisition defaults")
    if item["out_subblock_h"] * item["out_subblock_w"] > 8:
        raise LockValidationError(f"{path} output subblock exceeds the native destination-register bound")
    if item["per_core_m"] % item["out_subblock_h"] or item["per_core_n"] % item["out_subblock_w"]:
        raise LockValidationError(f"{path} output subblock must divide the per-core block")
    return item


def _program_config_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        {"multi_core_reuse": 0, "multi_cast_1d": 1, "multi_cast_2d": 2}[value["family"]],
        value["compute_grid_x"],
        value["compute_grid_y"],
        value["in0_block_w"],
        value["out_subblock_h"],
        value["out_subblock_w"],
        value["per_core_m"],
        value["per_core_n"],
        value["out_block_h"],
        value["out_block_w"],
        value["num_global_cb_receivers"],
        False,
        value["fuse_batch"],
        value["mcast_in0"],
        value["transpose_mcast"],
        value["fused_activation_present"],
        value["gather_in0"],
        value["hop_cores_present"],
        value["untilize_out"],
        value["stream_in1"],
    )


def _compute_kernel_config(value: Any, path: str) -> dict[str, Any]:
    item = _exact_fields(
        value,
        {
            "dst_full_sync_en",
            "fp32_dest_acc_en",
            "math_approx_mode",
            "math_fidelity",
            "packer_l1_acc",
            "throttle_level",
        },
        path,
    )
    if item["math_fidelity"] not in {"lofi", "hifi2", "hifi3", "hifi4"}:
        raise LockValidationError(f"{path}.math_fidelity is unknown")
    if item["throttle_level"] not in {
        "no_throttle",
        "throttle_1",
        "throttle_2",
        "throttle_3",
        "throttle_4",
        "throttle_5",
    }:
        raise LockValidationError(f"{path}.throttle_level is unknown")
    for name in ("math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        _boolean(item[name], f"{path}.{name}")
    return item


def _compute_kernel_config_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        {"lofi": 0, "hifi2": 1, "hifi3": 2, "hifi4": 3}[value["math_fidelity"]],
        {
            "no_throttle": 0,
            "throttle_1": 1,
            "throttle_2": 2,
            "throttle_3": 3,
            "throttle_4": 4,
            "throttle_5": 5,
        }[value["throttle_level"]],
        value["math_approx_mode"],
        value["fp32_dest_acc_en"],
        value["packer_l1_acc"],
        value["dst_full_sync_en"],
    )


def _native_recipe_sort_key(program_config: dict[str, Any], compute_kernel_config: dict[str, Any]) -> tuple[Any, ...]:
    return _program_config_sort_key(program_config), _compute_kernel_config_sort_key(compute_kernel_config)


def program_config_candidate_id(program_config: dict[str, Any], compute_kernel_config: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes({"program_config": program_config, "compute_kernel_config": compute_kernel_config})
    ).hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_recipe_evidence_hash(evidence: dict[str, Any]) -> str:
    payload = {key: value for key, value in evidence.items() if key != "proof_sha256"}
    return _sha256_value(payload)


def direct_bank_entry_inventory_hash(lock: dict[str, Any]) -> str:
    return _sha256_value(
        [
            {
                "bank_evidence": entry["bank_evidence"],
                "entry_id": entry["entry_id"],
                "table_kind": "exact_recipe",
            }
            for entry in lock.get("program_config_exact_entries", [])
        ]
    )


def _direct_bank_evidence(
    value: Any,
    *,
    domain: str,
    key: dict[str, Any],
    program_config: dict[str, Any],
    compute_kernel_config: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    fields = {"lookup_key_sha256", "policy_version", "native_recipe_sha256", "schema_version", "source_sha256"}
    item = _exact_fields(value, fields, path)
    if item["schema_version"] != 2 or item["policy_version"] != DIRECT_BANK_EVIDENCE_POLICY_VERSION:
        raise LockValidationError(f"{path} direct-bank policy/schema is unsupported")
    for name in ("lookup_key_sha256", "native_recipe_sha256", "source_sha256"):
        _hex(item[name], 64, f"{path}.{name}")
        if set(item[name]) == {"0"}:
            raise LockValidationError(f"{path}.{name} must be nonzero")
    if item["lookup_key_sha256"] != _sha256_value({"domain": domain, "key": key}):
        raise LockValidationError(f"{path}.lookup_key_sha256 mismatch")
    native_recipe = {"program_config": program_config, "compute_kernel_config": compute_kernel_config}
    if item["native_recipe_sha256"] != _sha256_value(native_recipe):
        raise LockValidationError(f"{path}.native_recipe_sha256 mismatch")
    return item


def exact_native_support_hash(lock: dict[str, Any]) -> str:
    inventory = [
        {
            "table_kind": "exact_recipe",
            "entry_id": entry["entry_id"],
            "native_support": {
                "architecture": entry["key"]["architecture"],
                "device_count": entry["key"]["device_count"],
                "domain": entry["domain"],
                "input_a": entry["key"]["input_a"],
                "input_b": entry["key"]["input_b"],
                "mesh_cols": entry["key"]["mesh_cols"],
                "mesh_rows": entry["key"]["mesh_rows"],
                "output": entry["key"]["output"],
                "program_family": entry["program_config"]["family"],
                "compute_kernel_config": entry["compute_kernel_config"],
            },
        }
        for entry in lock.get("program_config_exact_entries", [])
    ]
    return _sha256_value(sorted(inventory, key=lambda item: (item["table_kind"], item["entry_id"])))


def _exact_entry_inventory(lock: dict[str, Any]) -> list[dict[str, str]]:
    return sorted(
        [
            {"table_kind": "exact_recipe", "entry_id": entry["entry_id"]}
            for entry in lock.get("program_config_exact_entries", [])
        ],
        key=lambda item: (item["table_kind"], item["entry_id"]),
    )


def exact_recipe_safety_inventory_hash(lock: dict[str, Any]) -> str:
    """Bind every exact-recipe safety artifact into one proof root."""

    preimage = {
        "artifact_kind": "ttnn_matmul_program_config_safety_inventory",
        "schema_version": 1,
        "exact_entry_inventory": sorted(
            [
                {
                    "table_kind": "exact_recipe",
                    "entry_id": entry["entry_id"],
                    "evidence_sha256": entry["bank_evidence"]["source_sha256"],
                }
                for entry in lock.get("program_config_exact_entries", [])
            ],
            key=lambda item: (item["table_kind"], item["entry_id"]),
        ),
    }
    return _sha256_value(preimage)


def _exact_recipe_evidence(value: Any, lock: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    fields = {
        "authorizes_exact_recipes",
        "bank_artifact_sha256",
        "bank_entry_inventory_sha256",
        "bank_policy_version",
        "exact_entry_inventory_sha256",
        "exact_native_support_sha256",
        "matmul_kernel_equivalence",
        "proof_sha256",
        "safety_evidence_sha256",
        "schema_version",
        "semantic_source_sha256",
    }
    item = _exact_fields(value, fields, "$.exact_recipe_evidence")
    if item["schema_version"] != 2 or item["bank_policy_version"] != DIRECT_BANK_EVIDENCE_POLICY_VERSION:
        raise LockValidationError("$.exact_recipe_evidence direct-bank policy/schema is unsupported")
    _boolean(item["authorizes_exact_recipes"], "$.exact_recipe_evidence.authorizes_exact_recipes")
    if item["matmul_kernel_equivalence"] != MATMUL_KERNEL_EQUIVALENCE:
        raise LockValidationError("$.exact_recipe_evidence.matmul_kernel_equivalence is unsupported")
    for name in fields - {
        "schema_version",
        "bank_policy_version",
        "authorizes_exact_recipes",
        "matmul_kernel_equivalence",
    }:
        _hex(item[name], 64, f"$.exact_recipe_evidence.{name}")
        if set(item[name]) == {"0"}:
            raise LockValidationError(f"$.exact_recipe_evidence.{name} must be nonzero")
    bound_root = {
        "bank_entry_inventory_sha256": direct_bank_entry_inventory_hash(lock),
        "semantic_source_sha256": lock["semantic_source_sha256"],
        "exact_entry_inventory_sha256": _sha256_value(_exact_entry_inventory(lock)),
        "exact_native_support_sha256": exact_native_support_hash(lock),
        "safety_evidence_sha256": exact_recipe_safety_inventory_hash(lock),
    }
    for name, expected in bound_root.items():
        if item[name] != expected:
            raise LockValidationError(f"$.exact_recipe_evidence.{name} binding mismatch")
    for index, entry in enumerate(lock.get("program_config_exact_entries", [])):
        if entry["bank_evidence"]["source_sha256"] != item["bank_artifact_sha256"]:
            raise LockValidationError(
                f"$.program_config_exact_entries[{index}].bank_evidence.source_sha256 bank binding mismatch"
            )
    if item["proof_sha256"] != exact_recipe_evidence_hash(item):
        raise LockValidationError("$.exact_recipe_evidence.proof_sha256 mismatch")
    return item


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
    _uint(item["architecture"], 32, f"{path}.architecture", positive=True)
    _uint(item["board_capability_class"], 32, f"{path}.board_capability_class")
    for name in ("device_count", "mesh_rows", "mesh_cols", "compute_grid_x", "compute_grid_y"):
        _uint(item[name], 16, f"{path}.{name}", positive=True)
    _hex(item["topology_sha256"], 64, f"{path}.topology_sha256")
    if item["architecture"] != 3:
        raise LockValidationError(f"{path}.architecture must be BLACKHOLE (3) for direct-bank scope")
    if item["board_capability_class"] != 0:
        raise LockValidationError(f"{path}.board_capability_class must be zero for direct-bank wildcard scope")
    if set(item["topology_sha256"]) != {"0"}:
        raise LockValidationError(f"{path}.topology_sha256 must be zero for direct-bank wildcard scope")
    if (item["device_count"], item["mesh_rows"], item["mesh_cols"]) != (1, 1, 1):
        raise LockValidationError(f"{path} direct-bank exact entries are limited to one-chip 1x1 scope")
    for name in ("transpose_a", "transpose_b", "has_bias", "has_activation", "untilize_out", "run_batched"):
        _boolean(item[name], f"{path}.{name}", False)
    if domain == "dense.addmm":
        _uint(item["alpha_f32_bits"], 32, f"{path}.alpha_f32_bits")
        _uint(item["beta_f32_bits"], 32, f"{path}.beta_f32_bits")
        if item["alpha_f32_bits"] != 0x3F800000:
            raise LockValidationError(f"{path}.alpha_f32_bits must encode exactly 1.0 in v1")
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
    if any(tensor[axis] != 32 for tensor in (input_a, input_b) for axis in ("tile_height", "tile_width")):
        raise LockValidationError(f"{path}.input_a and {path}.input_b must use 32x32 tiles in v1")
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
    if n_tiles != program["per_core_n"]:
        raise LockValidationError(f"{path}.per_core_n must equal padded N tiles for multi_core_reuse")
    if program["per_core_m"] % program["out_subblock_h"] != 0:
        raise LockValidationError(f"{path}.out_subblock_h must divide per_core_m")
    if program["per_core_n"] % program["out_subblock_w"] != 0:
        raise LockValidationError(f"{path}.out_subblock_w must divide per_core_n")


def program_config_exact_entry_id(entry: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "artifact_kind": "ttnn_matmul_program_config_exact_entry",
                "schema_version": 1,
                "domain": entry["domain"],
                "key": entry["key"],
                "program_config": entry["program_config"],
                "compute_kernel_config": entry["compute_kernel_config"],
            }
        )
    ).hexdigest()


def _validate_program_config_for_key(key: dict[str, Any], program: dict[str, Any], path: str) -> None:
    if program["compute_grid_x"] > key["compute_grid_x"] or program["compute_grid_y"] > key["compute_grid_y"]:
        raise LockValidationError(f"{path} compute grid exceeds the attested device grid")
    input_a, input_b = key["input_a"], key["input_b"]
    dimensions = (
        (key["padded_m"], input_a["tile_height"], "padded_m"),
        (key["padded_k"], input_a["tile_width"], "padded_k/input_a"),
        (key["padded_k"], input_b["tile_height"], "padded_k/input_b"),
        (key["padded_n"], input_b["tile_width"], "padded_n"),
    )
    if any(dimension % tile for dimension, tile, _ in dimensions):
        raise LockValidationError(f"{path} padded dimensions must divide their tile axes")
    m_tiles = key["padded_m"] // input_a["tile_height"]
    a_k_tiles = key["padded_k"] // input_a["tile_width"]
    b_k_tiles = key["padded_k"] // input_b["tile_height"]
    n_tiles = key["padded_n"] // input_b["tile_width"]
    if a_k_tiles != b_k_tiles or a_k_tiles % program["in0_block_w"]:
        raise LockValidationError(f"{path}.in0_block_w is incompatible with padded K")
    family = program["family"]
    if family == "multi_core_reuse":
        _validate_multi_core_reuse_work_split(key, program, path)
        return
    m_blocks = (m_tiles + program["per_core_m"] - 1) // program["per_core_m"]
    n_blocks = (n_tiles + program["per_core_n"] - 1) // program["per_core_n"]
    if family == "multi_cast_1d":
        complete_axis = program["per_core_m"] == m_tiles if program["mcast_in0"] else program["per_core_n"] == n_tiles
        core_count = program["compute_grid_x"] * program["compute_grid_y"]
        if program["per_core_n"] > 64 or not complete_axis or not n_blocks or m_blocks > core_count // n_blocks:
            raise LockValidationError(f"{path} is not a legal multi_cast_1d work split")
        return
    fits_grid = (
        m_blocks <= program["compute_grid_x"] and n_blocks <= program["compute_grid_y"]
        if program["transpose_mcast"]
        else m_blocks <= program["compute_grid_y"] and n_blocks <= program["compute_grid_x"]
    )
    if not fits_grid:
        raise LockValidationError(f"{path} is not a legal multi_cast_2d work split")


def _program_config_exact_entry(value: Any, path: str) -> dict[str, Any]:
    item = _exact_fields(
        value,
        {"bank_evidence", "compute_kernel_config", "domain", "entry_id", "key", "program_config"},
        path,
    )
    if item["domain"] not in {"dense.matmul", "dense.linear", "dense.addmm"}:
        raise LockValidationError(f"{path}.domain is unsupported")
    _hex(item["entry_id"], 64, f"{path}.entry_id")
    key = _key(item["key"], f"{path}.key", item["domain"])
    program = _program_config(item["program_config"], f"{path}.program_config")
    compute_kernel_config = _compute_kernel_config(item["compute_kernel_config"], f"{path}.compute_kernel_config")
    _validate_program_config_for_key(key, program, f"{path}.program_config")
    maximum_subblock_area = 4 if compute_kernel_config["fp32_dest_acc_en"] else 8
    if program["out_subblock_h"] * program["out_subblock_w"] > maximum_subblock_area:
        raise LockValidationError(
            f"{path}.program_config output subblock exceeds the paired compute-kernel destination-register bound"
        )
    _direct_bank_evidence(
        item["bank_evidence"],
        domain=item["domain"],
        key=key,
        program_config=program,
        compute_kernel_config=compute_kernel_config,
        path=f"{path}.bank_evidence",
    )
    item["key"] = key
    item["program_config"] = program
    item["compute_kernel_config"] = compute_kernel_config
    if item["entry_id"] != program_config_exact_entry_id(item):
        raise LockValidationError(f"{path}.entry_id mismatch")
    return item


def content_sha256(lock: dict[str, Any]) -> str:
    unsigned = dict(lock)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def validate_lock(value: Any) -> dict[str, Any]:
    fields = {
        "artifact_kind",
        "content_sha256",
        "key_schema_version",
        "lock_schema_version",
        "policy_version",
        "producer",
        "semantic_source_sha256",
        "program_config_exact_entries",
        "exact_recipe_evidence",
    }
    if not isinstance(value, dict):
        raise LockValidationError("$ must be an object")
    actual_fields = set(value)
    optional_fields: set[str] = set()
    if not fields <= actual_fields or actual_fields - fields - optional_fields:
        missing = sorted(fields - actual_fields)
        unknown = sorted(actual_fields - fields - optional_fields)
        raise LockValidationError(f"$ field mismatch: missing={missing}, unknown={unknown}")
    lock = value
    if lock["artifact_kind"] != ARTIFACT_KIND:
        raise LockValidationError("$.artifact_kind is unsupported")
    if lock["lock_schema_version"] != LOCK_SCHEMA_VERSION or lock["key_schema_version"] != KEY_SCHEMA_VERSION:
        raise LockValidationError("lock schema version is unsupported")
    policy_version = _string(lock["policy_version"], "$.policy_version")
    if policy_version != POLICY_VERSION:
        raise LockValidationError("$.policy_version is unsupported")
    for name in ("semantic_source_sha256", "content_sha256"):
        _hex(lock[name], 64, f"$.{name}")
    if lock["content_sha256"] != content_sha256(lock):
        raise LockValidationError("$.content_sha256 mismatch")
    producer = _exact_fields(
        lock["producer"], {"codegen_commit", "generator_version", "registry_abi_tt_metal_commit"}, "$.producer"
    )
    _hex(producer["codegen_commit"], 40, "$.producer.codegen_commit")
    _hex(producer["registry_abi_tt_metal_commit"], 40, "$.producer.registry_abi_tt_metal_commit")
    if producer["generator_version"] != GENERATOR_VERSION:
        raise LockValidationError("$.producer.generator_version is unsupported")
    program_config_exact_entries = lock.get("program_config_exact_entries", [])
    if not isinstance(program_config_exact_entries, list) or len(program_config_exact_entries) > MAX_ENTRIES:
        raise LockValidationError("$.program_config_exact_entries must be a bounded array")
    if program_config_exact_entries:
        provenance_values = (
            lock["semantic_source_sha256"],
            producer["codegen_commit"],
            producer["registry_abi_tt_metal_commit"],
        )
        if any(set(item) == {"0"} for item in provenance_values):
            raise LockValidationError("nonempty locks require measured provenance digests")
    ids: set[str] = set()
    prior_pc_key: bytes | None = None
    pc_keys: set[bytes | tuple[Any, ...]] = set()
    for index, raw_entry in enumerate(program_config_exact_entries):
        path = f"$.program_config_exact_entries[{index}]"
        item = _program_config_exact_entry(raw_entry, path)
        canonical_key_bytes = canonical_bytes({"domain": item["domain"], "key": item["key"]})
        lookup_key = _direct_bank_compact_key_sort_key(item)
        if prior_pc_key is not None and canonical_key_bytes < prior_pc_key:
            raise LockValidationError("$.program_config_exact_entries are not sorted by canonical key")
        if lookup_key in pc_keys:
            raise LockValidationError(f"{path} duplicates a program-config exact key")
        if item["entry_id"] in ids:
            raise LockValidationError(f"{path} duplicates an entry_id")
        prior_pc_key = canonical_key_bytes
        pc_keys.add(lookup_key)
        ids.add(item["entry_id"])
        program_config_exact_entries[index] = item
    evidence = _exact_recipe_evidence(lock.get("exact_recipe_evidence"), lock)
    if evidence is not None and evidence["authorizes_exact_recipes"] and not program_config_exact_entries:
        raise LockValidationError("exact-entry authorization requires nonempty program_config_exact_entries")
    if program_config_exact_entries and (evidence is None or not evidence["authorizes_exact_recipes"]):
        raise LockValidationError("program_config_exact_entries require explicit bound exact-entry authorization")
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


def _compute_kernel_config_cpp(value: dict[str, Any]) -> str:
    fidelity = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi3": "HiFi3", "hifi4": "HiFi4"}[value["math_fidelity"]]
    throttle = {
        "no_throttle": "NoThrottle",
        "throttle_1": "Throttle1",
        "throttle_2": "Throttle2",
        "throttle_3": "Throttle3",
        "throttle_4": "Throttle4",
        "throttle_5": "Throttle5",
    }[value["throttle_level"]]
    return (
        "compact::ComputeKernelDescriptor{"
        f".math_fidelity = compact::MathFidelity::{fidelity}, "
        f".throttle_level = compact::ThrottleLevel::{throttle}, "
        f".math_approx_mode = {str(value['math_approx_mode']).lower()}, "
        f".fp32_dest_acc_en = {str(value['fp32_dest_acc_en']).lower()}, "
        f".packer_l1_acc = {str(value['packer_l1_acc']).lower()}, "
        f".dst_full_sync_en = {str(value['dst_full_sync_en']).lower()}"
        "}"
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


def _direct_bank_compact_key_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    normalized = dict(item)
    normalized["key"] = dict(item["key"])
    normalized["key"]["board_capability_class"] = 0
    normalized["key"]["topology_sha256"] = "0" * 64
    return _compact_key_sort_key(normalized)


def _program_config_exact_entry_cpp(item: dict[str, Any]) -> str:
    key = item["key"]
    program = item["program_config"]
    ckc = item["compute_kernel_config"]
    domain = {"dense.matmul": "DenseMatmul", "dense.linear": "DenseLinear", "dense.addmm": "DenseAddmm"}[item["domain"]]
    family = {
        "multi_core_reuse": "MultiCoreReuse",
        "multi_cast_1d": "MultiCast1D",
        "multi_cast_2d": "MultiCast2D",
    }[program["family"]]

    def boolean(value: bool) -> str:
        return "true" if value else "false"

    return f"""compact::ProgramConfigExactEntry{{
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
    .program_config = compact::ProgramConfigDescriptor{{
        .family = compact::ProgramFamily::{family},
        .compute_grid_x = {program["compute_grid_x"]}, .compute_grid_y = {program["compute_grid_y"]},
        .in0_block_w = {program["in0_block_w"]}, .out_subblock_h = {program["out_subblock_h"]},
        .out_subblock_w = {program["out_subblock_w"]}, .per_core_m = {program["per_core_m"]},
        .per_core_n = {program["per_core_n"]}, .out_block_h = {program["out_block_h"]},
        .out_block_w = {program["out_block_w"]},
        .num_global_cb_receivers = {program["num_global_cb_receivers"]},
        .allowed_worker_cores_present = false, .fuse_batch = {boolean(program["fuse_batch"])},
        .mcast_in0 = {boolean(program["mcast_in0"])},
        .transpose_mcast = {boolean(program["transpose_mcast"])},
        .fused_activation_present = false, .gather_in0 = false, .hop_cores_present = false,
        .untilize_out = false, .stream_in1 = false,
    }},
    .compute_kernel_config = {_compute_kernel_config_cpp(ckc)},
}}"""


def emit(lock: dict[str, Any]) -> tuple[bytes, bytes]:
    checked = validate_lock(lock)
    exact_recipe_evidence = _exact_recipe_evidence(checked.get("exact_recipe_evidence"), checked)
    header = """// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Generated by emit_cpp.py. Do not edit.
#pragma once
#include <span>
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_exact.hpp"
namespace ttnn::operations::matmul::registry::generated {
const compact::TableMetadata& metadata() noexcept;
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept;
}  // namespace ttnn::operations::matmul::registry::generated
"""
    # Lock review order is canonical JSON byte order. Runtime order follows the
    # POD's numeric/defaulted comparison, which is deliberately different for
    # values such as integer 2 versus 10.
    runtime_pc_exact_entries = sorted(
        checked.get("program_config_exact_entries", []), key=_direct_bank_compact_key_sort_key
    )
    pc_exact_entry_text = ",\n".join(_program_config_exact_entry_cpp(entry) for entry in runtime_pc_exact_entries)
    evidence_schema_version = exact_recipe_evidence["schema_version"] if exact_recipe_evidence else 0
    equivalence_schema_version = (
        exact_recipe_evidence["matmul_kernel_equivalence"]["schema_version"] if exact_recipe_evidence else 0
    )
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
    // Direct schema 2 owns the complete native tuning pair. Caller-supplied
    // program/CKC/core-grid state bypasses registry selection before lookup.
    .exact_recipe_evidence_schema_version = {evidence_schema_version if exact_recipe_evidence is not None and exact_recipe_evidence["authorizes_exact_recipes"] else 0},
    .matmul_kernel_equivalence_schema_version = {equivalence_schema_version if exact_recipe_evidence is not None and exact_recipe_evidence["authorizes_exact_recipes"] else 0},
    .content_sha256 = {_bytes_cpp(checked["content_sha256"])},
    .semantic_source_sha256 = {_bytes_cpp(checked["semantic_source_sha256"])},
}};
constexpr std::array<compact::ProgramConfigExactEntry, {len(runtime_pc_exact_entries)}> kProgramConfigExactEntries{{{{
{pc_exact_entry_text}
}}}};
constexpr bool program_config_exact_keys_are_strictly_sorted() {{
    for (std::size_t index = 1; index < kProgramConfigExactEntries.size(); ++index) {{
        const auto left = compact::direct_bank_key(kProgramConfigExactEntries[index - 1].key);
        const auto right = compact::direct_bank_key(kProgramConfigExactEntries[index].key);
        if (!(left < right)) {{
            return false;
        }}
    }}
    return true;
}}
static_assert(program_config_exact_keys_are_strictly_sorted());
}}  // namespace
const compact::TableMetadata& metadata() noexcept {{ return kMetadata; }}
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept {{
    return kProgramConfigExactEntries;
}}
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
