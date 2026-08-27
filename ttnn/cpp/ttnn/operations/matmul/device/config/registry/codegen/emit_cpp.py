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
POLICY_VERSION = "matmul-direct-bank-v1"
LEGACY_POLICY_VERSION = "matmul-promotion-v2"
DIRECT_BANK_EVIDENCE_POLICY_VERSION = "deterministic-matmul-bank-v1"
LOCK_SCHEMA_VERSION = 1
KEY_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 2
CODEGEN_RECIPE_ABI = 1
ONLINE_MODEL_SCHEMA_VERSION = 1
MAX_ENTRIES = 4096
MAX_MODEL_CANDIDATES = 4096
MAX_MODEL_TREES = 256
MAX_MODEL_NODES = 65535
MAX_MODEL_TRAINING_SHAPES = 65535
MAX_MODEL_SHAPE_DISTANCE_PPM = 250_000
MAX_LOCK_BYTES = 32 * 1024 * 1024
BASELINE_POLICY_ID = "dense-ttnn-auto-v1"
MIN_CERTIFICATE_SESSIONS = 10
MIN_CERTIFICATE_BLOCKS_PER_SESSION = 20
MIN_CERTIFICATE_SPEEDUP_PPM = 1_030_000
MIN_CERTIFICATE_OPERATIONAL_LOWER_BOUND_PPM = 1_000_000
MIN_CERTIFICATE_PCC_PPB = 990_000_000
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


PROGRAM_CONFIG_FEATURES = (
    "logical_m",
    "logical_k",
    "logical_n",
    "padded_m",
    "padded_k",
    "padded_n",
    "grid_x",
    "grid_y",
    "in0_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "per_core_m",
    "per_core_n",
    "family",
    "out_block_h",
    "out_block_w",
    "num_global_cb_receivers",
    "fuse_batch",
    "mcast_in0",
    "transpose_mcast",
    "fused_activation_present",
    "gather_in0",
    "hop_cores_present",
    "untilize_out",
    "stream_in1",
)
FEATURE_SCHEMA_PREIMAGE = {
    "artifact_kind": "ttnn_matmul_online_program_config_feature_schema",
    "features": list(PROGRAM_CONFIG_FEATURES),
    "numeric_encoding": "uint64_raw_v1",
    "schema_version": 1,
}
FEATURE_SCHEMA_SHA256 = hashlib.sha256(canonical_bytes(FEATURE_SCHEMA_PREIMAGE)).hexdigest()


def _int64(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -(1 << 63) <= value < 1 << 63:
        raise LockValidationError(f"{path} must be an int64")
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
    if item["out_subblock_h"] * item["out_subblock_w"] > 4:
        raise LockValidationError(f"{path} output subblock exceeds the conservative runtime bound")
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


def program_config_candidate_id(program_config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes({"program_config": program_config})).hexdigest()


def _model_support(value: Any, path: str, *, direct_bank: bool = False) -> dict[str, Any]:
    fields = {
        "architecture",
        "board_capability_class",
        "device_count",
        "domain",
        "input_a",
        "input_b",
        "maximum_k",
        "maximum_m",
        "maximum_n",
        "mesh_cols",
        "mesh_rows",
        "minimum_k",
        "minimum_m",
        "minimum_n",
        "output",
        "shape_geometry",
        "shape_scale",
        "topology_sha256",
    }
    item = _exact_fields(value, fields, path)
    _uint(item["architecture"], 32, f"{path}.architecture", positive=True)
    _uint(item["board_capability_class"], 32, f"{path}.board_capability_class", positive=not direct_bank)
    for name in ("device_count", "mesh_rows", "mesh_cols"):
        _uint(item[name], 16, f"{path}.{name}", positive=True)
    _hex(item["topology_sha256"], 64, f"{path}.topology_sha256")
    if direct_bank:
        if item["board_capability_class"] != 0:
            raise LockValidationError(f"{path}.board_capability_class must be zero for direct-bank wildcard scope")
        if set(item["topology_sha256"]) != {"0"}:
            raise LockValidationError(f"{path}.topology_sha256 must be zero for direct-bank wildcard scope")
        if (item["device_count"], item["mesh_rows"], item["mesh_cols"]) != (1, 1, 1):
            raise LockValidationError(f"{path} direct-bank support is limited to one-chip 1x1 scope")
    if item["domain"] not in {"dense.matmul", "dense.linear", "dense.addmm"}:
        raise LockValidationError(f"{path}.domain is unsupported")
    for name in ("input_a", "input_b", "output"):
        tensor = _tensor(item[name], f"{path}.{name}")
        if (
            tensor["layout"] != "tile"
            or tensor["memory_layout"] != "interleaved"
            or tensor["buffer_type"] != "dram"
            or tensor["tile_height"] != 32
            or tensor["tile_width"] != 32
        ):
            raise LockValidationError(f"{path}.{name} must be DRAM-interleaved tile32")
    if item["shape_scale"] not in {"decode", "small_batch", "prefill", "long_prefill"}:
        raise LockValidationError(f"{path}.shape_scale is unsupported")
    if item["shape_geometry"] not in {"contract_wide", "square_kn", "output_wide"}:
        raise LockValidationError(f"{path}.shape_geometry is unsupported")
    for axis in ("m", "k", "n"):
        minimum = _uint(item[f"minimum_{axis}"], 64, f"{path}.minimum_{axis}", positive=True)
        maximum = _uint(item[f"maximum_{axis}"], 64, f"{path}.maximum_{axis}", positive=True)
        if maximum < minimum:
            raise LockValidationError(f"{path} {axis.upper()} support bounds are reversed")
    return item


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def online_model_hash(model: dict[str, Any]) -> str:
    payload = {key: value for key, value in model.items() if key not in {"model_sha256", "bundle_binding_sha256"}}
    return _sha256_value(payload)


def online_models_bundle_binding(lock: dict[str, Any], models: list[dict[str, Any]]) -> str:
    model_inventory = []
    for model in models:
        model_inventory.append(
            {
                "feature_schema_sha256": model["feature_schema_sha256"],
                "model_sha256": model["model_sha256"],
                "training_table_sha256": model["training_table_sha256"],
                "safety_evidence_sha256": model["safety_evidence_sha256"],
                "candidate_policy_sha256": model["candidate_policy_sha256"],
                "lineage_sha256": model["lineage_sha256"],
                "evaluation_model_payload_sha256": model["evaluation_model_payload_sha256"],
                "quality_evaluation_sha256": model["quality_evaluation_sha256"],
                "unseen_abstention_policy_sha256": model["unseen_abstention_policy_sha256"],
                "minimum_score_margin": model["minimum_score_margin"],
                "maximum_normalized_shape_distance_ppm": model["maximum_normalized_shape_distance_ppm"],
                "training_shape_inventory_sha256": _sha256_value(model["training_shapes"]),
                "support_sha256": model["support_sha256"],
                "candidate_inventory_sha256": _sha256_value([item["candidate_id"] for item in model["candidates"]]),
                "tree_inventory_sha256": _sha256_value(model["trees"]),
                "node_inventory_sha256": _sha256_value(model["nodes"]),
            }
        )
    preimage = {
        "artifact_kind": "ttnn_matmul_online_program_config_bundle_binding",
        "schema_version": 1,
        "exact_entry_inventory_sha256": _sha256_value(_exact_entry_inventory(lock)),
        "model_inventory": model_inventory,
    }
    return _sha256_value(preimage)


def online_model_training_table_inventory_hash(models: list[dict[str, Any]]) -> str:
    """Bind each retained model to its own immutable fit table.

    Exact entries may advance to a newer bank independently of a previously
    accepted model. Keeping this as a distinct root preserves both lineages.
    """

    return _sha256_value(
        [
            {
                "model_sha256": model["model_sha256"],
                "training_table_sha256": model["training_table_sha256"],
            }
            for model in models
        ]
    )


def program_config_only_evidence_hash(evidence: dict[str, Any]) -> str:
    payload = {key: value for key, value in evidence.items() if key != "proof_sha256"}
    return _sha256_value(payload)


def direct_bank_entry_inventory_hash(lock: dict[str, Any]) -> str:
    return _sha256_value(
        [
            {
                "bank_evidence": entry["bank_evidence"],
                "entry_id": entry["entry_id"],
                "table_kind": "program_config_only",
            }
            for entry in lock.get("program_config_exact_entries", [])
        ]
    )


def _direct_bank_evidence(
    value: Any, *, domain: str, key: dict[str, Any], program_config: dict[str, Any], path: str
) -> dict[str, Any]:
    fields = {"lookup_key_sha256", "policy_version", "program_config_sha256", "schema_version", "source_sha256"}
    item = _exact_fields(value, fields, path)
    if item["schema_version"] != 1 or item["policy_version"] != DIRECT_BANK_EVIDENCE_POLICY_VERSION:
        raise LockValidationError(f"{path} direct-bank policy/schema is unsupported")
    for name in ("lookup_key_sha256", "program_config_sha256", "source_sha256"):
        _hex(item[name], 64, f"{path}.{name}")
        if set(item[name]) == {"0"}:
            raise LockValidationError(f"{path}.{name} must be nonzero")
    if item["lookup_key_sha256"] != _sha256_value({"domain": domain, "key": key}):
        raise LockValidationError(f"{path}.lookup_key_sha256 mismatch")
    if item["program_config_sha256"] != _sha256_value(program_config):
        raise LockValidationError(f"{path}.program_config_sha256 mismatch")
    return item


def exact_native_support_hash(lock: dict[str, Any]) -> str:
    inventory = [
        {
            "table_kind": "legacy_replay",
            "entry_id": entry["entry_id"],
            "native_support": {
                "architecture": entry["key"]["architecture"],
                "board_capability_class": entry["key"]["board_capability_class"],
                "device_count": entry["key"]["device_count"],
                "domain": entry["domain"],
                "input_a": entry["key"]["input_a"],
                "input_b": entry["key"]["input_b"],
                "mesh_cols": entry["key"]["mesh_cols"],
                "mesh_rows": entry["key"]["mesh_rows"],
                "output": entry["key"]["output"],
                "topology_sha256": entry["key"]["topology_sha256"],
            },
        }
        for entry in lock["entries"]
    ]
    direct_bank = lock["policy_version"] == POLICY_VERSION
    inventory.extend(
        {
            "table_kind": "program_config_only",
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
                **(
                    {}
                    if direct_bank
                    else {
                        "board_capability_class": entry["key"]["board_capability_class"],
                        "topology_sha256": entry["key"]["topology_sha256"],
                    }
                ),
            },
        }
        for entry in lock.get("program_config_exact_entries", [])
    )
    return _sha256_value(sorted(inventory, key=lambda item: (item["table_kind"], item["entry_id"])))


def _exact_entry_inventory(lock: dict[str, Any]) -> list[dict[str, str]]:
    return sorted(
        [{"table_kind": "legacy_replay", "entry_id": entry["entry_id"]} for entry in lock["entries"]]
        + [
            {"table_kind": "program_config_only", "entry_id": entry["entry_id"]}
            for entry in lock.get("program_config_exact_entries", [])
        ],
        key=lambda item: (item["table_kind"], item["entry_id"]),
    )


def program_config_safety_inventory_hash(lock: dict[str, Any], online_models: list[dict[str, Any]]) -> str:
    """Bind every exact-entry and online-model safety artifact into one proof root."""

    preimage = {
        "artifact_kind": "ttnn_matmul_program_config_safety_inventory",
        "schema_version": 1,
        "exact_entry_inventory": sorted(
            (
                [
                    {
                        "table_kind": "legacy_replay",
                        "entry_id": entry["entry_id"],
                        "evidence_sha256": entry["certificate"]["evidence_sha256"],
                    }
                    for entry in lock["entries"]
                ]
                + [
                    {
                        "table_kind": "program_config_only",
                        "entry_id": entry["entry_id"],
                        "evidence_sha256": (
                            entry["bank_evidence"]["source_sha256"]
                            if "bank_evidence" in entry
                            else entry["certificate"]["evidence_sha256"]
                        ),
                    }
                    for entry in lock.get("program_config_exact_entries", [])
                ]
            ),
            key=lambda item: (item["table_kind"], item["entry_id"]),
        ),
        "online_model_inventory": [
            {
                "model_sha256": model["model_sha256"],
                "safety_evidence_sha256": model["safety_evidence_sha256"],
            }
            for model in online_models
        ],
    }
    return _sha256_value(preimage)


def _program_config_only_evidence(
    value: Any, lock: dict[str, Any], online_models: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if lock["policy_version"] == POLICY_VERSION:
        fields = {
            "authorizes_exact_entries",
            "bank_artifact_sha256",
            "bank_entry_inventory_sha256",
            "bank_policy_version",
            "build_identity_sha256",
            "exact_entry_inventory_sha256",
            "exact_native_support_sha256",
            "online_model_bundle_binding_sha256",
            "online_model_training_table_inventory_sha256",
            "proof_sha256",
            "safety_evidence_sha256",
            "schema_version",
            "semantic_source_sha256",
        }
        item = _exact_fields(value, fields, "$.program_config_only_evidence")
        if item["schema_version"] != 2 or item["bank_policy_version"] != DIRECT_BANK_EVIDENCE_POLICY_VERSION:
            raise LockValidationError("$.program_config_only_evidence direct-bank policy/schema is unsupported")
        _boolean(item["authorizes_exact_entries"], "$.program_config_only_evidence.authorizes_exact_entries")
        for name in fields - {"schema_version", "bank_policy_version", "authorizes_exact_entries"}:
            _hex(item[name], 64, f"$.program_config_only_evidence.{name}")
            if set(item[name]) == {"0"} and not (name == "online_model_bundle_binding_sha256" and not online_models):
                raise LockValidationError(f"$.program_config_only_evidence.{name} must be nonzero")
        bound_root = {
            "bank_entry_inventory_sha256": direct_bank_entry_inventory_hash(lock),
            "build_identity_sha256": lock["build_identity_sha256"],
            "semantic_source_sha256": lock["semantic_source_sha256"],
            "exact_entry_inventory_sha256": _sha256_value(_exact_entry_inventory(lock)),
            "exact_native_support_sha256": exact_native_support_hash(lock),
            "online_model_bundle_binding_sha256": (
                online_models[0]["bundle_binding_sha256"] if online_models else "0" * 64
            ),
            "online_model_training_table_inventory_sha256": online_model_training_table_inventory_hash(online_models),
            "safety_evidence_sha256": program_config_safety_inventory_hash(lock, online_models),
        }
        for name, expected in bound_root.items():
            if item[name] != expected:
                raise LockValidationError(f"$.program_config_only_evidence.{name} binding mismatch")
        for index, entry in enumerate(lock.get("program_config_exact_entries", [])):
            if entry["bank_evidence"]["source_sha256"] != item["bank_artifact_sha256"]:
                raise LockValidationError(
                    f"$.program_config_exact_entries[{index}].bank_evidence.source_sha256 bank binding mismatch"
                )
        if item["proof_sha256"] != program_config_only_evidence_hash(item):
            raise LockValidationError("$.program_config_only_evidence.proof_sha256 mismatch")
        return item
    fields = {
        "build_identity_sha256",
        "authorizes_exact_entries",
        "compute_kernel_config_mode",
        "effective_default_attestation_sha256",
        "effective_default_ckc_inventory_sha256",
        "exact_entry_inventory_sha256",
        "exact_native_support_sha256",
        "fresh_confirmation_sha256",
        "measured_tt_metal_commit",
        "native_parity_sha256",
        "online_model_bundle_binding_sha256",
        "proof_sha256",
        "runtime_capability_sha256",
        "safety_evidence_sha256",
        "schema_version",
        "semantic_source_sha256",
        "throttle_policy_sha256",
    }
    item = _exact_fields(value, fields, "$.program_config_only_evidence")
    if item["schema_version"] != 1 or item["compute_kernel_config_mode"] != "op_default":
        raise LockValidationError("$.program_config_only_evidence mode/schema is unsupported")
    _boolean(item["authorizes_exact_entries"], "$.program_config_only_evidence.authorizes_exact_entries")
    for name in fields - {
        "schema_version",
        "compute_kernel_config_mode",
        "measured_tt_metal_commit",
        "authorizes_exact_entries",
    }:
        _hex(item[name], 64, f"$.program_config_only_evidence.{name}")
        if set(item[name]) == {"0"} and not (name == "online_model_bundle_binding_sha256" and not online_models):
            raise LockValidationError(f"$.program_config_only_evidence.{name} must be nonzero")
    _hex(item["measured_tt_metal_commit"], 40, "$.program_config_only_evidence.measured_tt_metal_commit")
    bound_root = {
        "build_identity_sha256": lock["build_identity_sha256"],
        "runtime_capability_sha256": lock["runtime_capability_sha256"],
        "semantic_source_sha256": lock["semantic_source_sha256"],
        "measured_tt_metal_commit": lock["producer"]["measured_tt_metal_commit"],
        "exact_entry_inventory_sha256": _sha256_value(_exact_entry_inventory(lock)),
        "exact_native_support_sha256": exact_native_support_hash(lock),
        "online_model_bundle_binding_sha256": (
            online_models[0]["bundle_binding_sha256"] if online_models else "0" * 64
        ),
        "safety_evidence_sha256": program_config_safety_inventory_hash(lock, online_models),
    }
    for name, expected in bound_root.items():
        if item[name] != expected:
            raise LockValidationError(f"$.program_config_only_evidence.{name} binding mismatch")
    if item["proof_sha256"] != program_config_only_evidence_hash(item):
        raise LockValidationError("$.program_config_only_evidence.proof_sha256 mismatch")
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


def _key(value: Any, path: str, domain: str, *, direct_bank: bool = False) -> dict[str, Any]:
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
    _uint(item["board_capability_class"], 32, f"{path}.board_capability_class", positive=not direct_bank)
    for name in ("device_count", "mesh_rows", "mesh_cols", "compute_grid_x", "compute_grid_y"):
        _uint(item[name], 16, f"{path}.{name}", positive=True)
    _hex(item["topology_sha256"], 64, f"{path}.topology_sha256")
    if direct_bank:
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
    _validate_multi_core_reuse_work_split(key, program, f"{path}.program_config")
    maximum_subblock_area = 4 if ckc["fp32_dest_acc_en"] else 8
    if program["out_subblock_h"] * program["out_subblock_w"] > maximum_subblock_area:
        raise LockValidationError(f"{path}.program_config output subblock exceeds the destination-register bound")

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
    minima = {
        "baseline_calls": MIN_CERTIFICATE_SESSIONS * MIN_CERTIFICATE_BLOCKS_PER_SESSION,
        "baseline_ns": 1,
        "baseline_sessions": MIN_CERTIFICATE_SESSIONS,
        "candidate_calls": MIN_CERTIFICATE_SESSIONS * MIN_CERTIFICATE_BLOCKS_PER_SESSION,
        "candidate_ns": 1,
        "candidate_sessions": MIN_CERTIFICATE_SESSIONS,
        "operational_lower_bound_ppm": MIN_CERTIFICATE_OPERATIONAL_LOWER_BOUND_PPM + 1,
        "pcc_min_ppb": MIN_CERTIFICATE_PCC_PPB,
        "speedup_ppm": MIN_CERTIFICATE_SPEEDUP_PPM,
    }
    for name, minimum in minima.items():
        _uint(item[name], 64, f"{path}.{name}")
        if item[name] < minimum:
            raise LockValidationError(f"{path}.{name} is below the production minimum {minimum}")
    if item["pcc_min_ppb"] > 1_000_000_000:
        raise LockValidationError(f"{path}.pcc_min_ppb exceeds one")
    _string(item["baseline_policy_id"], f"{path}.baseline_policy_id")
    if item["baseline_policy_id"] != BASELINE_POLICY_ID:
        raise LockValidationError(f"{path}.baseline_policy_id is not the production baseline")
    _hex(item["evidence_sha256"], 64, f"{path}.evidence_sha256")


def program_config_exact_entry_id(entry: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "artifact_kind": "ttnn_matmul_program_config_exact_entry",
                "schema_version": 1,
                "domain": entry["domain"],
                "key": entry["key"],
                "program_config": entry["program_config"],
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


def _program_config_exact_entry(value: Any, path: str, policy_version: str) -> dict[str, Any]:
    evidence_field = "bank_evidence" if policy_version == POLICY_VERSION else "certificate"
    item = _exact_fields(value, {evidence_field, "domain", "entry_id", "key", "program_config"}, path)
    if item["domain"] not in {"dense.matmul", "dense.linear", "dense.addmm"}:
        raise LockValidationError(f"{path}.domain is unsupported")
    _hex(item["entry_id"], 64, f"{path}.entry_id")
    key = _key(item["key"], f"{path}.key", item["domain"], direct_bank=policy_version == POLICY_VERSION)
    program = _program_config(item["program_config"], f"{path}.program_config")
    _validate_program_config_for_key(key, program, f"{path}.program_config")
    if policy_version == POLICY_VERSION:
        _direct_bank_evidence(
            item["bank_evidence"],
            domain=item["domain"],
            key=key,
            program_config=program,
            path=f"{path}.bank_evidence",
        )
    else:
        _certificate(item["certificate"], f"{path}.certificate")
    item["key"] = key
    item["program_config"] = program
    if item["entry_id"] != program_config_exact_entry_id(item):
        raise LockValidationError(f"{path}.entry_id mismatch")
    return item


def _online_program_config_model(value: Any, lock: dict[str, Any]) -> dict[str, Any]:
    if value is None:
        return {"enabled": False, "schema_version": ONLINE_MODEL_SCHEMA_VERSION}
    if not isinstance(value, dict):
        raise LockValidationError("$.online_program_config_model must be an object")
    if value.get("enabled") is False:
        item = _exact_fields(value, {"enabled", "schema_version"}, "$.online_program_config_model")
        if item["schema_version"] != ONLINE_MODEL_SCHEMA_VERSION:
            raise LockValidationError("$.online_program_config_model schema is unsupported")
        return item

    fields = {
        "base_score",
        "bundle_binding_sha256",
        "candidate_policy_sha256",
        "candidates",
        "enabled",
        "evaluation_model_payload_sha256",
        "feature_schema_sha256",
        "lineage_sha256",
        "minimum_score_margin",
        "maximum_normalized_shape_distance_ppm",
        "model_sha256",
        "nodes",
        "quality_evaluation_sha256",
        "safety_evidence_sha256",
        "schema_version",
        "score_orientation",
        "score_scale",
        "support",
        "support_sha256",
        "training_table_sha256",
        "training_shapes",
        "trees",
        "unseen_abstention_policy_sha256",
    }
    item = _exact_fields(value, fields, "$.online_program_config_model")
    if item["enabled"] is not True or item["schema_version"] != ONLINE_MODEL_SCHEMA_VERSION:
        raise LockValidationError("$.online_program_config_model active schema is unsupported")
    if item["score_orientation"] != "lower_is_better_negated_pairwise_margin":
        raise LockValidationError("$.online_program_config_model score orientation is unsupported")
    _int64(item["base_score"], "$.online_program_config_model.base_score")
    _uint(item["score_scale"], 32, "$.online_program_config_model.score_scale", positive=True)
    _uint(item["minimum_score_margin"], 64, "$.online_program_config_model.minimum_score_margin", positive=True)
    maximum_distance = _uint(
        item["maximum_normalized_shape_distance_ppm"],
        64,
        "$.online_program_config_model.maximum_normalized_shape_distance_ppm",
        positive=True,
    )
    if maximum_distance > MAX_MODEL_SHAPE_DISTANCE_PPM:
        raise LockValidationError("$.online_program_config_model.maximum_normalized_shape_distance_ppm exceeds 250000")
    for name in (
        "bundle_binding_sha256",
        "candidate_policy_sha256",
        "evaluation_model_payload_sha256",
        "feature_schema_sha256",
        "lineage_sha256",
        "model_sha256",
        "quality_evaluation_sha256",
        "safety_evidence_sha256",
        "support_sha256",
        "training_table_sha256",
        "unseen_abstention_policy_sha256",
    ):
        _hex(item[name], 64, f"$.online_program_config_model.{name}")
        if set(item[name]) == {"0"}:
            raise LockValidationError(f"$.online_program_config_model.{name} must be nonzero")
    if item["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256:
        raise LockValidationError("$.online_program_config_model.feature_schema_sha256 mismatch")

    support = _model_support(
        item["support"],
        "$.online_program_config_model.support",
        direct_bank=lock["policy_version"] == POLICY_VERSION,
    )
    if item["support_sha256"] != _sha256_value(support):
        raise LockValidationError("$.online_program_config_model.support_sha256 mismatch")

    training_shapes = item["training_shapes"]
    if not isinstance(training_shapes, list) or not training_shapes or len(training_shapes) > MAX_MODEL_TRAINING_SHAPES:
        raise LockValidationError("$.online_program_config_model.training_shapes must be a nonempty bounded array")
    previous_shape: tuple[int, int, int] | None = None
    for index, shape in enumerate(training_shapes):
        path = f"$.online_program_config_model.training_shapes[{index}]"
        if not isinstance(shape, list) or len(shape) != 3:
            raise LockValidationError(f"{path} must be an M/K/N triple")
        normalized = tuple(_uint(value, 64, f"{path}[{axis}]", positive=True) for axis, value in enumerate(shape))
        if previous_shape is not None and not previous_shape < normalized:
            raise LockValidationError("$.online_program_config_model.training_shapes must be strictly sorted")
        if not (
            support["minimum_m"] <= normalized[0] <= support["maximum_m"]
            and support["minimum_k"] <= normalized[1] <= support["maximum_k"]
            and support["minimum_n"] <= normalized[2] <= support["maximum_n"]
        ):
            raise LockValidationError(f"{path} is outside model support bounds")
        previous_shape = normalized

    candidates = item["candidates"]
    if not isinstance(candidates, list) or not candidates or len(candidates) > MAX_MODEL_CANDIDATES:
        raise LockValidationError("$.online_program_config_model.candidates must be a nonempty bounded array")
    prior_program: tuple[Any, ...] | None = None
    candidate_ids: set[str] = set()
    for index, candidate_value in enumerate(candidates):
        path = f"$.online_program_config_model.candidates[{index}]"
        candidate = _exact_fields(candidate_value, {"candidate_id", "program_config"}, path)
        _hex(candidate["candidate_id"], 64, f"{path}.candidate_id")
        program = _program_config(candidate["program_config"], f"{path}.program_config")
        expected_id = program_config_candidate_id(program)
        if candidate["candidate_id"] != expected_id:
            raise LockValidationError(f"{path}.candidate_id mismatch")
        sort_key = _program_config_sort_key(program)
        if prior_program is not None and not prior_program < sort_key:
            raise LockValidationError("$.online_program_config_model.candidates must be strictly sorted and unique")
        if candidate["candidate_id"] in candidate_ids:
            raise LockValidationError(f"{path} duplicates candidate_id")
        prior_program = sort_key
        candidate_ids.add(candidate["candidate_id"])

    nodes = item["nodes"]
    if not isinstance(nodes, list) or not nodes or len(nodes) > MAX_MODEL_NODES:
        raise LockValidationError("$.online_program_config_model.nodes must be a nonempty bounded array")
    for index, node_value in enumerate(nodes):
        path = f"$.online_program_config_model.nodes[{index}]"
        node = _exact_fields(node_value, {"feature", "leaf_value", "left", "right", "threshold"}, path)
        if node["feature"] == "leaf":
            if any(node[name] != 0 for name in ("threshold", "left", "right")):
                raise LockValidationError(f"{path} leaf branch fields must be zero")
            _int64(node["leaf_value"], f"{path}.leaf_value")
        else:
            if node["feature"] not in PROGRAM_CONFIG_FEATURES:
                raise LockValidationError(f"{path}.feature is unsupported")
            _uint(node["threshold"], 64, f"{path}.threshold")
            _uint(node["left"], 32, f"{path}.left")
            _uint(node["right"], 32, f"{path}.right")
            if node["leaf_value"] != 0:
                raise LockValidationError(f"{path} branch leaf_value must be zero")

    trees = item["trees"]
    if not isinstance(trees, list) or not trees or len(trees) > MAX_MODEL_TREES:
        raise LockValidationError("$.online_program_config_model.trees must be a nonempty bounded array")
    expected_offset = 0
    minimum_total_score = item["base_score"]
    maximum_total_score = item["base_score"]
    for tree_index, tree_value in enumerate(trees):
        path = f"$.online_program_config_model.trees[{tree_index}]"
        tree = _exact_fields(tree_value, {"node_count", "node_offset"}, path)
        offset = _uint(tree["node_offset"], 32, f"{path}.node_offset")
        count = _uint(tree["node_count"], 32, f"{path}.node_count", positive=True)
        if offset != expected_offset or count > len(nodes) - offset:
            raise LockValidationError("$.online_program_config_model trees must contiguously partition nodes")
        visited: set[int] = set()
        active: set[int] = set()

        def visit(relative: int) -> None:
            if relative >= count:
                raise LockValidationError(f"{path} child index escapes its tree")
            if relative in active:
                raise LockValidationError(f"{path} contains a cycle")
            if relative in visited:
                raise LockValidationError(f"{path} node has multiple parents")
            active.add(relative)
            node = nodes[offset + relative]
            if node["feature"] != "leaf":
                visit(node["left"])
                visit(node["right"])
            active.remove(relative)
            visited.add(relative)

        visit(0)
        if len(visited) != count:
            raise LockValidationError(f"{path} contains unreachable nodes")
        leaf_values = [
            nodes[offset + relative]["leaf_value"]
            for relative in visited
            if nodes[offset + relative]["feature"] == "leaf"
        ]
        if not leaf_values:
            raise LockValidationError(f"{path} has no leaf")
        minimum_total_score += min(leaf_values)
        maximum_total_score += max(leaf_values)
        expected_offset += count
    if expected_offset != len(nodes):
        raise LockValidationError("$.online_program_config_model trees do not consume all nodes")
    if not -(1 << 63) <= minimum_total_score < 1 << 63 or not (-(1 << 63) <= maximum_total_score < 1 << 63):
        raise LockValidationError("$.online_program_config_model fixed-point score envelope exceeds int64")

    if item["model_sha256"] != online_model_hash(item):
        raise LockValidationError("$.online_program_config_model.model_sha256 mismatch")
    return item


def _online_program_config_models(value: Any, lock: dict[str, Any]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise LockValidationError("$.online_program_config_models must be a bounded array")
    models = [_online_program_config_model(item, lock) for item in value]
    if any(not model["enabled"] for model in models):
        raise LockValidationError("$.online_program_config_models must not contain disabled entries")
    prior_support: bytes | None = None
    for index, model in enumerate(models):
        support_bytes = canonical_bytes(model["support"])
        if prior_support is not None and not prior_support < support_bytes:
            raise LockValidationError("$.online_program_config_models must be strictly support-sorted")
        prior_support = support_bytes
        for previous in models[:index]:
            left = previous["support"]
            right = model["support"]
            context_fields = set(left) - {
                "minimum_m",
                "maximum_m",
                "minimum_k",
                "maximum_k",
                "minimum_n",
                "maximum_n",
            }
            if all(left[field] == right[field] for field in context_fields):
                overlaps = all(
                    left[f"minimum_{axis}"] <= right[f"maximum_{axis}"]
                    and right[f"minimum_{axis}"] <= left[f"maximum_{axis}"]
                    for axis in ("m", "k", "n")
                )
                if overlaps:
                    raise LockValidationError("$.online_program_config_models have overlapping support")
    binding = online_models_bundle_binding(lock, models) if models else "0" * 64
    for index, model in enumerate(models):
        if model["bundle_binding_sha256"] != binding:
            raise LockValidationError(f"$.online_program_config_models[{index}].bundle_binding_sha256 mismatch")
    return models


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
    if not isinstance(value, dict):
        raise LockValidationError("$ must be an object")
    actual_fields = set(value)
    optional_fields = {
        "online_program_config_models",
        "program_config_exact_entries",
        "program_config_only_evidence",
    }
    if not fields <= actual_fields or actual_fields - fields - optional_fields:
        missing = sorted(fields - actual_fields)
        unknown = sorted(actual_fields - fields - optional_fields)
        raise LockValidationError(f"$ field mismatch: missing={missing}, unknown={unknown}")
    lock = value
    if lock["artifact_kind"] != ARTIFACT_KIND:
        raise LockValidationError("$.artifact_kind is unsupported")
    if (
        lock["lock_schema_version"] != LOCK_SCHEMA_VERSION
        or lock["key_schema_version"] != KEY_SCHEMA_VERSION
        or lock["replay_schema_version"] != REPLAY_SCHEMA_VERSION
    ):
        raise LockValidationError("lock schema version is unsupported")
    policy_version = _string(lock["policy_version"], "$.policy_version")
    if policy_version not in {POLICY_VERSION, LEGACY_POLICY_VERSION}:
        raise LockValidationError("$.policy_version is unsupported")
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
    program_config_exact_entries = lock.get("program_config_exact_entries", [])
    if not isinstance(program_config_exact_entries, list) or len(program_config_exact_entries) > MAX_ENTRIES:
        raise LockValidationError("$.program_config_exact_entries must be a bounded array")
    if policy_version == POLICY_VERSION and entries:
        raise LockValidationError("direct-bank locks must not contain legacy replay entries")
    if entries or program_config_exact_entries:
        compatibility_values = (
            lock["semantic_source_sha256"],
            lock["build_identity_sha256"],
            producer["codegen_commit"],
            producer["measured_tt_metal_commit"],
        )
        if policy_version == LEGACY_POLICY_VERSION:
            compatibility_values += (lock["runtime_capability_sha256"],)
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
    prior_pc_key: bytes | None = None
    pc_keys: set[bytes] = set()
    for index, raw_entry in enumerate(program_config_exact_entries):
        path = f"$.program_config_exact_entries[{index}]"
        item = _program_config_exact_entry(raw_entry, path, policy_version)
        key_bytes = canonical_bytes({"domain": item["domain"], "key": item["key"]})
        if prior_pc_key is not None and key_bytes < prior_pc_key:
            raise LockValidationError("$.program_config_exact_entries are not sorted by canonical key")
        if key_bytes in pc_keys:
            raise LockValidationError(f"{path} duplicates a program-config exact key")
        if item["entry_id"] in ids:
            raise LockValidationError(f"{path} duplicates an entry_id")
        prior_pc_key = key_bytes
        pc_keys.add(key_bytes)
        ids.add(item["entry_id"])
        program_config_exact_entries[index] = item
    online_models = _online_program_config_models(lock.get("online_program_config_models"), lock)
    evidence = _program_config_only_evidence(lock.get("program_config_only_evidence"), lock, online_models)
    if online_models and evidence is None:
        raise LockValidationError("enabled online models require bound program_config_only_evidence")
    if evidence is not None and evidence["authorizes_exact_entries"] and not program_config_exact_entries:
        raise LockValidationError("exact-entry authorization requires nonempty program_config_exact_entries")
    if program_config_exact_entries and (evidence is None or not evidence["authorizes_exact_entries"]):
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


def _program_config_exact_entry_cpp(item: dict[str, Any]) -> str:
    key = item["key"]
    program = item["program_config"]
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
}}"""


def _online_model_cpp(model: dict[str, Any], index: int) -> str:
    family_names = {
        "multi_core_reuse": "MultiCoreReuse",
        "multi_cast_1d": "MultiCast1D",
        "multi_cast_2d": "MultiCast2D",
    }
    feature_names = {
        "logical_m": "LogicalM",
        "logical_k": "LogicalK",
        "logical_n": "LogicalN",
        "padded_m": "PaddedM",
        "padded_k": "PaddedK",
        "padded_n": "PaddedN",
        "grid_x": "GridX",
        "grid_y": "GridY",
        "in0_block_w": "In0BlockW",
        "out_subblock_h": "OutSubblockH",
        "out_subblock_w": "OutSubblockW",
        "per_core_m": "PerCoreM",
        "per_core_n": "PerCoreN",
        "family": "Family",
        "out_block_h": "OutBlockH",
        "out_block_w": "OutBlockW",
        "num_global_cb_receivers": "NumGlobalCbReceivers",
        "fuse_batch": "FuseBatch",
        "mcast_in0": "McastIn0",
        "transpose_mcast": "TransposeMcast",
        "fused_activation_present": "FusedActivationPresent",
        "gather_in0": "GatherIn0",
        "hop_cores_present": "HopCoresPresent",
        "untilize_out": "UntilizeOut",
        "stream_in1": "StreamIn1",
        "leaf": "Count",
    }
    candidate_lines = []
    for item in model["candidates"]:
        program = item["program_config"]
        candidate_lines.append(
            "compact::ProgramConfigCandidate{"
            ".program_config = compact::ProgramConfigDescriptor{"
            f".family = compact::ProgramFamily::{family_names[program['family']]}, "
            f".compute_grid_x = {program['compute_grid_x']}, .compute_grid_y = {program['compute_grid_y']}, "
            f".in0_block_w = {program['in0_block_w']}, .out_subblock_h = {program['out_subblock_h']}, "
            f".out_subblock_w = {program['out_subblock_w']}, .per_core_m = {program['per_core_m']}, "
            f".per_core_n = {program['per_core_n']}, "
            f".out_block_h = {program['out_block_h']}, .out_block_w = {program['out_block_w']}, "
            f".num_global_cb_receivers = {program['num_global_cb_receivers']}, "
            ".allowed_worker_cores_present = false, "
            f".fuse_batch = {str(program['fuse_batch']).lower()}, "
            f".mcast_in0 = {str(program['mcast_in0']).lower()}, "
            f".transpose_mcast = {str(program['transpose_mcast']).lower()}, "
            f".fused_activation_present = {str(program['fused_activation_present']).lower()}, "
            f".gather_in0 = {str(program['gather_in0']).lower()}, "
            f".hop_cores_present = {str(program['hop_cores_present']).lower()}, "
            f".untilize_out = {str(program['untilize_out']).lower()}, "
            f".stream_in1 = {str(program['stream_in1']).lower()}"
            "}, "
            f".candidate_id = {_bytes_cpp(item['candidate_id'])}"
            "}"
        )
    node_lines = [
        "compact::GbdtNode{"
        f".feature = compact::ProgramConfigFeature::{feature_names[item['feature']]}, "
        f".threshold = {item['threshold']}ULL, .left = {item['left']}, .right = {item['right']}, "
        f".leaf_value = {item['leaf_value']}LL"
        "}"
        for item in model["nodes"]
    ]
    tree_lines = [
        f"compact::GbdtTree{{.node_offset = {item['node_offset']}, .node_count = {item['node_count']}}}"
        for item in model["trees"]
    ]
    training_shape_lines = [
        "compact::TrainingShapeLandmark{"
        f".logical_m = {shape[0]}ULL, .logical_k = {shape[1]}ULL, .logical_n = {shape[2]}ULL"
        "}"
        for shape in model["training_shapes"]
    ]
    support = model["support"]
    domain = {"dense.matmul": "DenseMatmul", "dense.linear": "DenseLinear", "dense.addmm": "DenseAddmm"}[
        support["domain"]
    ]
    scale = {
        "decode": "Decode",
        "small_batch": "SmallBatch",
        "prefill": "Prefill",
        "long_prefill": "LongPrefill",
    }[support["shape_scale"]]
    geometry = {
        "contract_wide": "ContractWide",
        "square_kn": "SquareKn",
        "output_wide": "OutputWide",
    }[support["shape_geometry"]]
    candidate_text = ",\n".join(candidate_lines)
    node_text = ",\n".join(node_lines)
    tree_text = ",\n".join(tree_lines)
    training_shape_text = ",\n".join(training_shape_lines)
    return f"""constexpr std::array<compact::ProgramConfigCandidate, {len(candidate_lines)}> kModelCandidates{index}{{{{
{candidate_text}
}}}};
constexpr std::array<compact::GbdtNode, {len(node_lines)}> kModelNodes{index}{{{{
{node_text}
}}}};
constexpr std::array<compact::GbdtTree, {len(tree_lines)}> kModelTrees{index}{{{{
{tree_text}
}}}};
constexpr std::array<compact::TrainingShapeLandmark, {len(training_shape_lines)}> kModelTrainingShapes{index}{{{{
{training_shape_text}
}}}};
constexpr compact::ProgramConfigGbdtModel kOnlineModel{index}{{
    .schema_version = {model["schema_version"]}, .enabled = true,
    .score_orientation = compact::GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin,
    .feature_schema_sha256 = {_bytes_cpp(model["feature_schema_sha256"])},
    .model_sha256 = {_bytes_cpp(model["model_sha256"])},
    .training_table_sha256 = {_bytes_cpp(model["training_table_sha256"])},
    .safety_evidence_sha256 = {_bytes_cpp(model["safety_evidence_sha256"])},
    .candidate_policy_sha256 = {_bytes_cpp(model["candidate_policy_sha256"])},
    .lineage_sha256 = {_bytes_cpp(model["lineage_sha256"])},
    .evaluation_model_payload_sha256 = {_bytes_cpp(model["evaluation_model_payload_sha256"])},
    .quality_evaluation_sha256 = {_bytes_cpp(model["quality_evaluation_sha256"])},
    .unseen_abstention_policy_sha256 = {_bytes_cpp(model["unseen_abstention_policy_sha256"])},
    .support_sha256 = {_bytes_cpp(model["support_sha256"])},
    .bundle_binding_sha256 = {_bytes_cpp(model["bundle_binding_sha256"])},
    .support = compact::ProgramConfigModelSupport{{
        .architecture = {support["architecture"]}, .board_capability_class = {support["board_capability_class"]},
        .device_count = {support["device_count"]}, .mesh_rows = {support["mesh_rows"]},
        .mesh_cols = {support["mesh_cols"]}, .topology_sha256 = {_bytes_cpp(support["topology_sha256"])},
        .domain = compact::Domain::{domain}, .input_a = {_tensor_cpp(support["input_a"])},
        .input_b = {_tensor_cpp(support["input_b"])}, .output = {_tensor_cpp(support["output"])},
        .shape_scale = compact::ShapeScaleClass::{scale},
        .shape_geometry = compact::ShapeGeometryClass::{geometry},
        .minimum_m = {support["minimum_m"]}ULL, .maximum_m = {support["maximum_m"]}ULL,
        .minimum_k = {support["minimum_k"]}ULL, .maximum_k = {support["maximum_k"]}ULL,
        .minimum_n = {support["minimum_n"]}ULL, .maximum_n = {support["maximum_n"]}ULL,
    }},
    .base_score = {model["base_score"]}LL, .score_scale = {model["score_scale"]},
    .minimum_score_margin = {model["minimum_score_margin"]}ULL,
    .maximum_normalized_shape_distance_ppm = {model["maximum_normalized_shape_distance_ppm"]}ULL,
    .training_shapes = kModelTrainingShapes{index},
    .candidates = kModelCandidates{index}, .trees = kModelTrees{index}, .nodes = kModelNodes{index},
}};"""


def emit(lock: dict[str, Any]) -> tuple[bytes, bytes]:
    checked = validate_lock(lock)
    online_models = _online_program_config_models(checked.get("online_program_config_models"), checked)
    program_config_only_evidence = _program_config_only_evidence(
        checked.get("program_config_only_evidence"), checked, online_models
    )
    header = """// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Generated by emit_cpp.py. Do not edit.
#pragma once
#include <span>
#include "ttnn/operations/matmul/device/config/registry/matmul_program_config_model.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"
namespace ttnn::operations::matmul::registry::generated {
const compact::TableMetadata& metadata() noexcept;
std::span<const compact::EntryDescriptor> entries() noexcept;
compact::ExactIndex index() noexcept;
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept;
std::span<const compact::ProgramConfigGbdtModel> online_models() noexcept;
}  // namespace ttnn::operations::matmul::registry::generated
"""
    # Lock review order is canonical JSON byte order. Runtime order follows the
    # POD's numeric/defaulted comparison, which is deliberately different for
    # values such as integer 2 versus 10.
    runtime_entries = sorted(checked["entries"], key=_compact_key_sort_key)
    entry_text = ",\n".join(_entry_cpp(entry) for entry in runtime_entries)
    runtime_pc_exact_entries = sorted(checked.get("program_config_exact_entries", []), key=_compact_key_sort_key)
    pc_exact_entry_text = ",\n".join(_program_config_exact_entry_cpp(entry) for entry in runtime_pc_exact_entries)
    model_declarations = "\n".join(_online_model_cpp(model, index) for index, model in enumerate(online_models))
    model_names = ", ".join(f"kOnlineModel{index}" for index in range(len(online_models)))
    online_model_text = (
        model_declarations
        + f"\nconstexpr std::array<compact::ProgramConfigGbdtModel, {len(online_models)}> kOnlineModels"
        + "{{"
        + model_names
        + "}};"
    )
    bundle_binding = online_models[0]["bundle_binding_sha256"] if online_models else "0" * 64
    model_training_tables = online_model_training_table_inventory_hash(online_models)
    evidence_schema_version = program_config_only_evidence["schema_version"] if program_config_only_evidence else 0
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
    // Schema 2 is deterministic direct-bank evidence. Legacy schema 1 remains
    // readable for old checked locks, but neither form owns caller CKC state.
    .program_config_only_evidence_schema_version = {evidence_schema_version if program_config_only_evidence is not None and program_config_only_evidence["authorizes_exact_entries"] else 0},
    .online_program_config_model_evidence_schema_version = {evidence_schema_version},
    .content_sha256 = {_bytes_cpp(checked["content_sha256"])},
    .semantic_source_sha256 = {_bytes_cpp(checked["semantic_source_sha256"])},
    .build_identity_sha256 = {_bytes_cpp(checked["build_identity_sha256"])},
    .runtime_capability_sha256 = {_bytes_cpp(checked["runtime_capability_sha256"])},
    .online_model_bundle_binding_sha256 = {_bytes_cpp(bundle_binding)},
    .online_model_training_table_inventory_sha256 = {_bytes_cpp(model_training_tables)},
}};
constexpr std::array<compact::EntryDescriptor, {len(checked["entries"])}> kEntries{{{{
{entry_text}
}}}};
constexpr std::array<compact::ProgramConfigExactEntry, {len(runtime_pc_exact_entries)}> kProgramConfigExactEntries{{{{
{pc_exact_entry_text}
}}}};
{online_model_text}
constexpr bool keys_are_strictly_sorted() {{
    for (std::size_t index = 1; index < kEntries.size(); ++index) {{
        if (!(kEntries[index - 1].key < kEntries[index].key)) {{
            return false;
        }}
    }}
    return true;
}}
static_assert(keys_are_strictly_sorted());
constexpr bool program_config_exact_keys_are_strictly_sorted() {{
    for (std::size_t index = 1; index < kProgramConfigExactEntries.size(); ++index) {{
        if (!(kProgramConfigExactEntries[index - 1].key < kProgramConfigExactEntries[index].key)) {{
            return false;
        }}
    }}
    return true;
}}
static_assert(program_config_exact_keys_are_strictly_sorted());
}}  // namespace
const compact::TableMetadata& metadata() noexcept {{ return kMetadata; }}
std::span<const compact::EntryDescriptor> entries() noexcept {{ return kEntries; }}
compact::ExactIndex index() noexcept {{ return compact::ExactIndex{{kEntries}}; }}
std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries() noexcept {{
    return kProgramConfigExactEntries;
}}
std::span<const compact::ProgramConfigGbdtModel> online_models() noexcept {{ return kOnlineModels; }}
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
