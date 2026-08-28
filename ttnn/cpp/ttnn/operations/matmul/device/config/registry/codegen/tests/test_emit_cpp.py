# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

CODEGEN_DIR = Path(__file__).resolve().parents[1]
REGISTRY_DIR = CODEGEN_DIR.parent
EMITTER_PATH = CODEGEN_DIR / "emit_cpp.py"
CHECKED_IN_LOCK_PATH = REGISTRY_DIR / "matmul_registry.lock.json"
CHECKED_IN_GENERATED_DIR = REGISTRY_DIR / "checked_in"

SPEC = importlib.util.spec_from_file_location("matmul_registry_emit_cpp", EMITTER_PATH)
assert SPEC is not None and SPEC.loader is not None
emitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(emitter)

ZERO = "0" * 64
BANK_SHA = "d" * 64


def _seed() -> dict:
    tensor = {
        "buffer_type": "dram",
        "dtype": "bfloat16",
        "layout": "tile",
        "memory_layout": "interleaved",
        "tile_height": 32,
        "tile_width": 32,
    }
    return {
        "compute_kernel_config": {
            "dst_full_sync_en": False,
            "fp32_dest_acc_en": False,
            "math_approx_mode": True,
            "math_fidelity": "hifi2",
            "packer_l1_acc": False,
            "throttle_level": "no_throttle",
        },
        "domain": "dense.matmul",
        "key": {
            "alpha_f32_bits": None,
            "architecture": 3,
            "bcast_batch": None,
            "beta_f32_bits": None,
            "board_capability_class": 0,
            "codegen_recipe_abi": emitter.CODEGEN_RECIPE_ABI,
            "compute_grid_x": 8,
            "compute_grid_y": 8,
            "device_count": 1,
            "has_activation": False,
            "has_bias": False,
            "input_a": copy.deepcopy(tensor),
            "input_b": copy.deepcopy(tensor),
            "logical_k": 256,
            "logical_m": 128,
            "logical_n": 512,
            "mesh_cols": 1,
            "mesh_rows": 1,
            "output": copy.deepcopy(tensor),
            "padded_k": 256,
            "padded_m": 128,
            "padded_n": 512,
            "run_batched": False,
            "schema_version": 1,
            "topology_sha256": ZERO,
            "transpose_a": False,
            "transpose_b": False,
            "untilize_out": False,
        },
        "producer": {
            "codegen_commit": "a" * 40,
            "generator_version": emitter.GENERATOR_VERSION,
            "registry_abi_tt_metal_commit": "b" * 40,
        },
        "program_config": {
            "allowed_worker_cores": None,
            "compute_grid_x": 8,
            "compute_grid_y": 8,
            "family": "multi_core_reuse",
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "per_core_m": 4,
            "per_core_n": 16,
        },
        "semantic_source_sha256": "1" * 64,
    }


def _program(seed: dict, family: str) -> dict:
    replay = seed["program_config"]
    program = {
        "allowed_worker_cores": None,
        "compute_grid_x": replay["compute_grid_x"],
        "compute_grid_y": replay["compute_grid_y"],
        "family": family,
        "fused_activation_present": False,
        "fuse_batch": False,
        "gather_in0": False,
        "hop_cores_present": False,
        "in0_block_w": replay["in0_block_w"],
        "mcast_in0": False,
        "num_global_cb_receivers": 0,
        "out_block_h": 0,
        "out_block_w": 0,
        "out_subblock_h": replay["out_subblock_h"],
        "out_subblock_w": replay["out_subblock_w"],
        "per_core_m": replay["per_core_m"],
        "per_core_n": replay["per_core_n"],
        "stream_in1": False,
        "transpose_mcast": False,
        "untilize_out": False,
    }
    key = seed["key"]
    if family == "multi_cast_1d":
        program.update(
            fuse_batch=True,
            mcast_in0=True,
            num_global_cb_receivers=1,
            out_block_h=key["padded_m"] // 32,
            out_block_w=2,
            per_core_m=key["padded_m"] // 32,
            per_core_n=2,
        )
    elif family == "multi_cast_2d":
        program.update(
            fuse_batch=True,
            out_block_h=1,
            out_block_w=2,
            per_core_m=1,
            per_core_n=2,
            transpose_mcast=True,
        )
    return program


def _bank_evidence(entry: dict) -> dict:
    return {
        "lookup_key_sha256": emitter._sha256_value({"domain": entry["domain"], "key": entry["key"]}),
        "native_recipe_sha256": emitter._sha256_value(
            {
                "program_config": entry["program_config"],
                "compute_kernel_config": entry["compute_kernel_config"],
            }
        ),
        "policy_version": emitter.DIRECT_BANK_EVIDENCE_POLICY_VERSION,
        "schema_version": 2,
        "source_sha256": BANK_SHA,
    }


def _seal(lock: dict) -> dict:
    evidence = lock["exact_recipe_evidence"]
    evidence["authorizes_exact_recipes"] = bool(lock["program_config_exact_entries"])
    evidence["bank_entry_inventory_sha256"] = emitter.direct_bank_entry_inventory_hash(lock)
    evidence["exact_entry_inventory_sha256"] = emitter._sha256_value(emitter._exact_entry_inventory(lock))
    evidence["exact_native_support_sha256"] = emitter.exact_native_support_hash(lock)
    evidence["safety_evidence_sha256"] = emitter.exact_recipe_safety_inventory_hash(lock)
    evidence["proof_sha256"] = emitter.exact_recipe_evidence_hash(evidence)
    lock["content_sha256"] = emitter.content_sha256(lock)
    return lock


def direct_lock(family: str = "multi_core_reuse", *, include_exact: bool = True) -> dict:
    seed = _seed()
    key = copy.deepcopy(seed["key"])
    entry = {
        "bank_evidence": {},
        "compute_kernel_config": copy.deepcopy(seed["compute_kernel_config"]),
        "domain": seed["domain"],
        "entry_id": ZERO,
        "key": key,
        "program_config": _program(seed, family),
    }
    entry["entry_id"] = emitter.program_config_exact_entry_id(entry)
    entry["bank_evidence"] = _bank_evidence(entry)
    lock = {
        "artifact_kind": emitter.ARTIFACT_KIND,
        "content_sha256": ZERO,
        "key_schema_version": emitter.KEY_SCHEMA_VERSION,
        "lock_schema_version": emitter.LOCK_SCHEMA_VERSION,
        "policy_version": emitter.POLICY_VERSION,
        "producer": copy.deepcopy(seed["producer"]),
        "program_config_exact_entries": [entry] if include_exact else [],
        "exact_recipe_evidence": {
            "authorizes_exact_recipes": include_exact,
            "bank_artifact_sha256": BANK_SHA,
            "bank_entry_inventory_sha256": ZERO,
            "bank_policy_version": emitter.DIRECT_BANK_EVIDENCE_POLICY_VERSION,
            "exact_entry_inventory_sha256": ZERO,
            "exact_native_support_sha256": ZERO,
            "matmul_kernel_equivalence": copy.deepcopy(emitter.MATMUL_KERNEL_EQUIVALENCE),
            "proof_sha256": ZERO,
            "safety_evidence_sha256": ZERO,
            "schema_version": 2,
            "semantic_source_sha256": seed["semantic_source_sha256"],
        },
        "semantic_source_sha256": seed["semantic_source_sha256"],
    }
    return _seal(lock)


def _compile_generated(tmp_path: Path, lock: dict, probe_text: str) -> None:
    header, source = emitter.emit(lock)
    (tmp_path / "matmul_registry_data.hpp").write_bytes(header)
    (tmp_path / "matmul_registry_data.cpp").write_bytes(source)
    include_dir = tmp_path / "ttnn" / "operations" / "matmul" / "device" / "config" / "registry"
    include_dir.mkdir(parents=True)
    shutil.copyfile(REGISTRY_DIR / "matmul_registry_exact.hpp", include_dir / "matmul_registry_exact.hpp")
    shutil.copyfile(REGISTRY_DIR / "matmul_registry_descriptor.hpp", include_dir / "matmul_registry_descriptor.hpp")
    (tmp_path / "probe.cpp").write_text(probe_text, encoding="utf-8")
    subprocess.run(
        ["c++", "-std=c++20", "-I.", "matmul_registry_data.cpp", "probe.cpp", "-o", "probe"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    subprocess.run(["./probe"], check=True, capture_output=True, text=True, cwd=tmp_path)


def test_direct_lock_emits_deterministically_and_cli_is_reproducible(tmp_path: Path) -> None:
    lock = emitter.validate_lock(direct_lock())
    first = emitter.emit(copy.deepcopy(lock))
    assert first == emitter.emit(copy.deepcopy(lock))
    assert b"EntryDescriptor" not in first[0] + first[1]
    assert b"ProgramConfigExactEntry, 1" in first[1]
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(emitter.canonical_bytes(lock) + b"\n")
    emitter.main(
        [
            "--lock",
            str(lock_path),
            "--header",
            str(tmp_path / "generated.hpp"),
            "--source",
            str(tmp_path / "generated.cpp"),
        ]
    )
    assert (tmp_path / "generated.hpp").read_bytes() == first[0]
    assert (tmp_path / "generated.cpp").read_bytes() == first[1]


def test_checked_in_registry_snapshot_is_fresh() -> None:
    expected_header, expected_source = emitter.emit(emitter.load_lock(CHECKED_IN_LOCK_PATH))
    assert (CHECKED_IN_GENERATED_DIR / "matmul_registry_data.hpp").read_bytes() == expected_header
    assert (CHECKED_IN_GENERATED_DIR / "matmul_registry_data.cpp").read_bytes() == expected_source


def test_optional_compatibility_manifest_is_strict_and_emitted(expect_error) -> None:
    lock = direct_lock()
    lock["compatibility"] = {
        "build_identity_sha256": "2" * 64,
        "runtime_capability_sha256": "3" * 64,
        "schema_version": 1,
    }
    checked = emitter.validate_lock(_seal(lock))
    _, source = emitter.emit(checked)
    assert b".compatibility_schema_version = 1" in source
    assert b"0x22, 0x22" in source
    assert b"0x33, 0x33" in source

    lock = direct_lock()
    lock["compatibility"] = {
        "build_identity_sha256": ZERO,
        "runtime_capability_sha256": "3" * 64,
        "schema_version": 1,
    }
    with expect_error(emitter.LockValidationError, "must be nonzero"):
        emitter.validate_lock(_seal(lock))


@pytest.mark.parametrize("family", ["multi_core_reuse", "multi_cast_1d", "multi_cast_2d"])
def test_exact_table_supports_every_native_family_as_a_paired_recipe(family: str) -> None:
    lock = emitter.validate_lock(direct_lock(family))
    _, source = emitter.emit(lock)
    assert b"ComputeKernelDescriptor" in source
    assert {
        "multi_core_reuse": b"ProgramFamily::MultiCoreReuse",
        "multi_cast_1d": b"ProgramFamily::MultiCast1D",
        "multi_cast_2d": b"ProgramFamily::MultiCast2D",
    }[family] in source


def test_exact_generated_runtime_compiles_and_looks_up(tmp_path: Path) -> None:
    lock = emitter.validate_lock(direct_lock())
    _compile_generated(
        tmp_path,
        lock,
        r"""#include "matmul_registry_data.hpp"
int main() {
    namespace compact = ttnn::operations::matmul::registry::compact;
    namespace generated = ttnn::operations::matmul::registry::generated;
    const auto entries = generated::program_config_exact_entries();
    if (entries.size() != 1) return 1;
    const auto* exact = compact::lookup_program_config_exact(entries.front().key, entries);
    return exact != nullptr && compact::legal_program_config_candidate(
        exact->key, {.program_config = exact->program_config, .compute_kernel_config = exact->compute_kernel_config})
        ? 0 : 2;
}
""",
    )


def test_exact_grid_cohorts_are_distinct_and_sorted(expect_error) -> None:
    lock = direct_lock()
    first = lock["program_config_exact_entries"][0]
    first["key"]["compute_grid_x"] = 12
    first["entry_id"] = emitter.program_config_exact_entry_id(first)
    first["bank_evidence"] = _bank_evidence(first)
    second = copy.deepcopy(first)
    second["key"]["compute_grid_x"] = 13
    second["entry_id"] = emitter.program_config_exact_entry_id(second)
    second["bank_evidence"] = _bank_evidence(second)
    lock["program_config_exact_entries"] = sorted(
        [second, first], key=lambda item: emitter.canonical_bytes({"domain": item["domain"], "key": item["key"]})
    )
    _seal(lock)
    checked = emitter.validate_lock(lock)
    assert [item["key"]["compute_grid_x"] for item in checked["program_config_exact_entries"]] == [12, 13]

    duplicate = copy.deepcopy(lock)
    duplicate["program_config_exact_entries"].append(copy.deepcopy(duplicate["program_config_exact_entries"][0]))
    duplicate["program_config_exact_entries"].sort(
        key=lambda item: emitter.canonical_bytes({"domain": item["domain"], "key": item["key"]})
    )
    _seal(duplicate)
    with expect_error(emitter.LockValidationError, "duplicates a program-config exact key"):
        emitter.validate_lock(duplicate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lock: lock["program_config_exact_entries"][0]["bank_evidence"].__setitem__(
            "lookup_key_sha256", "e" * 64
        ),
        lambda lock: lock["program_config_exact_entries"][0]["bank_evidence"].__setitem__(
            "native_recipe_sha256", "e" * 64
        ),
        lambda lock: lock["program_config_exact_entries"][0]["compute_kernel_config"].__setitem__(
            "math_fidelity", "hifi4"
        ),
        lambda lock: lock["exact_recipe_evidence"].__setitem__("semantic_source_sha256", "e" * 64),
        lambda lock: lock["exact_recipe_evidence"]["matmul_kernel_equivalence"].__setitem__(
            "policy_id", "unreviewed-equivalence"
        ),
    ],
)
def test_exact_or_evidence_tamper_fails_closed(mutation, expect_error) -> None:
    lock = direct_lock()
    mutation(lock)
    lock["content_sha256"] = emitter.content_sha256(lock)
    with expect_error(emitter.LockValidationError, "."):
        emitter.validate_lock(lock)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda lock: lock.__setitem__("policy_version", "matmul-promotion-v2"), "policy_version is unsupported"),
        (lambda lock: lock.__setitem__("entries", []), "field mismatch"),
        (lambda lock: lock.__setitem__("online_program_config_models", []), "field mismatch"),
        (lambda lock: lock.__setitem__("unknown", True), "field mismatch"),
    ],
)
def test_non_direct_or_runtime_identity_surface_is_rejected(mutation, message: str, expect_error) -> None:
    lock = direct_lock()
    mutation(lock)
    lock["content_sha256"] = emitter.content_sha256(lock)
    with expect_error(emitter.LockValidationError, message):
        emitter.validate_lock(lock)


def test_runtime_model_field_is_rejected(expect_error) -> None:
    lock = direct_lock()
    lock["online_program_config_models"] = [{}]
    lock["content_sha256"] = emitter.content_sha256(lock)
    with expect_error(emitter.LockValidationError, "field mismatch"):
        emitter.validate_lock(lock)


def test_public_domains_are_disjoint_and_addmm_scalars_are_exact(expect_error) -> None:
    keys = []
    for domain in ("dense.matmul", "dense.linear", "dense.addmm"):
        lock = direct_lock()
        entry = lock["program_config_exact_entries"][0]
        entry["domain"] = domain
        if domain == "dense.addmm":
            entry["key"]["alpha_f32_bits"] = 0x3F800000
            entry["key"]["beta_f32_bits"] = 0
        entry["entry_id"] = emitter.program_config_exact_entry_id(entry)
        entry["bank_evidence"] = _bank_evidence(entry)
        _seal(lock)
        checked = emitter.validate_lock(lock)
        keys.append(
            emitter.canonical_bytes(
                {
                    "domain": checked["program_config_exact_entries"][0]["domain"],
                    "key": checked["program_config_exact_entries"][0]["key"],
                }
            )
        )
    assert len(set(keys)) == 3

    bad = direct_lock()
    bad["program_config_exact_entries"][0]["key"]["alpha_f32_bits"] = 0x3F800000
    bad["content_sha256"] = emitter.content_sha256(bad)
    with expect_error(emitter.LockValidationError, "exclusive to dense.addmm"):
        emitter.validate_lock(bad)


def test_load_lock_rejects_duplicate_members_and_noncanonical_bytes(tmp_path: Path, expect_error) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"artifact_kind":"x","artifact_kind":"y"}', encoding="utf-8")
    with expect_error(emitter.LockValidationError, "duplicate JSON key"):
        emitter.load_lock(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(direct_lock(), indent=2), encoding="utf-8")
    with expect_error(emitter.LockValidationError, "not canonical"):
        emitter.load_lock(noncanonical)


def test_empty_direct_bank_emits_typed_empty_tables() -> None:
    lock = emitter.validate_lock(direct_lock(include_exact=False))
    header, source = emitter.emit(lock)
    assert b"ProgramConfigExactEntry, 0" in source
    assert b"ProgramConfigGbdtModel" not in source
    assert b"program_config_exact_entries()" in header
