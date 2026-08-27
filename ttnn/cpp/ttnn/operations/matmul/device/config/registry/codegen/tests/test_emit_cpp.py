# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path

CODEGEN_DIR = Path(__file__).resolve().parents[1]
REGISTRY_DIR = CODEGEN_DIR.parent
EMITTER_PATH = CODEGEN_DIR / "emit_cpp.py"
FIXTURE_PATH = REGISTRY_DIR / "fixtures" / "valid_multi_core_reuse.lock.json"
EXPECTED_CONTENT_SHA256 = "c2e28f624f15358b46632995603cb682ae7f18703e7fae23bdb592163dd0a919"
EXPECTED_ENTRY_ID = "2e5422c0e502fec492a20472f0e7f4318095b6fd6cc39dfb0e9225ab6ca4d677"

SPEC = importlib.util.spec_from_file_location("matmul_registry_emit_cpp", EMITTER_PATH)
assert SPEC is not None and SPEC.loader is not None
emitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(emitter)


def fixture() -> dict:
    return copy.deepcopy(emitter.load_lock(FIXTURE_PATH))


def resign(lock: dict, *, entries: bool = False) -> None:
    if entries:
        for item in lock["entries"]:
            item["entry_id"] = emitter.entry_id(item)
    lock["content_sha256"] = emitter.content_sha256(lock)


def op_default_evidence(lock: dict) -> dict:
    evidence = {
        "authorizes_exact_entries": True,
        "build_identity_sha256": lock["build_identity_sha256"],
        "compute_kernel_config_mode": "op_default",
        "effective_default_attestation_sha256": "7" * 64,
        "effective_default_ckc_inventory_sha256": "8" * 64,
        "exact_entry_inventory_sha256": emitter._sha256_value(emitter._exact_entry_inventory(lock)),
        "exact_native_support_sha256": emitter.exact_native_support_hash(lock),
        "fresh_confirmation_sha256": "9" * 64,
        "measured_tt_metal_commit": lock["producer"]["measured_tt_metal_commit"],
        "native_parity_sha256": "a" * 64,
        "online_model_bundle_binding_sha256": "0" * 64,
        "proof_sha256": "0" * 64,
        "runtime_capability_sha256": lock["runtime_capability_sha256"],
        "safety_evidence_sha256": emitter.program_config_safety_inventory_hash(lock, []),
        "schema_version": 1,
        "semantic_source_sha256": lock["semantic_source_sha256"],
        "throttle_policy_sha256": "c" * 64,
    }
    evidence["proof_sha256"] = emitter.program_config_only_evidence_hash(evidence)
    return evidence


def add_program_config_exact_entry(lock: dict, family: str = "multi_core_reuse") -> dict:
    legacy = lock["entries"][0]
    recipe = legacy["recipe"]["program_config"]
    program = {
        "allowed_worker_cores": None,
        "compute_grid_x": recipe["compute_grid_x"],
        "compute_grid_y": recipe["compute_grid_y"],
        "family": family,
        "fused_activation_present": False,
        "fuse_batch": False,
        "gather_in0": False,
        "hop_cores_present": False,
        "in0_block_w": recipe["in0_block_w"],
        "mcast_in0": False,
        "num_global_cb_receivers": 0,
        "out_block_h": 0,
        "out_block_w": 0,
        "out_subblock_h": recipe["out_subblock_h"],
        "out_subblock_w": recipe["out_subblock_w"],
        "per_core_m": recipe["per_core_m"],
        "per_core_n": recipe["per_core_n"],
        "stream_in1": False,
        "transpose_mcast": False,
        "untilize_out": False,
    }
    if family == "multi_cast_1d":
        program.update(
            fuse_batch=True,
            mcast_in0=True,
            out_block_h=lock["entries"][0]["key"]["padded_m"] // 32,
            out_block_w=2,
            num_global_cb_receivers=1,
            per_core_m=lock["entries"][0]["key"]["padded_m"] // 32,
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
    entry = {
        "certificate": copy.deepcopy(legacy["certificate"]),
        "domain": legacy["domain"],
        "entry_id": "0" * 64,
        "key": copy.deepcopy(legacy["key"]),
        "program_config": program,
    }
    entry["entry_id"] = emitter.program_config_exact_entry_id(entry)
    lock["program_config_exact_entries"] = [entry]
    return entry


def convert_to_direct_bank_lock(lock: dict, *, include_exact: bool = True) -> dict:
    entry = add_program_config_exact_entry(lock) if include_exact else None
    lock["entries"] = []
    lock["policy_version"] = emitter.POLICY_VERSION
    lock["build_identity_sha256"] = "0" * 64
    lock["runtime_capability_sha256"] = "0" * 64
    bank_artifact_sha256 = "d" * 64
    if entry is not None:
        key = entry["key"]
        key["board_capability_class"] = 0
        key["device_count"] = 1
        key["mesh_rows"] = 1
        key["mesh_cols"] = 1
        key["topology_sha256"] = "0" * 64
        entry.pop("certificate")
        entry["entry_id"] = emitter.program_config_exact_entry_id(entry)
        entry["bank_evidence"] = {
            "lookup_key_sha256": emitter._sha256_value({"domain": entry["domain"], "key": entry["key"]}),
            "policy_version": emitter.DIRECT_BANK_EVIDENCE_POLICY_VERSION,
            "program_config_sha256": emitter._sha256_value(entry["program_config"]),
            "schema_version": 1,
            "source_sha256": bank_artifact_sha256,
        }
    else:
        lock["program_config_exact_entries"] = []

    models = lock.get("online_program_config_models", [])
    for model in models:
        model["support"]["board_capability_class"] = 0
        model["support"]["device_count"] = 1
        model["support"]["mesh_rows"] = 1
        model["support"]["mesh_cols"] = 1
        model["support"]["topology_sha256"] = "0" * 64
        model["support_sha256"] = emitter._sha256_value(model["support"])
        model["model_sha256"] = emitter.online_model_hash(model)
    if models:
        binding = emitter.online_models_bundle_binding(lock, models)
        for model in models:
            model["bundle_binding_sha256"] = binding

    evidence = {
        "authorizes_exact_entries": include_exact,
        "bank_artifact_sha256": bank_artifact_sha256,
        "bank_entry_inventory_sha256": emitter.direct_bank_entry_inventory_hash(lock),
        "bank_policy_version": emitter.DIRECT_BANK_EVIDENCE_POLICY_VERSION,
        "build_identity_sha256": lock["build_identity_sha256"],
        "exact_entry_inventory_sha256": emitter._sha256_value(emitter._exact_entry_inventory(lock)),
        "exact_native_support_sha256": emitter.exact_native_support_hash(lock),
        "online_model_bundle_binding_sha256": models[0]["bundle_binding_sha256"] if models else "0" * 64,
        "online_model_training_table_inventory_sha256": emitter.online_model_training_table_inventory_hash(models),
        "proof_sha256": "0" * 64,
        "safety_evidence_sha256": emitter.program_config_safety_inventory_hash(lock, models),
        "schema_version": 2,
        "semantic_source_sha256": lock["semantic_source_sha256"],
    }
    evidence["proof_sha256"] = emitter.program_config_only_evidence_hash(evidence)
    lock["program_config_only_evidence"] = evidence
    resign(lock)
    return lock


def active_online_model(lock: dict) -> dict:
    if "program_config_only_evidence" not in lock:
        lock["program_config_only_evidence"] = op_default_evidence(lock)
        lock["program_config_only_evidence"]["authorizes_exact_entries"] = False
        lock["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
            lock["program_config_only_evidence"]
        )
    key = lock["entries"][0]["key"]
    tensor_fields = ("buffer_type", "dtype", "layout", "memory_layout", "tile_height", "tile_width")

    def tensor(name: str) -> dict:
        return {field: key[name][field] for field in tensor_fields}

    programs = [
        {
            "allowed_worker_cores": None,
            "compute_grid_x": 4,
            "compute_grid_y": 4,
            "family": "multi_core_reuse",
            "fused_activation_present": False,
            "fuse_batch": False,
            "gather_in0": False,
            "hop_cores_present": False,
            "in0_block_w": 2,
            "mcast_in0": False,
            "num_global_cb_receivers": 0,
            "out_block_h": 0,
            "out_block_w": 0,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "per_core_m": 4,
            "per_core_n": key["padded_n"] // 32,
            "transpose_mcast": False,
            "untilize_out": False,
            "stream_in1": False,
        },
        {
            "allowed_worker_cores": None,
            "compute_grid_x": 8,
            "compute_grid_y": 8,
            "family": "multi_cast_1d",
            "fused_activation_present": False,
            "fuse_batch": True,
            "gather_in0": False,
            "hop_cores_present": False,
            "in0_block_w": 2,
            "mcast_in0": True,
            "num_global_cb_receivers": 1,
            "out_block_h": key["padded_m"] // 32,
            "out_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "per_core_m": key["padded_m"] // 32,
            "per_core_n": 2,
            "transpose_mcast": False,
            "untilize_out": False,
            "stream_in1": False,
        },
        {
            "allowed_worker_cores": None,
            "compute_grid_x": 8,
            "compute_grid_y": 8,
            "family": "multi_cast_2d",
            "fused_activation_present": False,
            "fuse_batch": True,
            "gather_in0": False,
            "hop_cores_present": False,
            "in0_block_w": 2,
            "mcast_in0": False,
            "num_global_cb_receivers": 0,
            "out_block_h": 1,
            "out_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "per_core_m": 1,
            "per_core_n": 2,
            "transpose_mcast": True,
            "untilize_out": False,
            "stream_in1": False,
        },
    ]
    candidates = [
        {"candidate_id": emitter.program_config_candidate_id(program), "program_config": program}
        for program in programs
    ]
    nodes = [
        {"feature": "family", "threshold": 0, "left": 1, "right": 2, "leaf_value": 0},
        {"feature": "leaf", "threshold": 0, "left": 0, "right": 0, "leaf_value": 10},
        {"feature": "leaf", "threshold": 0, "left": 0, "right": 0, "leaf_value": -10},
        {"feature": "family", "threshold": 1, "left": 1, "right": 2, "leaf_value": 0},
        {"feature": "leaf", "threshold": 0, "left": 0, "right": 0, "leaf_value": 10},
        {"feature": "leaf", "threshold": 0, "left": 0, "right": 0, "leaf_value": -10},
    ]
    support = {
        "architecture": key["architecture"],
        "board_capability_class": key["board_capability_class"],
        "device_count": key["device_count"],
        "domain": lock["entries"][0]["domain"],
        "input_a": tensor("input_a"),
        "input_b": tensor("input_b"),
        "maximum_k": key["logical_k"] * 2,
        "maximum_m": key["logical_m"] * 2,
        "maximum_n": key["logical_n"] * 2,
        "mesh_cols": key["mesh_cols"],
        "mesh_rows": key["mesh_rows"],
        "minimum_k": max(1, key["logical_k"] // 2),
        "minimum_m": max(1, key["logical_m"] // 2),
        "minimum_n": max(1, key["logical_n"] // 2),
        "output": tensor("output"),
        "shape_geometry": "output_wide",
        "shape_scale": "small_batch",
        "topology_sha256": key["topology_sha256"],
    }
    model = {
        "base_score": 0,
        "bundle_binding_sha256": "0" * 64,
        "candidate_policy_sha256": "6" * 64,
        "candidates": candidates,
        "enabled": True,
        "evaluation_model_payload_sha256": "a" * 64,
        "feature_schema_sha256": emitter.FEATURE_SCHEMA_SHA256,
        "lineage_sha256": "7" * 64,
        "maximum_normalized_shape_distance_ppm": 250_000,
        "minimum_score_margin": 1,
        "model_sha256": "0" * 64,
        "nodes": nodes,
        "quality_evaluation_sha256": "8" * 64,
        "safety_evidence_sha256": "5" * 64,
        "schema_version": 1,
        "score_orientation": "lower_is_better_negated_pairwise_margin",
        "score_scale": 1_000_000,
        "support": support,
        "support_sha256": emitter._sha256_value(support),
        "training_table_sha256": "4" * 64,
        "training_shapes": [
            [key["logical_m"], key["logical_k"], key["logical_n"]],
        ],
        "unseen_abstention_policy_sha256": "9" * 64,
        "trees": [{"node_count": 3, "node_offset": 0}, {"node_count": 3, "node_offset": 3}],
    }
    model["model_sha256"] = emitter.online_model_hash(model)
    model["bundle_binding_sha256"] = emitter.online_models_bundle_binding(lock, [model])
    lock["program_config_only_evidence"]["online_model_bundle_binding_sha256"] = model["bundle_binding_sha256"]
    lock["program_config_only_evidence"]["safety_evidence_sha256"] = emitter.program_config_safety_inventory_hash(
        lock, [model]
    )
    lock["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        lock["program_config_only_evidence"]
    )
    return model


def test_canonical_fixture_validates_and_emits_deterministically(tmp_path: Path) -> None:
    lock = fixture()
    checked = emitter.validate_lock(lock)
    assert checked["content_sha256"] == EXPECTED_CONTENT_SHA256
    assert checked["entries"][0]["entry_id"] == EXPECTED_ENTRY_ID
    first_header, first_source = emitter.emit(copy.deepcopy(checked))
    second_header, second_source = emitter.emit(copy.deepcopy(checked))
    assert (first_header, first_source) == (second_header, second_source)
    assert b"constexpr std::array<compact::EntryDescriptor, 1>" in first_source
    assert b"std::vector" not in first_header + first_source

    outputs: list[tuple[bytes, bytes]] = []
    for run in range(2):
        output_dir = tmp_path / str(run)
        header = output_dir / "matmul_registry_data.hpp"
        source = output_dir / "matmul_registry_data.cpp"
        emitter.main(
            [
                "--lock",
                str(FIXTURE_PATH),
                "--header",
                str(header),
                "--source",
                str(source),
            ]
        )
        outputs.append((header.read_bytes(), source.read_bytes()))
    assert outputs[0] == outputs[1] == (first_header, first_source)


def test_program_config_only_exact_activation_requires_bound_op_default_proof() -> None:
    legacy = fixture()
    _, legacy_source = emitter.emit(legacy)
    assert b".program_config_only_evidence_schema_version = 0" in legacy_source

    active = fixture()
    add_program_config_exact_entry(active)
    active["program_config_only_evidence"] = op_default_evidence(active)
    resign(active)
    _, active_source = emitter.emit(active)
    assert b".program_config_only_evidence_schema_version = 1" in active_source

    tampered = copy.deepcopy(active)
    tampered["program_config_only_evidence"]["effective_default_ckc_inventory_sha256"] = "d" * 64
    resign(tampered)
    try:
        emitter.validate_lock(tampered)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("tampered op-default proof must fail closed")

    rebound = copy.deepcopy(active)
    rebound["entries"][0]["key"]["topology_sha256"] = "e" * 64
    resign(rebound, entries=True)
    rebound["program_config_only_evidence"]["exact_entry_inventory_sha256"] = emitter._sha256_value(
        emitter._exact_entry_inventory(rebound)
    )
    # Deliberately retain the old exact_native_support_sha256: entry identity
    # and support-set hashes alone must not permit an entry/support rebind.
    rebound["program_config_only_evidence"]["safety_evidence_sha256"] = emitter.program_config_safety_inventory_hash(
        rebound, []
    )
    rebound["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        rebound["program_config_only_evidence"]
    )
    resign(rebound)
    try:
        emitter.validate_lock(rebound)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("exact entry/native support reassociation must fail closed")


def test_program_config_only_exact_entries_cover_all_native_families() -> None:
    expected = {
        "multi_core_reuse": b"ProgramFamily::MultiCoreReuse",
        "multi_cast_1d": b"ProgramFamily::MultiCast1D",
        "multi_cast_2d": b"ProgramFamily::MultiCast2D",
    }
    for family, emitted_family in expected.items():
        lock = fixture()
        exact = add_program_config_exact_entry(lock, family)
        lock["program_config_only_evidence"] = op_default_evidence(lock)
        resign(lock)
        checked = emitter.validate_lock(lock)
        header, source = emitter.emit(checked)
        assert b"program_config_exact_entries()" in header
        assert emitted_family in source
        assert emitter._bytes_cpp(exact["entry_id"]).encode() in source


def test_program_config_only_exact_entries_require_bound_authorization() -> None:
    lock = fixture()
    add_program_config_exact_entry(lock, "multi_cast_1d")
    resign(lock)
    try:
        emitter.validate_lock(lock)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("unauthorized program-config exact entry must fail closed")


def test_direct_bank_exact_entry_emits_without_session_certificate_and_rejects_tamper() -> None:
    lock = convert_to_direct_bank_lock(fixture())
    checked = emitter.validate_lock(lock)
    _, source = emitter.emit(checked)
    assert b".program_config_only_evidence_schema_version = 2" in source
    assert b"kProgramConfigExactEntries" in source
    assert "certificate" not in lock["program_config_exact_entries"][0]
    assert "runtime_capability_sha256" not in lock["program_config_only_evidence"]

    mutations = (
        lambda item: item["program_config_exact_entries"][0]["bank_evidence"].__setitem__("source_sha256", "e" * 64),
        lambda item: item["program_config_exact_entries"][0]["bank_evidence"].__setitem__(
            "lookup_key_sha256", "e" * 64
        ),
        lambda item: item["program_config_exact_entries"][0]["bank_evidence"].__setitem__(
            "program_config_sha256", "e" * 64
        ),
        lambda item: item["program_config_only_evidence"].__setitem__("semantic_source_sha256", "e" * 64),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(lock)
        mutate(tampered)
        resign(tampered)
        try:
            emitter.validate_lock(tampered)
        except emitter.LockValidationError:
            pass
        else:
            raise AssertionError("direct-bank evidence tamper must fail closed")

    nonzero_build = copy.deepcopy(lock)
    nonzero_build["build_identity_sha256"] = "e" * 64
    nonzero_build["program_config_only_evidence"]["build_identity_sha256"] = "e" * 64
    nonzero_build["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        nonzero_build["program_config_only_evidence"]
    )
    resign(nonzero_build)
    try:
        emitter.validate_lock(nonzero_build)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("static direct-bank lock must use the explicit zero build wildcard")


def test_direct_bank_exact_update_coexists_with_prior_online_model_fit_bank() -> None:
    lock = fixture()
    lock["online_program_config_models"] = [active_online_model(lock)]
    convert_to_direct_bank_lock(lock)
    checked = emitter.validate_lock(lock)
    _, source = emitter.emit(checked)
    assert b".online_program_config_model_evidence_schema_version = 2" in source
    assert b"kOnlineModel0" in source
    assert checked["online_program_config_models"][0]["support"]["board_capability_class"] == 0
    assert checked["online_program_config_models"][0]["support"]["topology_sha256"] == "0" * 64
    assert (
        checked["online_program_config_models"][0]["training_table_sha256"]
        != checked["program_config_only_evidence"]["bank_artifact_sha256"]
    )
    assert (
        emitter._bytes_cpp(
            checked["program_config_only_evidence"]["online_model_training_table_inventory_sha256"]
        ).encode()
        in source
    )


def test_active_online_program_config_model_validates_emits_and_has_reference_parity() -> None:
    lock = fixture()
    lock["online_program_config_models"] = [active_online_model(lock)]
    resign(lock)

    checked = emitter.validate_lock(lock)
    header, source = emitter.emit(checked)
    model = checked["online_program_config_models"][0]
    assert model["bundle_binding_sha256"] == emitter.online_models_bundle_binding(checked, [model])
    assert b"online_models()" in header
    assert b"kModelCandidates" in source
    assert b"ProgramFamily::MultiCast1D" in source
    assert b"ProgramFamily::MultiCast2D" in source
    assert b".transpose_mcast = true" in source
    assert emitter._bytes_cpp(model["bundle_binding_sha256"]).encode() in source

    def score(family: int) -> int:
        total = model["base_score"]
        for tree in model["trees"]:
            relative = 0
            while True:
                node = model["nodes"][tree["node_offset"] + relative]
                if node["feature"] == "leaf":
                    total += node["leaf_value"]
                    break
                value = family if node["feature"] == "family" else 0
                relative = node["left"] if value <= node["threshold"] else node["right"]
        return total

    assert [score(family) for family in range(3)] == [20, 0, -20]
    assert model["candidates"][2]["program_config"]["family"] == "multi_cast_2d"

    missing_activation_proof = copy.deepcopy(lock)
    del missing_activation_proof["program_config_only_evidence"]
    resign(missing_activation_proof)
    try:
        emitter.validate_lock(missing_activation_proof)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("enabled online model without activation proof must fail closed")

    overflowing = copy.deepcopy(lock)
    overflowing["online_program_config_models"][0]["base_score"] = 2**63 - 1
    resign(overflowing)
    try:
        emitter.validate_lock(overflowing)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("fixed-point score envelope overflow must fail closed")


def test_emitted_runtime_uses_gbdt_only_for_unseen_shapes(tmp_path: Path) -> None:
    lock = fixture()
    add_program_config_exact_entry(lock)
    lock["online_program_config_models"] = [active_online_model(lock)]
    lock["program_config_only_evidence"]["authorizes_exact_entries"] = True
    lock["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        lock["program_config_only_evidence"]
    )
    resign(lock)
    checked = emitter.validate_lock(lock)
    header, source = emitter.emit(checked)
    generated_header = tmp_path / "matmul_registry_data.hpp"
    generated_source = tmp_path / "matmul_registry_data.cpp"
    generated_header.write_bytes(header)
    generated_source.write_bytes(source)

    test_source = tmp_path / "unseen_shape_contract.cpp"
    test_source.write_text(
        r"""#include <span>

#include "matmul_registry_data.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_program_config_model.hpp"

int main() {
    namespace compact = ttnn::operations::matmul::registry::compact;
    namespace generated = ttnn::operations::matmul::registry::generated;
    const auto exact_entries = generated::program_config_exact_entries();
    const auto models = generated::online_models();
    if (exact_entries.size() != 1 || models.size() != 1) {
        return 1;
    }
    const auto& model = models.front();
    const auto binding = generated::metadata().online_model_bundle_binding_sha256;
    const auto training_key = exact_entries.front().key;

    const auto exact = compact::lookup_program_config(training_key, exact_entries, model, binding);
    if (exact.source != compact::ProgramConfigLookupSource::Exact) {
        return 2;
    }
    const auto missing_exact = compact::lookup_program_config(
        training_key, std::span<const compact::ProgramConfigExactEntry>{}, model, binding);
    if (missing_exact.source != compact::ProgramConfigLookupSource::None) {
        return 3;
    }

    auto unseen_key = training_key;
    --unseen_key.logical_m;
    const auto unseen = compact::lookup_program_config(
        unseen_key, std::span<const compact::ProgramConfigExactEntry>{}, model, binding);
    return unseen.source == compact::ProgramConfigLookupSource::Gbdt ? 0 : 4;
}
""",
        encoding="utf-8",
    )
    registry_include_dir = tmp_path / "ttnn" / "operations" / "matmul" / "device" / "config" / "registry"
    registry_include_dir.mkdir(parents=True)
    shutil.copyfile(
        REGISTRY_DIR / "matmul_program_config_model.hpp",
        registry_include_dir / "matmul_program_config_model.hpp",
    )
    shutil.copyfile(
        REGISTRY_DIR / "matmul_registry_descriptor.hpp",
        registry_include_dir / "matmul_registry_descriptor.hpp",
    )

    # Compile and run only fixed filenames inside pytest's private temporary
    # directory. None of the filesystem paths are interpolated into an OS
    # command, and subprocess never invokes a shell.
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-I.",
            "matmul_registry_data.cpp",
            "unseen_shape_contract.cpp",
            "-o",
            "unseen_shape_contract",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    subprocess.run(
        ["./unseen_shape_contract"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_absent_and_explicit_empty_online_models_emit_disabled_cpp() -> None:
    absent = fixture()
    explicit = fixture()
    explicit["online_program_config_models"] = []
    resign(explicit)
    absent_header, absent_source = emitter.emit(absent)
    explicit_header, explicit_source = emitter.emit(explicit)
    assert absent_header == explicit_header
    marker = b"constexpr std::array<compact::ProgramConfigGbdtModel, 0> kOnlineModels{{}};"
    assert marker in absent_source
    assert marker in explicit_source


def test_disjoint_online_model_set_emits_and_overlap_fails_closed() -> None:
    lock = fixture()
    first = active_online_model(lock)
    second = copy.deepcopy(first)
    second["support"]["domain"] = "dense.linear"
    second["support_sha256"] = emitter._sha256_value(second["support"])
    second["model_sha256"] = emitter.online_model_hash(second)
    models = sorted([first, second], key=lambda model: emitter.canonical_bytes(model["support"]))
    binding = emitter.online_models_bundle_binding(lock, models)
    for model in models:
        model["bundle_binding_sha256"] = binding
    lock["program_config_only_evidence"]["online_model_bundle_binding_sha256"] = binding
    lock["program_config_only_evidence"]["safety_evidence_sha256"] = emitter.program_config_safety_inventory_hash(
        lock, models
    )
    lock["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        lock["program_config_only_evidence"]
    )
    lock["online_program_config_models"] = models
    resign(lock)
    checked = emitter.validate_lock(lock)
    _, source = emitter.emit(checked)
    assert b"std::array<compact::ProgramConfigGbdtModel, 2>" in source

    tampered_safety = copy.deepcopy(lock)
    tampered_safety["online_program_config_models"][1]["safety_evidence_sha256"] = "d" * 64
    tampered_safety["online_program_config_models"][1]["model_sha256"] = emitter.online_model_hash(
        tampered_safety["online_program_config_models"][1]
    )
    tampered_binding = emitter.online_models_bundle_binding(
        tampered_safety, tampered_safety["online_program_config_models"]
    )
    for model in tampered_safety["online_program_config_models"]:
        model["bundle_binding_sha256"] = tampered_binding
    tampered_safety["program_config_only_evidence"]["online_model_bundle_binding_sha256"] = tampered_binding
    tampered_safety["program_config_only_evidence"]["proof_sha256"] = emitter.program_config_only_evidence_hash(
        tampered_safety["program_config_only_evidence"]
    )
    resign(tampered_safety)
    try:
        emitter.validate_lock(tampered_safety)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("model safety inventory tamper must invalidate the activation proof")

    tampered_lineage = copy.deepcopy(lock)
    tampered_lineage["online_program_config_models"][1]["lineage_sha256"] = "e" * 64
    tampered_lineage["online_program_config_models"][1]["model_sha256"] = emitter.online_model_hash(
        tampered_lineage["online_program_config_models"][1]
    )
    lineage_binding = emitter.online_models_bundle_binding(
        tampered_lineage, tampered_lineage["online_program_config_models"]
    )
    for model in tampered_lineage["online_program_config_models"]:
        model["bundle_binding_sha256"] = lineage_binding
    # Retain the old lock-level activation proof: it must bind the full-source
    # lineage transitively through the plural bundle, not merely each model's
    # locally projected safety ledger.
    resign(tampered_lineage)
    try:
        emitter.validate_lock(tampered_lineage)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("model lineage tamper must invalidate the activation proof")

    overlapping = copy.deepcopy(lock)
    overlapping["online_program_config_models"][1]["support"]["domain"] = overlapping["online_program_config_models"][
        0
    ]["support"]["domain"]
    overlapping["online_program_config_models"][1]["support_sha256"] = emitter._sha256_value(
        overlapping["online_program_config_models"][1]["support"]
    )
    overlapping["online_program_config_models"][1]["model_sha256"] = emitter.online_model_hash(
        overlapping["online_program_config_models"][1]
    )
    try:
        emitter.validate_lock(overlapping)
    except emitter.LockValidationError:
        pass
    else:
        raise AssertionError("overlapping online model support must fail closed")


def test_active_online_model_rejects_malformed_content(expect_error) -> None:
    mutations = []

    duplicate = fixture()
    duplicate["online_program_config_models"] = [active_online_model(duplicate)]
    duplicate["online_program_config_models"][0]["candidates"][1] = copy.deepcopy(
        duplicate["online_program_config_models"][0]["candidates"][0]
    )
    mutations.append(duplicate)

    escaped_tree = fixture()
    escaped_tree["online_program_config_models"] = [active_online_model(escaped_tree)]
    escaped_tree["online_program_config_models"][0]["nodes"][0]["right"] = 99
    mutations.append(escaped_tree)

    unsupported_tensor = fixture()
    unsupported_tensor["online_program_config_models"] = [active_online_model(unsupported_tensor)]
    unsupported_tensor["online_program_config_models"][0]["support"]["input_a"]["layout"] = "row_major"
    mutations.append(unsupported_tensor)

    wrong_feature_schema = fixture()
    wrong_feature_schema["online_program_config_models"] = [active_online_model(wrong_feature_schema)]
    wrong_feature_schema["online_program_config_models"][0]["feature_schema_sha256"] = "f" * 64
    mutations.append(wrong_feature_schema)

    stale_bundle = fixture()
    stale_bundle["online_program_config_models"] = [active_online_model(stale_bundle)]
    stale_bundle["online_program_config_models"][0]["bundle_binding_sha256"] = "e" * 64
    mutations.append(stale_bundle)

    missing_landmarks = fixture()
    missing_landmarks["online_program_config_models"] = [active_online_model(missing_landmarks)]
    missing_landmarks["online_program_config_models"][0]["training_shapes"] = []
    mutations.append(missing_landmarks)

    excessive_distance = fixture()
    excessive_distance["online_program_config_models"] = [active_online_model(excessive_distance)]
    excessive_distance["online_program_config_models"][0]["maximum_normalized_shape_distance_ppm"] = 250_001
    mutations.append(excessive_distance)

    for lock in mutations:
        resign(lock)
        with expect_error(emitter.LockValidationError, ".+"):
            emitter.validate_lock(lock)


def test_emitted_table_uses_pod_numeric_order_not_json_number_spelling() -> None:
    lock = fixture()
    second = copy.deepcopy(lock["entries"][0])
    lock["entries"][0]["key"]["architecture"] = 10
    second["key"]["architecture"] = 2
    lock["entries"].append(second)
    lock["entries"].sort(key=lambda item: emitter.canonical_bytes(item["key"]))
    resign(lock, entries=True)

    _, source = emitter.emit(lock)
    assert source.index(b".architecture = 2,") < source.index(b".architecture = 10,")


def test_each_public_domain_emits_a_disjoint_nonempty_key() -> None:
    emitted: dict[str, bytes] = {}
    for domain, alpha, beta, cpp_domain in (
        ("dense.matmul", None, None, b"compact::Domain::DenseMatmul"),
        ("dense.linear", None, None, b"compact::Domain::DenseLinear"),
        ("dense.addmm", 0x3F800000, 0x80000000, b"compact::Domain::DenseAddmm"),
    ):
        lock = fixture()
        entry = lock["entries"][0]
        entry["domain"] = domain
        entry["key"]["alpha_f32_bits"] = alpha
        entry["key"]["beta_f32_bits"] = beta
        resign(lock, entries=True)
        checked = emitter.validate_lock(lock)
        _, source = emitter.emit(checked)
        assert cpp_domain in source
        emitted[domain] = source

    assert len(set(emitted.values())) == 3

    combined = fixture()
    dense = combined["entries"][0]
    linear = copy.deepcopy(dense)
    linear["domain"] = "dense.linear"
    addmm = copy.deepcopy(dense)
    addmm["domain"] = "dense.addmm"
    addmm["key"]["alpha_f32_bits"] = 0x3F800000
    addmm["key"]["beta_f32_bits"] = 0x80000000
    combined["entries"] = sorted(
        [dense, linear, addmm], key=lambda item: emitter.canonical_bytes({"domain": item["domain"], "key": item["key"]})
    )
    resign(combined, entries=True)
    checked = emitter.validate_lock(combined)
    _, source = emitter.emit(checked)
    assert b"constexpr std::array<compact::EntryDescriptor, 3>" in source


def test_scalar_semantics_are_exclusive_to_addmm(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["key"]["alpha_f32_bits"] = 0x3F800000
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "exclusive to dense.addmm"):
        emitter.validate_lock(lock)

    lock = fixture()
    lock["entries"][0]["domain"] = "dense.addmm"
    lock["entries"][0]["key"]["alpha_f32_bits"] = 0
    lock["entries"][0]["key"]["beta_f32_bits"] = 0x3F800000
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "exactly 1.0"):
        emitter.validate_lock(lock)

    lock["entries"][0]["key"]["alpha_f32_bits"] = 0x40000000
    lock["entries"][0]["key"]["beta_f32_bits"] = 0
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "exactly 1.0"):
        emitter.validate_lock(lock)

    lock["entries"][0]["key"]["alpha_f32_bits"] = 0x3F800000
    lock["entries"][0]["key"]["beta_f32_bits"] = 0x3F800000
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "positive or negative zero"):
        emitter.validate_lock(lock)


def test_content_hash_tamper_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["content_sha256"] = "0" * 64
    with expect_error(emitter.LockValidationError, "content_sha256 mismatch"):
        emitter.validate_lock(lock)


def test_entry_id_tamper_is_rejected_even_with_resigned_content(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["entry_id"] = "0" * 64
    resign(lock)
    with expect_error(emitter.LockValidationError, "entry_id mismatch"):
        emitter.validate_lock(lock)


def test_duplicate_json_member_is_rejected(tmp_path: Path, expect_error) -> None:
    path = tmp_path / "duplicate.lock.json"
    path.write_text('{"artifact_kind":"first","artifact_kind":"second"}', encoding="utf-8")
    with expect_error(emitter.LockValidationError, "duplicate JSON key"):
        emitter.load_lock(path)


def test_duplicate_exact_key_is_rejected(expect_error) -> None:
    lock = fixture()
    duplicate = copy.deepcopy(lock["entries"][0])
    duplicate["certificate"]["evidence_sha256"] = "6" * 64
    lock["entries"].append(duplicate)
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "duplicates an exact key"):
        emitter.validate_lock(lock)


def test_certificate_production_policy_is_enforced_for_every_threshold(expect_error) -> None:
    invalid_values = {
        "baseline_policy_id": "unreviewed-baseline",
        "baseline_calls": emitter.MIN_CERTIFICATE_SESSIONS * emitter.MIN_CERTIFICATE_BLOCKS_PER_SESSION - 1,
        "baseline_ns": 0,
        "baseline_sessions": emitter.MIN_CERTIFICATE_SESSIONS - 1,
        "candidate_calls": emitter.MIN_CERTIFICATE_SESSIONS * emitter.MIN_CERTIFICATE_BLOCKS_PER_SESSION - 1,
        "candidate_ns": 0,
        "candidate_sessions": emitter.MIN_CERTIFICATE_SESSIONS - 1,
        "operational_lower_bound_ppm": emitter.MIN_CERTIFICATE_OPERATIONAL_LOWER_BOUND_PPM,
        "pcc_min_ppb": emitter.MIN_CERTIFICATE_PCC_PPB - 1,
        "speedup_ppm": emitter.MIN_CERTIFICATE_SPEEDUP_PPM - 1,
    }
    for field, value in invalid_values.items():
        lock = fixture()
        lock["entries"][0]["certificate"][field] = value
        resign(lock)
        with expect_error(emitter.LockValidationError, f"certificate.{field}"):
            emitter.validate_lock(lock)

    lock = fixture()
    lock["entries"][0]["certificate"]["pcc_min_ppb"] = 1_000_000_001
    resign(lock)
    with expect_error(emitter.LockValidationError, "pcc_min_ppb exceeds one"):
        emitter.validate_lock(lock)


def test_unknown_field_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["key"]["surprise"] = False
    resign(lock)
    with expect_error(emitter.LockValidationError, "field mismatch"):
        emitter.validate_lock(lock)


def test_unknown_program_family_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["recipe"]["program_config"]["family"] = "invented"
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "family is unknown"):
        emitter.validate_lock(lock)


def test_malformed_multi_core_reuse_work_splits_are_rejected(expect_error) -> None:
    cases: list[tuple[str, object]] = [
        ("key.input_a.tile_height", 16),
        ("key.input_a.tile_width", 16),
        ("key.padded_m", 129),
        ("key.input_b.tile_height", 16),
        ("key.input_b.tile_width", 16),
        ("key.output.tile_height", 16),
        ("key.output.tile_width", 16),
        ("program_config.in0_block_w", 3),
        ("program_config.per_core_m", 3),
        ("program_config.per_core_n", 3),
        ("program_config.out_subblock_h", 3),
        ("program_config.out_subblock_w", 3),
        ("program_config.compute_grid_x", 9),
    ]
    for field, value in cases:
        lock = fixture()
        entry = lock["entries"][0]
        if field.startswith("key."):
            target = entry["key"]
            components = field.removeprefix("key.").split(".")
        else:
            target = entry["recipe"]
            components = field.split(".")
        for component in components[:-1]:
            target = target[component]
        target[components[-1]] = value
        resign(lock, entries=True)
        with expect_error(emitter.LockValidationError, ".+"):
            emitter.validate_lock(lock)

    lock = fixture()
    program = lock["entries"][0]["recipe"]["program_config"]
    program["out_subblock_h"] = 4
    program["out_subblock_w"] = 4
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "destination-register bound"):
        emitter.validate_lock(lock)

    lock = fixture()
    program = lock["entries"][0]["recipe"]["program_config"]
    program["out_subblock_h"] = 2
    program["out_subblock_w"] = 4
    lock["entries"][0]["recipe"]["compute_kernel_config"]["fp32_dest_acc_en"] = True
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "destination-register bound"):
        emitter.validate_lock(lock)


def test_nondefault_output_call_state_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["key"]["output"]["buffer_type"] = "l1"
    lock["entries"][0]["recipe"]["call_state"]["output"]["buffer_type"] = "l1"
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "outside the first dense"):
        emitter.validate_lock(lock)


def test_non_tile_inputs_are_rejected(expect_error) -> None:
    for input_name in ("input_a", "input_b"):
        lock = fixture()
        lock["entries"][0]["key"][input_name]["layout"] = "row_major"
        resign(lock, entries=True)
        with expect_error(emitter.LockValidationError, "must use tile layout in v1"):
            emitter.validate_lock(lock)


def test_unsupported_schema_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["lock_schema_version"] += 1
    resign(lock)
    with expect_error(emitter.LockValidationError, "schema version is unsupported"):
        emitter.validate_lock(lock)


def test_unsupported_promotion_policy_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["policy_version"] = "unreviewed-promotion-policy"
    resign(lock)
    with expect_error(emitter.LockValidationError, "policy_version is unsupported"):
        emitter.validate_lock(lock)


def test_nonempty_lock_rejects_unmeasured_compatibility_sentinel(expect_error) -> None:
    lock = fixture()
    lock["runtime_capability_sha256"] = "0" * 64
    resign(lock)
    with expect_error(emitter.LockValidationError, "require measured compatibility"):
        emitter.validate_lock(lock)


def test_noncanonical_lock_bytes_are_rejected(tmp_path: Path, expect_error) -> None:
    path = tmp_path / "noncanonical.lock.json"
    path.write_text(FIXTURE_PATH.read_text(encoding="utf-8").replace(":", ": ", 1), encoding="utf-8")
    with expect_error(emitter.LockValidationError, "not canonical JSON"):
        emitter.load_lock(path)
