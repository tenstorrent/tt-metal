# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "emit_build_attestation.py"
SPEC = importlib.util.spec_from_file_location("emit_build_attestation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
attestation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attestation)


def make_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.cpp").write_bytes(b"alpha\n")
    (root / "z.hpp").write_bytes(b"omega\n")
    manifest = root / "manifest.txt"
    manifest.write_text("a.cpp\nz.hpp\n", encoding="utf-8")
    return root, manifest


def test_build_attestation_is_deterministic_and_independent(tmp_path: Path) -> None:
    root, manifest = make_tree(tmp_path)
    facts = attestation.parse_facts(["compiler_version=19.1", "build_type=Release"])
    header = tmp_path / "first.hpp"
    receipt = tmp_path / "first.json"
    attestation.emit(root, manifest, facts, header, receipt)
    second_header = tmp_path / "second.hpp"
    second_receipt = tmp_path / "second.json"
    attestation.emit(root, manifest, dict(reversed(tuple(facts.items()))), second_header, second_receipt)

    assert header.read_bytes() == second_header.read_bytes()
    assert receipt.read_bytes() == second_receipt.read_bytes()
    checked = json.loads(receipt.read_text(encoding="utf-8"))
    assert (
        checked["semantic_source_sha256"]
        == hashlib.sha256(attestation.canonical_json(checked["semantic_preimage"])).hexdigest()
    )
    assert (
        checked["build_identity_sha256"]
        == hashlib.sha256(attestation.canonical_json(checked["build_preimage"])).hexdigest()
    )
    assert "semantic_source_sha256" not in checked["build_preimage"]


def test_source_and_build_fact_mutations_change_only_their_digest(tmp_path: Path) -> None:
    root, manifest = make_tree(tmp_path)
    facts = {"build_type": "Release"}
    semantic_before = attestation.digest(attestation.semantic_preimage(root, attestation.load_manifest(root, manifest)))
    build_before = attestation.digest(attestation.build_preimage(facts))

    (root / "a.cpp").write_bytes(b"changed\n")
    semantic_after = attestation.digest(attestation.semantic_preimage(root, attestation.load_manifest(root, manifest)))
    assert semantic_after != semantic_before
    assert attestation.digest(attestation.build_preimage(facts)) == build_before
    assert attestation.digest(attestation.build_preimage({"build_type": "Debug"})) != build_before


@pytest.mark.parametrize("contents", ["z.hpp\na.cpp\n", "a.cpp\na.cpp\n", "../a.cpp\n", "/a.cpp\n"])
def test_manifest_rejects_noncanonical_or_duplicate_paths(tmp_path: Path, contents: str, expect_error) -> None:
    root, manifest = make_tree(tmp_path)
    manifest.write_text(contents, encoding="utf-8")
    with expect_error(attestation.AttestationError, "manifest"):
        attestation.load_manifest(root, manifest)


def test_build_facts_reject_duplicates(expect_error) -> None:
    with expect_error(attestation.AttestationError, "duplicate build fact"):
        attestation.parse_facts(["build_type=Release", "build_type=Debug"])


def test_unity_mode_is_bound_into_production_build_identity() -> None:
    matmul_cmake = Path(__file__).resolve().parents[5] / "CMakeLists.txt"
    assert "tt_unity_builds=${TT_UNITY_BUILDS}" in matmul_cmake.read_text(encoding="utf-8")


def test_device_attestation_binary_contract_is_frozen() -> None:
    facts = {
        "architecture": 1,
        "board_class": 3,
        "cluster_class": 6,
        "device_count": 1,
        "mesh_rows": 1,
        "mesh_cols": 1,
        "system_mesh_id": 0,
        "compute_grid_x": 13,
        "compute_grid_y": 10,
        "physical_grid_x": 17,
        "physical_grid_y": 12,
        "logical_grid_x": 13,
        "logical_grid_y": 10,
        "dram_grid_x": 8,
        "dram_grid_y": 1,
        "tensix_harvesting_mask": 0,
        "num_hw_cqs": 1,
        "num_dram_channels": 8,
        "l1_size_per_core": 1464320,
        "dram_size_per_channel": 4278190080,
        "firmware_bundle_major": 18,
        "firmware_bundle_minor": 10,
        "firmware_bundle_patch": 0,
        "ethernet_firmware_major": 6,
        "ethernet_firmware_minor": 8,
        "ethernet_firmware_patch": 1,
    }
    assert attestation.device_attestation_digests(facts) == (
        "fbe64700cb3163cc7dfdbba653d929ee24768fd9a0fbc5fabcb16c1757e8579e",
        "334e50711ede66bf5b973b9d5ab0b0ebfa47005f6e8db88582a941bb43552ee8",
    )

    mutated = dict(facts)
    mutated["firmware_bundle_patch"] += 1
    assert attestation.device_attestation_digests(mutated)[0] == attestation.device_attestation_digests(facts)[0]
    assert attestation.device_attestation_digests(mutated)[1] != attestation.device_attestation_digests(facts)[1]


def test_every_device_attestation_axis_is_digest_bound() -> None:
    facts = {
        "architecture": 1,
        "board_class": 3,
        "cluster_class": 6,
        "device_count": 1,
        "mesh_rows": 1,
        "mesh_cols": 1,
        "system_mesh_id": 0,
        "compute_grid_x": 13,
        "compute_grid_y": 10,
        "physical_grid_x": 17,
        "physical_grid_y": 12,
        "logical_grid_x": 13,
        "logical_grid_y": 10,
        "dram_grid_x": 8,
        "dram_grid_y": 1,
        "tensix_harvesting_mask": 0,
        "num_hw_cqs": 1,
        "num_dram_channels": 8,
        "l1_size_per_core": 1464320,
        "dram_size_per_channel": 4278190080,
        "firmware_bundle_major": 18,
        "firmware_bundle_minor": 10,
        "firmware_bundle_patch": 0,
        "ethernet_firmware_major": 6,
        "ethernet_firmware_minor": 8,
        "ethernet_firmware_patch": 1,
    }
    topology_only = {
        "architecture",
        "board_class",
        "device_count",
        "mesh_rows",
        "mesh_cols",
        "system_mesh_id",
        "compute_grid_x",
        "compute_grid_y",
        "physical_grid_x",
        "physical_grid_y",
        "logical_grid_x",
        "logical_grid_y",
        "dram_grid_x",
        "dram_grid_y",
        "tensix_harvesting_mask",
    }
    expected = attestation.device_attestation_digests(facts)
    for field in facts:
        mutated = dict(facts)
        mutated[field] += 1
        actual = attestation.device_attestation_digests(mutated)
        if field in topology_only:
            assert actual[0] != expected[0], field
        else:
            assert actual[0] == expected[0], field
        assert actual[1] != expected[1], field
