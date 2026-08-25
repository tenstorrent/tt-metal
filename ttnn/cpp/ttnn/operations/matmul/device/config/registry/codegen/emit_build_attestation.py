#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Emit independent build-time compatibility digests for the matmul registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path, PurePosixPath
from typing import Iterable


SCHEMA_VERSION = 1


class AttestationError(ValueError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def load_manifest(root: Path, manifest: Path) -> list[str]:
    entries: list[str] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = PurePosixPath(line)
        if path.is_absolute() or ".." in path.parts or str(path) != line:
            raise AttestationError(f"manifest line {line_number} is not a canonical repository-relative path")
        entries.append(line)
    if entries != sorted(entries) or len(entries) != len(set(entries)):
        raise AttestationError("manifest paths must be sorted and unique")
    if not entries:
        raise AttestationError("manifest must contain at least one semantic dependency")
    for entry in entries:
        source = root / entry
        if not source.is_file():
            raise AttestationError(f"semantic dependency does not exist: {entry}")
    return entries


def semantic_preimage(root: Path, entries: Iterable[str]) -> dict[str, object]:
    return {
        "artifact_kind": "ttnn_matmul_registry_semantic_preimage",
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {"path": entry, "sha256": hashlib.sha256((root / entry).read_bytes()).hexdigest()} for entry in entries
        ],
    }


def parse_facts(raw_facts: Iterable[str]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for raw_fact in raw_facts:
        name, separator, value = raw_fact.partition("=")
        if (
            not separator
            or not name
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name)
        ):
            raise AttestationError(f"invalid build fact: {raw_fact!r}")
        if name in facts:
            raise AttestationError(f"duplicate build fact: {name}")
        facts[name] = value
    if not facts:
        raise AttestationError("at least one build fact is required")
    return dict(sorted(facts.items()))


def build_preimage(facts: dict[str, str]) -> dict[str, object]:
    return {
        "artifact_kind": "ttnn_matmul_registry_build_preimage",
        "schema_version": SCHEMA_VERSION,
        "facts": dict(sorted(facts.items())),
    }


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def device_attestation_digests(facts: dict[str, int]) -> tuple[str, str]:
    """Exporter-side implementation of the v1 native binary preimage contract."""
    topology_fields = (
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
    )
    runtime_u32_fields = (
        "cluster_class",
        "num_hw_cqs",
        "num_dram_channels",
        "l1_size_per_core",
    )
    firmware_fields = (
        "firmware_bundle_major",
        "firmware_bundle_minor",
        "firmware_bundle_patch",
        "ethernet_firmware_major",
        "ethernet_firmware_minor",
        "ethernet_firmware_patch",
    )
    required = set(topology_fields + runtime_u32_fields + firmware_fields + ("dram_size_per_channel",))
    if set(facts) != required:
        raise AttestationError(f"device facts field mismatch: expected {sorted(required)}, got {sorted(facts)}")
    for name, value in facts.items():
        width = 64 if name == "dram_size_per_channel" else 32
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= 1 << width:
            raise AttestationError(f"device fact {name} is not uint{width}")

    topology_preimage = b"ttnn.matmul.registry.topology.v1" + b"".join(
        struct.pack("<I", facts[name]) for name in topology_fields
    )
    topology = hashlib.sha256(topology_preimage).digest()
    capability_preimage = (
        b"ttnn.matmul.registry.runtime-capability.v1"
        + topology
        + b"".join(struct.pack("<I", facts[name]) for name in runtime_u32_fields)
        + struct.pack("<Q", facts["dram_size_per_channel"])
        + b"".join(struct.pack("<I", facts[name]) for name in firmware_fields)
    )
    return topology.hex(), hashlib.sha256(capability_preimage).hexdigest()


def _bytes_cpp(value: str) -> str:
    return "{{" + ", ".join(f"0x{value[index : index + 2]}" for index in range(0, len(value), 2)) + "}}"


def emit(root: Path, manifest: Path, facts: dict[str, str], header: Path, receipt: Path) -> None:
    entries = load_manifest(root, manifest)
    semantic = semantic_preimage(root, entries)
    build = build_preimage(facts)
    semantic_sha256 = digest(semantic)
    build_sha256 = digest(build)
    receipt_value = {
        "artifact_kind": "ttnn_matmul_registry_build_attestation",
        "schema_version": SCHEMA_VERSION,
        "semantic_source_sha256": semantic_sha256,
        "build_identity_sha256": build_sha256,
        "semantic_preimage": semantic,
        "build_preimage": build,
    }
    header.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        """// Generated by emit_build_attestation.py. Do not edit.
#pragma once

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::generated_build {

inline constexpr std::uint16_t kAttestationSchemaVersion = 1;
inline constexpr compact::Sha256 kActualSemanticSourceSha256 = %s;
inline constexpr compact::Sha256 kActualBuildIdentitySha256 = %s;

}  // namespace ttnn::operations::matmul::registry::generated_build
"""
        % (_bytes_cpp(semantic_sha256), _bytes_cpp(build_sha256)),
        encoding="utf-8",
    )
    receipt.write_bytes(canonical_json(receipt_value) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fact", action="append", default=[])
    args = parser.parse_args()
    emit(args.root.resolve(), args.manifest.resolve(), parse_facts(args.fact), args.header, args.receipt)


if __name__ == "__main__":
    main()
