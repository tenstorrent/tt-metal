# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


CODEGEN_DIR = Path(__file__).resolve().parents[1]
REGISTRY_DIR = CODEGEN_DIR.parent
EMITTER_PATH = CODEGEN_DIR / "emit_cpp.py"
FIXTURE_PATH = REGISTRY_DIR / "fixtures" / "valid_multi_core_reuse.lock.json"
EXPECTED_CONTENT_SHA256 = "138cd5e90783e2fc23b475bfecdeaa3c00541d1321b71f3635aef7ae6bb4f338"
EXPECTED_ENTRY_ID = "9659081a8164b4e168b6751015cf53adbef2fab822e780c1c15ca00c29a15bd2"

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
        subprocess.run(
            [
                sys.executable,
                str(EMITTER_PATH),
                "--lock",
                str(FIXTURE_PATH),
                "--header",
                str(header),
                "--source",
                str(source),
            ],
            check=True,
        )
        outputs.append((header.read_bytes(), source.read_bytes()))
    assert outputs[0] == outputs[1] == (first_header, first_source)


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


def test_content_hash_tamper_is_rejected() -> None:
    lock = fixture()
    lock["content_sha256"] = "0" * 64
    with pytest.raises(emitter.LockValidationError, match="content_sha256 mismatch"):
        emitter.validate_lock(lock)


def test_entry_id_tamper_is_rejected_even_with_resigned_content() -> None:
    lock = fixture()
    lock["entries"][0]["entry_id"] = "0" * 64
    resign(lock)
    with pytest.raises(emitter.LockValidationError, match="entry_id mismatch"):
        emitter.validate_lock(lock)


def test_duplicate_json_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.lock.json"
    path.write_text('{"artifact_kind":"first","artifact_kind":"second"}', encoding="utf-8")
    with pytest.raises(emitter.LockValidationError, match="duplicate JSON key"):
        emitter.load_lock(path)


def test_duplicate_exact_key_is_rejected() -> None:
    lock = fixture()
    duplicate = copy.deepcopy(lock["entries"][0])
    duplicate["certificate"]["evidence_sha256"] = "6" * 64
    lock["entries"].append(duplicate)
    resign(lock, entries=True)
    with pytest.raises(emitter.LockValidationError, match="duplicates an exact key"):
        emitter.validate_lock(lock)


def test_unknown_field_is_rejected() -> None:
    lock = fixture()
    lock["entries"][0]["key"]["surprise"] = False
    resign(lock)
    with pytest.raises(emitter.LockValidationError, match="field mismatch"):
        emitter.validate_lock(lock)


def test_unknown_program_family_is_rejected() -> None:
    lock = fixture()
    lock["entries"][0]["recipe"]["program_config"]["family"] = "invented"
    resign(lock, entries=True)
    with pytest.raises(emitter.LockValidationError, match="family is unknown"):
        emitter.validate_lock(lock)


def test_nondefault_output_call_state_is_rejected() -> None:
    lock = fixture()
    lock["entries"][0]["key"]["output"]["buffer_type"] = "l1"
    lock["entries"][0]["recipe"]["call_state"]["output"]["buffer_type"] = "l1"
    resign(lock, entries=True)
    with pytest.raises(emitter.LockValidationError, match="outside the first dense"):
        emitter.validate_lock(lock)


def test_unsupported_schema_is_rejected() -> None:
    lock = fixture()
    lock["lock_schema_version"] += 1
    resign(lock)
    with pytest.raises(emitter.LockValidationError, match="schema version is unsupported"):
        emitter.validate_lock(lock)


def test_nonempty_lock_rejects_unmeasured_compatibility_sentinel() -> None:
    lock = fixture()
    lock["runtime_capability_sha256"] = "0" * 64
    resign(lock)
    with pytest.raises(emitter.LockValidationError, match="require measured compatibility"):
        emitter.validate_lock(lock)


def test_noncanonical_lock_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "noncanonical.lock.json"
    path.write_text(FIXTURE_PATH.read_text(encoding="utf-8").replace(":", ": ", 1), encoding="utf-8")
    with pytest.raises(emitter.LockValidationError, match="not canonical JSON"):
        emitter.load_lock(path)
