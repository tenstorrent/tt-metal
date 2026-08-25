# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

CODEGEN_DIR = Path(__file__).resolve().parents[1]
REGISTRY_DIR = CODEGEN_DIR.parent
EMITTER_PATH = CODEGEN_DIR / "emit_cpp.py"
FIXTURE_PATH = REGISTRY_DIR / "fixtures" / "valid_multi_core_reuse.lock.json"
EXPECTED_CONTENT_SHA256 = "3992cfdb6654bad0eb86db62fb50cc9623e5ba5c27a0c9a6c813b088b00e850d"
EXPECTED_ENTRY_ID = "cd7887816e01e45f91c79b8377aa63d3b9551153c5d32c0defbd9f6ffeb0fb3d"

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
    with expect_error(emitter.LockValidationError, "nonzero binary32"):
        emitter.validate_lock(lock)

    lock["entries"][0]["key"]["alpha_f32_bits"] = 0x3F800000
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


def test_nondefault_output_call_state_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["entries"][0]["key"]["output"]["buffer_type"] = "l1"
    lock["entries"][0]["recipe"]["call_state"]["output"]["buffer_type"] = "l1"
    resign(lock, entries=True)
    with expect_error(emitter.LockValidationError, "outside the first dense"):
        emitter.validate_lock(lock)


def test_unsupported_schema_is_rejected(expect_error) -> None:
    lock = fixture()
    lock["lock_schema_version"] += 1
    resign(lock)
    with expect_error(emitter.LockValidationError, "schema version is unsupported"):
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
