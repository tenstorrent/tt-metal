# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "emit_build_attestation.py"
SPEC = importlib.util.spec_from_file_location("emit_build_attestation", SCRIPT)
assert SPEC and SPEC.loader
emitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(emitter)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a" / "one.cpp").write_text("one\n", encoding="utf-8")
    (root / "two.hpp").write_text("two\n", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a/one.cpp\ntwo.hpp\n", encoding="utf-8")
    return root, manifest


def test_emit_is_deterministic_and_each_identity_is_independent(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    first = tmp_path / "first.hpp"
    second = tmp_path / "second.hpp"
    facts = emitter.parse_facts(["compiler=clang-19", "mode=release"])
    emitter.emit(root, manifest, facts, first)
    emitter.emit(root, manifest, facts, second)
    assert first.read_bytes() == second.read_bytes()

    original = first.read_text(encoding="utf-8")
    (root / "a" / "one.cpp").write_text("changed\n", encoding="utf-8")
    emitter.emit(root, manifest, facts, second)
    assert second.read_text(encoding="utf-8") != original

    (root / "a" / "one.cpp").write_text("one\n", encoding="utf-8")
    emitter.emit(root, manifest, emitter.parse_facts(["compiler=clang-20", "mode=release"]), second)
    assert second.read_text(encoding="utf-8") != original


def test_manifest_rejects_noncanonical_input(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    for body in ("two.hpp\na/one.cpp\n", "a/one.cpp\na/one.cpp\n", "../one.cpp\n", "/one.cpp\n"):
        manifest.write_text(body, encoding="utf-8")
        try:
            emitter.load_manifest(root, manifest)
        except emitter.AttestationError:
            continue
        raise AssertionError(f"invalid manifest was accepted: {body!r}")


def test_unlisted_generated_table_cannot_affect_semantic_digest(tmp_path: Path) -> None:
    root, manifest = _fixture(tmp_path)
    generated = root / "generated.cpp"
    generated.write_text("table one\n", encoding="utf-8")
    entries = emitter.load_manifest(root, manifest)
    before = emitter.digest(emitter.semantic_preimage(root, entries))
    generated.write_text("table two\n", encoding="utf-8")
    after = emitter.digest(emitter.semantic_preimage(root, entries))
    assert after == before
