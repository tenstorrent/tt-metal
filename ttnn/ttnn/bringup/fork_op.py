# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fork an existing C++ TTNN op into ttnn/ttnn/bringup/<name> and register it as ttnn.bringup.<python name>.

    python ttnn/ttnn/bringup/fork_op.py ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch \
        --model <model> --task <task> [--name <dir name>] [--source-sha <sha>]

The copy is mechanical. The script then does the following:
- It moves the op's C++ namespace under ttnn::operations::bringup.
- It renames the CMake target to ttnn_op_bringup_<name> and the alias to TTNN::Ops::Bringup::<Name>.
- It points every kernel path and fork-internal #include at the copy.
- It binds the op under "ttnn.bringup." instead of its original Python prefix.
- It adds the op to bringup_nanobind.cpp and INDEX.md, and writes CHANGELOG.md with the source path and SHA.
The source op is not touched. Build afterwards (./build_metal.sh) and run the copied tests against ttnn.bringup.

Refuses to overwrite an existing fork: if the op is already in INDEX.md, extend that fork instead.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEXT = {".cpp", ".hpp", ".h", ".txt", ".cmake", ".py", ".md", ".inl"}
BEGIN, END = "// BEGIN FORKED OPS (fork_op.py)", "// END FORKED OPS (fork_op.py)"


def camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def rel_include(from_file: Path, target: Path) -> str:
    return os.path.relpath(target, from_file.parent).replace(os.sep, "/")


def rewrite(text: str, path: Path, src: Path, dst: Path, src_inc: str, ns: str, py_prefix: str, name: str) -> str:
    src_root = str(src.relative_to(ROOT))  # ttnn/cpp/ttnn/operations/.../<op>
    dst_root = str(dst.relative_to(ROOT))  # ttnn/ttnn/bringup/<name>
    here = dst / path.relative_to(src)

    # Fork-internal #includes by full path become relative to the including file (host and JIT kernels alike).
    def inc(m):
        target = dst / m.group(2)
        return f'{m.group(1)}"{rel_include(here, target)}"'

    for prefix in (src_root + "/", src_inc + "/"):
        text = re.sub(r'(#include\s*)"' + re.escape(prefix) + r'([^"]+)"', inc, text)
    # Kernel paths and install destinations (root-relative strings) point at the copy.
    text = text.replace(src_root + "/", dst_root + "/").replace(src_root + '"', dst_root + '"')
    text = text.replace("/" + src_root + "\n", "/" + dst_root + "\n")

    # The fork lives in ttnn::operations::bringup (still inside ttnn::operations, so the source's unqualified
    # lookups such as ccl::common:: keep resolving). `namespace ttnn { using operations::...::<op>::f; }` would put
    # the fork's functions next to the originals in namespace ttnn; they go to ttnn::bringup instead.
    rel_ns = ns.split("::", 1)[1]  # operations::experimental::deepseek_prefill

    def using_block(m):
        body = m.group(1).replace(f"using {rel_ns}::", "using ::ttnn::operations::bringup::")
        return f"namespace ttnn::bringup {{\n{body}}}  // namespace ttnn::bringup"

    text = re.sub(
        r"namespace ttnn \{\n((?:using " + re.escape(rel_ns) + r"::[^\n]*\n)+)\}  // namespace ttnn\b",
        using_block,
        text,
    )
    text = text.replace(ns + "::", "ttnn::operations::bringup::").replace(ns + " ", "ttnn::operations::bringup ")
    text = text.replace(ns + "\n", "ttnn::operations::bringup\n")
    text = text.replace(f'"{py_prefix}"', '"ttnn.bringup."')

    if path.name in ("CMakeLists.txt", "sources.cmake"):
        m = re.search(r"add_library\((\S+) \$\{LIB_TYPE\}\)", text)
        old_target = m.group(1) if m else None
        if path.name == "sources.cmake":
            cm = (src / "CMakeLists.txt").read_text()
            old_target = re.search(r"add_library\((\S+) \$\{LIB_TYPE\}\)", cm).group(1)
        new_target = f"ttnn_op_bringup_{name}"
        text = re.sub(r"TTNN::Ops::[A-Za-z:]*::" + r"\w+ ALIAS", f"TTNN::Ops::Bringup::{camel(name)} ALIAS", text)
        text = re.sub(r"(add_library\(\s*)TTNN::Ops::[A-Za-z:]+", rf"\1TTNN::Ops::Bringup::{camel(name)}", text)
        text = text.replace(old_target.upper(), new_target.upper()).replace(old_target, new_target)
        text = text.replace("BASE_DIRS ${FixmeOpAPIDir}", "BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}")
        if path.name == "CMakeLists.txt":
            # The source sat under ttnn/cpp and got it as an include dir from its API header set's BASE_DIRS.
            text = re.sub(
                r"(TT_ENABLE_UNITY_BUILD\(" + new_target + r"\)\n)",
                r"\1target_include_directories(" + new_target + r" PRIVATE ${FixmeOpAPIDir})\n",
                text,
            )
    return text


def isolate_namespaces(dst: Path) -> list[str]:
    """Nest every other namespace the fork declares (e.g. ttnn::prim, where device ops register their prim
    function) one level deeper, N -> N::bringup, and rewrite qualified references to the symbols the fork declares
    there. Otherwise both copies define the same symbols and ttnn fails to link. Idempotent."""
    files = [f for f in dst.rglob("*") if f.is_file() and f.suffix in {".cpp", ".hpp", ".h", ".inl"}]
    texts = {f: f.read_text() for f in files}
    decl = re.compile(r"^namespace ((?:\w+::)*\w+) \{", re.M)
    spaces = sorted(
        {n for t in texts.values() for n in decl.findall(t) if "::" in n and "bringup" not in n.split("::")},
        key=len,
        reverse=True,
    )
    for n in spaces:
        names = set()
        for t in texts.values():
            for m in re.finditer(
                r"^namespace " + re.escape(n) + r" \{\n(.*?)^\}  // namespace " + re.escape(n), t, re.M | re.S
            ):
                names |= set(re.findall(r"\b([A-Za-z_]\w*)\b", m.group(1)))
        for f, t in texts.items():
            t = re.sub(r"^namespace " + re.escape(n) + r" \{", f"namespace {n}::bringup {{", t, flags=re.M)
            t = re.sub(r"^\}  // namespace " + re.escape(n) + r"$", f"}}  // namespace {n}::bringup", t, flags=re.M)
            t = re.sub(
                r"(?<![\w:])(::)?" + re.escape(n) + r"::(\w+)",
                lambda m: f"{m.group(1) or ''}{n}::bringup::{m.group(2)}"
                if m.group(2) in names and m.group(2) != "bringup"
                else m.group(0),
                t,
            )
            texts[f] = t
    for f, t in texts.items():
        f.write_text(t)
    return spaces


def add_unlisted_sources(dst: Path, name: str) -> list[str]:
    """Host .cpp files the source had compiled from ttnn's central sources.cmake (not its own target) go into the
    fork's target, so the copy links on its own. Idempotent."""
    cm = dst / "CMakeLists.txt"
    listed = cm.read_text() + ((dst / "sources.cmake").read_text() if (dst / "sources.cmake").exists() else "")
    missing = [
        str(f.relative_to(dst))
        for f in sorted(dst.rglob("*.cpp"))
        if "kernels" not in f.relative_to(dst).parts
        and not f.name.endswith("_nanobind.cpp")
        and f.name != "bringup_nanobind.cpp"
        and str(f.relative_to(dst)) not in listed
    ]
    if missing:
        target = f"ttnn_op_bringup_{name}"
        cm.write_text(
            cm.read_text().rstrip("\n")
            + "\n\n# Compiled from ttnn/sources.cmake for the source op; the fork builds them in its own target.\n"
            + f"target_sources({target} PRIVATE {' '.join(missing)})\n"
        )
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "source", help="the op's folder, e.g. ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch"
    )
    ap.add_argument("--name", help="folder name under ttnn/ttnn/bringup (default: the source folder's name)")
    ap.add_argument("--model", required=True, help="the model whose bring-up needs the fork")
    ap.add_argument("--task", required=True, help="the task id that needs it")
    ap.add_argument("--source-sha", help="commit the source was taken from (default: HEAD)")
    a = ap.parse_args(argv)

    src = (ROOT / a.source).resolve()
    name = a.name or src.name
    dst = HERE / name
    if not (src / "CMakeLists.txt").is_file():
        sys.exit(f"{src} has no CMakeLists.txt; only C++ ops with their own CMake target can be forked")
    if dst.exists():
        sys.exit(f"{dst.relative_to(ROOT)} exists: extend that fork (INDEX.md) instead of forking again")
    src_inc = str(src.relative_to(ROOT / "ttnn" / "cpp"))  # ttnn/operations/.../<op>
    ns = "::".join(src_inc.split("/")[:-1])  # ttnn::operations::experimental::deepseek_prefill
    py_prefix = "ttnn." + ".".join(src_inc.split("/")[2:-1]) + "."  # ttnn.experimental.deepseek_prefill.
    sha = a.source_sha or git("rev-parse", "HEAD")
    sha = git("rev-parse", sha)

    for f in sorted(src.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT or f.name == "CMakeLists.txt":
            out.write_text(rewrite(f.read_text(), f, src, dst, src_inc, ns, py_prefix, name))
        else:
            shutil.copy2(f, out)

    isolate_namespaces(dst)
    add_unlisted_sources(dst, name)

    # Anything still pointing at the source tree is a sibling op the fork depends on; list it for the person.
    left = []
    for f in sorted(dst.rglob("*")):
        if f.is_file() and f.suffix in TEXT | {""}:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if ns in line or src_inc.rsplit("/", 1)[0] + "/" in line:
                    left.append(f"  {f.relative_to(ROOT)}:{i}: {line.strip()}")

    hpp = next(dst.glob("*_nanobind.hpp"), None)
    found = []
    for m in re.finditer(
        r"namespace (ttnn::operations::bringup[\w:]*::detail) \{(.*?)\n\}", hpp.read_text() if hpp else "", re.S
    ):
        found += [
            (m.group(1), f) for f in re.findall(r"void (bind_\w+)\(\s*(?:::)?(?:nb|nanobind)::module_", m.group(2))
        ]
    if not found:
        sys.exit(
            f"no bind_* function in a ttnn::operations::bringup...::detail namespace of {hpp}; register it by hand"
        )
    bns, bind = found[-1]  # the domain-level wrapper when there is one, else the op's own detail::bind_*
    reg = HERE / "bringup_nanobind.cpp"
    t = reg.read_text()
    t = t.replace(BEGIN + "\n", f"{BEGIN}\nnamespace {bns} {{\nvoid {bind}(nb::module_& mod);\n}}\n", 1)
    t = t.replace(f"    {BEGIN}\n", f"    {BEGIN}\n    ::{bns}::{bind}(mod);\n", 1)
    reg.write_text(t)
    binds = [f"{bns}::{bind}"]

    (dst / "CHANGELOG.md").write_text(
        f"# {name} (fork)\n\n"
        f"- Source: `{src.relative_to(ROOT)}`\n"
        f"- Source SHA: `{sha}`\n"
        f"- Python: `ttnn.bringup.*` (was `{py_prefix}*`)\n"
        f"- Forked for: {a.model} {a.task}\n"
        f"- Used by: {a.model}\n\n"
        "Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target "
        f"`ttnn_op_bringup_{name}`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.\n\n"
        "## Changes\n\n"
        "<!-- One entry per change, newest last:\n"
        "### <short title>\n"
        "- What: the change, and the switch or argument that turns it on (default = source behaviour).\n"
        "- Why: the symptom it fixes or the feature it adds.\n"
        "- Needed by: <model> <task>\n"
        "- Files: <paths inside this folder>\n"
        "-->\n"
    )
    idx = HERE / "INDEX.md"
    idx.write_text(
        idx.read_text().rstrip("\n")
        + f"\n| `{name}` | `{src.relative_to(ROOT)}` @ `{sha[:11]}` | (see CHANGELOG.md) | {a.model} |\n"
    )

    print(f"forked {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)} (source {sha[:11]}), bind {binds[0]}")
    if left:
        print("still referencing the source tree (sibling ops, check each):")
        print("\n".join(left))
    print("next: make the change, fill CHANGELOG.md and the INDEX.md row, ./build_metal.sh, run the copied tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
