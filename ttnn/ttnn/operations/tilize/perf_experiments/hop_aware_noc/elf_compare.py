"""Binary identity of the OFF switch: hash the .text/.data of every JIT-built tilize reader (NCRISC)
and writer (BRISC) ELF, grouped by source (HEAD kernels/ vs graduate/kernels/) and hop define, keeping
only builds newer than their source. usage (repo root): python3 elf_compare.py [built/tt-metal-cache*/]"""
import glob, hashlib, os, subprocess, sys, tempfile
from collections import defaultdict

OBJCOPY = "runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"
HEAD = "ttnn/ttnn/operations/tilize/kernels/"
GRAD = "ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_noc/graduate/kernels/"
MIN_MTIME = float(os.environ.get("ELF_MIN_MTIME", "0"))  # graduate builds only after this (one release session)
roots = sys.argv[1:] or glob.glob("built/tt-metal-cache*/")
groups = defaultdict(set)
for root in roots:
    for kname, risc in (("tilize_writer", "brisc"), ("tilize_reader", "ncrisc")):
        for elf in glob.glob(f"{root}/kernels/{kname}/*/{risc}/{risc}.elf"):
            d = os.path.dirname(os.path.dirname(elf))
            inc = open(f"{d}/kernel_includes.hpp").read()
            src = HEAD if HEAD in inc else GRAD if GRAD in inc else None
            if src is None or os.path.getmtime(elf) < max(
                MIN_MTIME if src == GRAD else 0, os.path.getmtime(src + kname + ".cpp")
            ):
                continue
            ii = glob.glob(f"{d}/{risc}/*.ii")
            text = open(ii[0], errors="ignore").read() if ii else ""
            hop = "on" if "other_noc_banks" in text else "off"
            with tempfile.NamedTemporaryFile() as t:
                subprocess.run([OBJCOPY, "-O", "binary", "-j", ".text", "-j", ".data", elf, t.name], check=True)
                h = hashlib.sha1(open(t.name, "rb").read()).hexdigest()[:12]
            groups[(kname, "head" if src == HEAD else f"grad-{hop}")].add(h)
for kname in ("tilize_reader", "tilize_writer"):
    head, off, on = (groups[(kname, v)] for v in ("head", "grad-off", "grad-on"))
    print(
        f"{kname}: head {len(head)} binaries, grad-off {len(off)} (identical to a head binary: {len(off & head)}), "
        f"grad-on {len(on)} (identical to head: {len(on & head)})"
    )
    missing = off - head
    if missing:
        print(f"  {kname} grad-off binaries with no identical head binary: {sorted(missing)}")
