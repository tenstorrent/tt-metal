# SPDX-License-Identifier: Apache-2.0
"""One-shot source patch: replace the RESIDENT block/depth pick in desc_fork.py
with an env-driven rule so baseline and every candidate live in ONE binary-stable
fork (the JIT cache key does not hash kernel CONTENT, but it does hash the kernel
PATH -- which is why the whole descriptor + kernels were forked into this dir).

Rules (env RMS_RBC):
  shipped    the shipped D41 rule verbatim: per depth take the LARGEST br that
             fits, pick the depth with the most row-blocks, tie-break shallowest.
  maxblocks  IDEA A: sweep br in 1..brmax at EVERY depth, pick the most
             row-blocks; tie-break finest block, then shallowest depth.
  shallow    IDEA A minus the deeper ring: sweep br at the SHALLOWEST FEASIBLE
             depth only (never buy depth).
Modifiers:
  RMS_RBC_MAXB=N     cap the useful row-block count at N (0 = uncapped)
  RMS_RBC_BR=N       hard-force BLOCK_ROWS (grid probe; 0 = use the rule)
  RMS_RBC_DEPTH=N    hard-force the ring depth  (grid probe; 0 = use the rule)
  RMS_RBC_DEPTHS=a,b widen/narrow the resident depth ladder
"""
import pathlib
import sys

F = pathlib.Path(__file__).with_name("desc_fork.py")
src = F.read_text()

OLD = """        best = None
        for depth in reversed(resident_depths):
            brmax, _ = _resident_fit(depth, compact=True)
            if brmax < 2:
                brmax = min(1, _resident_fit(depth, compact=False)[0])
            if brmax >= 1:
                br = min(max_rows, brmax)
                blocks = -(-max_rows // br)
                if best is None or blocks > best[0]:
                    best = (blocks, depth, br)
        if best is not None:
            _, depth, br = best
            return br, wt_core, 1, depth, depth, True, CB_RM_STAGE_DEPTH, False, False
"""

NEW = """        # ---- perf_experiments/resident_block_count (IDEA A) -----------------
        _rule = os.environ.get("RMS_RBC", "shipped")
        _maxb = int(os.environ.get("RMS_RBC_MAXB", "0"))
        _fbr = int(os.environ.get("RMS_RBC_BR", "0"))
        _fdepth = int(os.environ.get("RMS_RBC_DEPTH", "0"))
        _ladder = os.environ.get("RMS_RBC_DEPTHS", "")
        if _ladder and is_tile:
            resident_depths = tuple(int(t) for t in _ladder.split(",") if t.strip())

        def _brmax(depth):
            brmax, _ = _resident_fit(depth, compact=True)
            if brmax < 2:
                brmax = min(1, _resident_fit(depth, compact=False)[0])
            return brmax

        cands = []  # (blocks, br, depth), feasible only
        for depth in reversed(resident_depths):
            brmax = _brmax(depth)
            if brmax < 1:
                continue
            top = min(max_rows, brmax)
            brs = (top,) if _rule == "shipped" else tuple(range(1, top + 1))
            for br in brs:
                cands.append((-(-max_rows // br), br, depth))
        if _rule == "shallow" and cands:
            d0 = min(d for _, _, d in cands)
            cands = [c for c in cands if c[2] == d0]
        if _maxb and cands:
            capped = [c for c in cands if c[0] <= _maxb]
            if capped:
                cands = capped
        best = None
        for blocks, br, depth in cands:
            # shipped == (blocks, shallowest); IDEA A adds the br sweep and
            # tie-breaks the FINEST block at equal block count (evenness).
            key = (blocks, -br, -depth) if _rule != "shipped" else (blocks, 0, -depth)
            if best is None or key > best[0]:
                best = (key, depth, br)
        if (_fbr or _fdepth) and cands:
            depth = _fdepth or (best[1] if best else resident_depths[-1])
            br = min(max_rows, _fbr) if _fbr else (best[2] if best else 1)
            fit = _brmax(depth)
            if fit >= 1 and br <= max(fit, 1):
                best = ((0, 0, 0), depth, br)
            else:
                print(f"RMS_RBC_FORCE INFEASIBLE depth={depth} br={br} brmax={fit}", flush=True)
        if best is not None:
            _, depth, br = best
            return br, wt_core, 1, depth, depth, True, CB_RM_STAGE_DEPTH, False, False
"""

if OLD not in src:
    sys.exit("PATCH FAILED: anchor not found")
F.write_text(src.replace(OLD, NEW, 1))
print("patched desc_fork.py")
