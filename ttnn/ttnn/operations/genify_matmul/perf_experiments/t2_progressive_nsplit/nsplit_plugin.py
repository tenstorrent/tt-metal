"""pytest plugin for t2_progressive_nsplit (run.sh passes `-p nsplit_plugin` with this dir on PYTHONPATH).

NSPLIT_KDIR=1  the op builds its kernels from this dir's kernels/ copy (the real op files stay untouched).
NSPLIT=1       (needs NSPLIT_KDIR=1) on an R1b epilogue-none walk of more than one block per core, add the
               experiment defines: writer GM_NSPLIT=1 (a non-last block's output drains during the next block's A
               K-walk), compute GM_NSPLIT_LAST_N (the ragged last N-block computes only its nominal valid columns).
               The output CB must hold a whole block: pass GENIFY_MATMUL_KNOBS DEPTH_OUT=<row-group batches>.
NSPLIT_DEPTH=d pin the R1b operand ring depth (both rings) of the chosen blocking (re-solved block_k).
Knobs (DEPTH_OUT, TAIL_NC_PCT, ...) go through GENIFY_MATMUL_KNOBS; blockings through GENIFY_MATMUL_BLOCK."""
import os
import sys
from pathlib import Path

CAP = {}


def pytest_configure(config):
    if os.environ.get("NSPLIT_KDIR") != "1":
        return
    import ttnn.operations.genify_matmul.genify_matmul  # noqa: F401
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    op_mod = sys.modules["ttnn.operations.genify_matmul.genify_matmul"]
    pd.KERNEL_DIR = Path(__file__).parent / "kernels"
    print(f"NSPLIT: KERNEL_DIR -> {pd.KERNEL_DIR}")

    orig_plan = pd.plan

    def plan(*a, **k):
        r = orig_plan(*a, **k)
        CAP["plan"] = r
        return r

    pd.plan = plan

    depth = os.environ.get("NSPLIT_DEPTH")
    if depth:
        d = int(depth)
        orig_cb = pd.choose_blocking

        def choose_blocking(geom, grid, budget):
            blk = orig_cb(geom, grid, budget)
            if pd.operand_mode(geom, grid, blk) != 2:
                return blk
            bk_max = min(blk.block_k_tiles, pd._block_k_max(geom, blk.block_m_tiles, blk.block_n_tiles, budget, (d, d)))
            new = pd._make_blocking(geom, blk.block_batches, blk.block_m_tiles, blk.block_n_tiles, bk_max, (d, d))
            return pd.replace(new, write_batch=blk.write_batch)

        pd.choose_blocking = choose_blocking

    if os.environ.get("NSPLIT_AUTO") == "1":
        # Domain sweep: every R1b epilogue-none walk of ONE block (one M-block, one N-block, one batch) is re-blocked
        # into two N-blocks, ceil(core_n / 2) + the ragged rest, re-solving block_k at the same ring depths; the
        # output CB is sized to one whole block (DEPTH_OUT, restored per call by create_program_descriptor below).
        orig_cs = pd.choose_split

        def reblock(geom, grid, blk, budget):
            cn, cm = grid.core_n_tiles, grid.core_m_tiles
            if (
                pd.operand_mode(geom, grid, blk) != 2
                or geom.epilogue != "none"
                or blk.block_n_tiles != cn
                or blk.block_m_tiles != cm
                or grid.core_batches != 1
                or cn < 2
            ):
                return blk
            bn = -(-cn // 2)
            sh, sw = pd.choose_subblock(cm, bn, geom.dest_limit, geom)
            last_n = cn - bn
            depths = (blk.depth_act, blk.depth_weight)
            saved = pd.DEPTH_OUT
            pd.DEPTH_OUT = max(saved, cm // sh)
            # block_k capped at the baseline's (same K-block boundaries -> same accumulation order)
            bk_max = min(blk.block_k_tiles, pd._block_k_max(geom, cm, bn, budget, depths))
            if bk_max < min(geom.tensor_k_tiles, pd.BLOCK_K_FLOOR):
                pd.DEPTH_OUT = saved
                return blk
            new = pd._make_blocking(geom, 1, cm, bn, bk_max, depths)
            if pd.operand_mode(geom, grid, new) != 2:
                pd.DEPTH_OUT = saved
                return blk
            print(
                f"NSPLIT_AUTO: {cm}x{cn} -> block_n {bn} (+{last_n}), bk {new.block_k_tiles}, DEPTH_OUT {pd.DEPTH_OUT}"
            )
            return new

        def choose_split_auto(geom, gx, gy, budget):
            # the op's own split choice, THEN the re-blocking (the split ranking itself is untouched)
            grid, blk = orig_cs(geom, gx, gy, budget)
            return grid, reblock(geom, grid, blk, budget)

        pd.choose_split = choose_split_auto

    sb = os.environ.get("NSPLIT_SB")  # "h,w": pin the output subblock of every blocking (when it fits the block)
    if sb:
        sh, sw = (int(v) for v in sb.split(","))
        orig_sb = pd.choose_subblock

        def choose_subblock(block_m, block_n, dest_limit, geom=None):
            if block_m % sh == 0 and block_n % sw == 0 and sh * sw <= dest_limit:
                return sh, sw
            return orig_sb(block_m, block_n, dest_limit, geom)

        pd.choose_subblock = choose_subblock

    if os.environ.get("NSPLIT") != "1":
        return
    orig_cpd = op_mod.create_program_descriptor

    def create_program_descriptor(
        a, b, out, compute_config, epilogue="none", row=None, residual=None, placement="interleaved"
    ):
        CAP.clear()
        saved_depth_out = pd.DEPTH_OUT
        try:
            desc = orig_cpd(
                a, b, out, compute_config, epilogue=epilogue, row=row, residual=residual, placement=placement
            )
        finally:
            depth_out_used = pd.DEPTH_OUT
            pd.DEPTH_OUT = saved_depth_out
        if "plan" not in CAP or placement in ("dram_sharded", "block_sharded", "height_sharded"):
            return desc
        geom, grid, blk, _ = CAP["plan"]
        nb_m = -(-grid.core_m_tiles // blk.block_m_tiles)
        nb_n = -(-grid.core_n_tiles // blk.block_n_tiles)
        if pd.operand_mode(geom, grid, blk) != 2 or epilogue != "none" or nb_m * nb_n * grid.core_batches <= 1:
            print(f"NSPLIT: inactive (mode {pd.operand_mode(geom, grid, blk)}, nb_m {nb_m}, nb_n {nb_n})")
            return desc
        groups = blk.block_m_tiles // blk.sb_h
        assert depth_out_used * blk.write_batch >= groups, "NSPLIT needs an output CB of one whole block (DEPTH_OUT)"
        last_n = grid.core_n_tiles - (nb_n - 1) * blk.block_n_tiles
        for k in desc.kernels:
            src = str(k.kernel_source)
            if src.endswith("genify_matmul_writer.cpp") and os.environ.get("NSPLIT_W", "1") == "1":
                k.defines = list(k.defines) + [("GM_NSPLIT", "1")]
            elif (
                src.endswith("genify_matmul_compute.cpp")
                and last_n != blk.block_n_tiles
                and os.environ.get("NSPLIT_C", "1") == "1"
            ):
                in1_sub = blk.block_n_tiles // blk.sb_w
                if not ((blk.sb_h == 1 and (in1_sub - 1) * blk.sb_w < last_n) or blk.block_n_tiles <= geom.dest_limit):
                    continue  # ragged block computes its full nominal width (correct, wasted columns)
                k.defines = list(k.defines) + [("GM_NSPLIT_LAST_N", str(last_n))]
        print(f"NSPLIT: active nb_m {nb_m} nb_n {nb_n} block_n {blk.block_n_tiles} last_n {last_n}")
        return desc

    op_mod.create_program_descriptor = create_program_descriptor
