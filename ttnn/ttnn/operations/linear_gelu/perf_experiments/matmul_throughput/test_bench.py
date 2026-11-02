# matmul_throughput bake-off: whole-op runs of linear_gelu with the kernels copied + edited into this dir and the
# descriptor's host knobs monkeypatched. Every run is correctness-gated vs torch (full GELU: gelu_tanh(x@W+b);
# nogelu: x@W+b). Perf is read from the profiler CSV afterwards (parse.py), never asserted.
#
# Env:
#   MT_VARIANTS  comma list of variant names (see VARIANTS)
#   MT_MODES     comma list of nogelu,full
#   MT_SHAPES    comma list like 8x512,2x512 (optionally suffixed :dram for DRAM output)
import math
import os
import shutil
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

HERE = Path(__file__).parent
SRC = Path(pd.__file__).parent / "kernels"
ORDER_LOG = HERE / "order.log"

pytestmark = pytest.mark.skipif(not os.environ.get("MT_VARIANTS"), reason="set MT_VARIANTS")


def _prefer(kind):
    """Subblock chooser: 'tall' = descriptor default (largest area, larger sb_h on ties), 'square', 'wide'."""

    def choose(rows, block_w, dest_tiles, area_divides=None):
        best, best_key = (1, 1), None
        for sb_h in range(1, rows + 1):
            if rows % sb_h:
                continue
            for sb_w in range(1, block_w + 1):
                if block_w % sb_w or sb_h * sb_w > dest_tiles or (area_divides and area_divides % (sb_h * sb_w)):
                    continue
                area = sb_h * sb_w
                tie = {"tall": sb_h, "wide": sb_w, "square": -abs(sb_h - sb_w)}[kind]
                key = (area, tie)
                if best_key is None or key > best_key:
                    best, best_key = (sb_h, sb_w), key
        return best

    return choose


# name -> (host attrs, subblock preference, compute kernel override file or None)
VARIANTS = {
    "base": ({}, "tall", None),
    "lofi_diag": ({}, "tall", None),  # diagnostic only (fidelity is the user's knob): is the FPU binding?
    "skipmm_diag": ({}, "tall", None),  # diagnostic: matmul LLK elided (DM + pack + sync floor)
    "bh4": ({"BLOCK_H_MAX": 4}, "tall", None),
    "bh2": ({"BLOCK_H_MAX": 2}, "tall", None),
    "bw2_bh4_2x2": ({"BLOCK_W": 2, "BLOCK_H_MAX": 4}, "square", None),
    "bw2_bh4_4x1": ({"BLOCK_W": 2, "BLOCK_H_MAX": 4}, "tall", None),
    "bw2_bh2_2x2": ({"BLOCK_W": 2, "BLOCK_H_MAX": 2}, "square", None),
    "bw3_bh1_1x3": ({"BLOCK_W": 3, "BLOCK_H_MAX": 1}, "wide", None),
    "bw4_bh1_1x4": ({"BLOCK_W": 4, "BLOCK_H_MAX": 1}, "wide", None),
    "pf2": ({"W_PREFETCH_ROUNDS": 2}, "tall", None),
    "pf3": ({"W_PREFETCH_ROUNDS": 3}, "tall", None),
    "pf5": ({"W_PREFETCH_ROUNDS": 5}, "tall", None),
    "pf10": ({"W_PREFETCH_ROUNDS": 10}, "tall", None),
    "bh4_pf3": ({"BLOCK_H_MAX": 4, "W_PREFETCH_ROUNDS": 3}, "tall", None),
    "bh4_pf10": ({"BLOCK_H_MAX": 4, "W_PREFETCH_ROUNDS": 10}, "tall", None),
    "xearly2": ({}, "tall", None),
    "xearly4": ({}, "tall", None),
    "bh4_xearly2": ({"BLOCK_H_MAX": 4}, "tall", None),
    "bh4_xearly4": ({"BLOCK_H_MAX": 4}, "tall", None),
    "x1pre1": ({}, "tall", None),
    "x1pre5": ({}, "tall", None),
    "bh4_x1pre1": ({"BLOCK_H_MAX": 4}, "tall", None),
    "bh4_x1pre5": ({"BLOCK_H_MAX": 4}, "tall", None),
    "raw": ({}, "tall", "compute_raw.cpp"),
    "raw_bias_dest": ({}, "tall", "compute_raw.cpp"),
    "raw_bh4": ({"BLOCK_H_MAX": 4}, "tall", "compute_raw.cpp"),
    "raw_bias_dest_bh4": ({"BLOCK_H_MAX": 4}, "tall", "compute_raw.cpp"),
}


def _kernel_dir(variant, mode):
    attrs, _, compute_override = VARIANTS[variant]
    out = HERE / "gen_kernels" / f"{variant}_{mode}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in SRC.glob("*.cpp"):
        shutil.copy(f, out / f.name)
    comp = out / "linear_gelu_compute.cpp"
    if compute_override:
        shutil.copy(HERE / compute_override, comp)
    text = comp.read_text()
    defines = []
    if mode == "nogelu":
        if compute_override:
            defines.append("#define MT_NOGELU 1")
        else:
            assert "using Gelu = GeluTanhActivation<gelu_fast>;" in text
            text = text.replace("using Gelu = GeluTanhActivation<gelu_fast>;", "using Gelu = NoneActivation;")
    if variant == "skipmm_diag":
        defines.append("#define SKIP_COMPUTE")
    if "bias_dest" in variant:
        defines.append("#define MT_BIAS_DEST 1")
    if defines:
        text = text.replace("#include <cstdint>", "\n".join(defines) + "\n#include <cstdint>", 1)
    comp.write_text(text)
    if "x1pre" in variant:
        rd = out / "linear_gelu_reader.cpp"
        rd.write_text(_x1_pre_reader(rd.read_text(), int(variant.split("x1pre")[1])))
    if "xearly" in variant:
        rd = out / "linear_gelu_reader.cpp"
        rd.write_text(_x_early_reader(rd.read_text(), int(variant.split("xearly")[1])))
    if "bias_dest" in variant:
        # Bias tile becomes a full row-broadcast tile in the reader (all 32 rows = bias), consumed by copy into DEST.
        rd = out / "linear_gelu_reader.cpp"
        rd.write_text(_bias_bcast_reader(rd.read_text()))
    return out


def _x1_pre_reader(text, at):
    """x row-block 1's L1 reads are issued (no barrier) after W chunk `at` of row-block 0; its multicast stays after
    the last W chunk. Only the read latency moves off the critical path; the W channel's barriers absorb it."""
    old = """            for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {
                deliver_weight_block(n_block_idx);
            }"""
    new = f"""            constexpr uint32_t x1_pre_at = {at};
            for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {{
                deliver_weight_block(n_block_idx);
                if (n_block_idx == x1_pre_at && num_row_blocks > 1) {{
                    issue_activation_reads(1);
                }}
            }}
            if (x1_pre_at >= num_n_blocks && num_row_blocks > 1) {{
                issue_activation_reads(1);
            }}"""
    assert old in text
    text = text.replace(old, new)
    old2 = """    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        issue_activation_reads(row_block_idx);"""
    new2 = """    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        if (row_block_idx != 1) {
            issue_activation_reads(row_block_idx);
        }"""
    assert old2 in text
    return text.replace(old2, new2)


def _x_early_reader(text, at):
    """x row-block 1 is read + multicast after W chunk `at` of row-block 0 instead of after the last W chunk, so
    row-block 1's x does not wait for the whole W slice. Every core walks the same (x, W) round sequence, so the
    multicast handshakes stay deadlock-free."""
    old = """        finish_activation_block(row_block_idx);
        if (row_block_idx == 0) {
            MaybeDeviceZoneScope("reader_w_deliver");
            for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {
                deliver_weight_block(n_block_idx);
            }
        }"""
    new = f"""        if (row_block_idx == 1) {{
            continue;  // MT x-early: already delivered inside row-block 0's W loop
        }}
        finish_activation_block(row_block_idx);
        if (row_block_idx == 0) {{
            MaybeDeviceZoneScope("reader_w_deliver");
            constexpr uint32_t x_early_at = {at};
            const uint32_t at_chunk = num_n_blocks < x_early_at ? num_n_blocks : x_early_at;
            for (uint32_t n_block_idx = num_w_first; n_block_idx < num_n_blocks; ++n_block_idx) {{
                if (n_block_idx == at_chunk && num_row_blocks > 1) {{
                    issue_activation_reads(1);
                    finish_activation_block(1);
                }}
                deliver_weight_block(n_block_idx);
            }}
            if (at_chunk >= num_n_blocks && num_row_blocks > 1) {{
                issue_activation_reads(1);
                finish_activation_block(1);
            }}
        }}"""
    assert old in text
    text = text.replace(old, new)
    old2 = """    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        issue_activation_reads(row_block_idx);"""
    new2 = """    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        if (row_block_idx != 1) {
            issue_activation_reads(row_block_idx);
        }"""
    assert old2 in text
    return text.replace(old2, new2)


def _bias_bcast_reader(text):
    old = """            for (uint32_t w = 0; w < block_w; ++w) {
                const uint32_t tile_off = (k_tiles * block_w + w) * tile_bytes;
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_row_bytes});
                noc.async_write_zeros(
                    weight_cb, face_bytes - face_row_bytes, {.offset_bytes = tile_off + face_bytes + face_row_bytes});
                noc.async_write_zeros(weight_cb, 2 * face_bytes, {.offset_bytes = tile_off + 2 * face_bytes});
            }
            noc.write_zeros_l1_barrier();"""
    new = """            // MT_BIAS_DEST: replicate bias row 0 into every row (faces 0/1 rows 1..15, faces 2/3 rows 0..15).
            for (uint32_t w = 0; w < block_w; ++w) {
                volatile tt_l1_ptr uint32_t* t =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_addr + (k_tiles * block_w + w) * tile_bytes);
                constexpr uint32_t row_words = face_row_bytes / 4;
                constexpr uint32_t face_words = face_bytes / 4;
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t r = 0; r < 16; ++r) {
                        for (uint32_t c = 0; c < row_words; ++c) {
                            const uint32_t v = t[f * face_words + c];
                            if (r) t[f * face_words + r * row_words + c] = v;
                            t[(f + 2) * face_words + r * row_words + c] = v;
                        }
                    }
                }
            }"""
    assert old in text
    return text.replace(old, new)


def _ref(x, w, b, gelu):
    import torch

    y = x.float().reshape(-1, 768) @ w.float() + b.float()
    if gelu:
        y = torch.nn.functional.gelu(y, approximate="tanh")
    return y.reshape(*x.shape[:-1], 3072)


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _cases():
    out = []
    for v in os.environ.get("MT_VARIANTS", "base").split(","):
        for m in os.environ.get("MT_MODES", "nogelu,full").split(","):
            for s in os.environ.get("MT_SHAPES", "8x512").split(","):
                out.append((v, m, s))
    return out


@pytest.mark.parametrize("variant,mode,shape", _cases(), ids=lambda p: str(p))
def test_bench(device, variant, mode, shape, monkeypatch):
    import torch

    attrs, pref, _ = VARIANTS[variant]
    dram_out = shape.endswith(":dram")
    batch, seq = (int(v) for v in shape.split(":")[0].split("x"))
    for k, val in attrs.items():
        monkeypatch.setattr(pd, k, val)
    monkeypatch.setattr(pd, "_subblock", _prefer(pref))
    monkeypatch.setattr(pd, "KERNEL_DIR", _kernel_dir(variant, mode))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out_mem = ttnn.DRAM_MEMORY_CONFIG if dram_out else ttnn.L1_MEMORY_CONFIG
    cfg = None
    if variant == "lofi_diag":
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )
    blocking = pd.select_blocking(xd, cfg or ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True))
    y = linear_gelu(xd, wd, bd, memory_config=out_mem, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(y).float()
    ref = _ref(x, w, b, mode == "full")
    pcc = _pcc(got, ref)
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    sb = pd._subblock(blocking.block_h, blocking.block_w, blocking.dest_tiles)
    line = (
        f"{variant}|{mode}|{shape}|pcc={pcc:.6f}|relrms={rel_rms:.4e}|bh={blocking.block_h}|bw={blocking.block_w}"
        f"|sb={sb[0]}x{sb[1]}|T={int(blocking.transposed)}|tiles={blocking.core_m_tiles * blocking.core_n_tiles_max}"
    )
    with open(ORDER_LOG, "a") as f:
        f.write(line + "\n")
    print(line)
    if "diag" not in variant:
        assert pcc > 0.9999, line
