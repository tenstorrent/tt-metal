# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: moe_fused_swiglu's cross-column REDUCE ACCUMULATE mechanism.

The op's reduce tree (``moe_fused_swiglu_compute.cpp`` §3, ``_reduce_tree`` in the program
descriptor) folds up to ``num_children`` remote partials onto a per-core running accumulator,
sequentially:

    add<input(cb_gate_acc), input(cb_reduce_gate_in), output(cb_gate_acc)>(EltwiseShape::tiles(48));
    add<input(cb_up_acc),   input(cb_reduce_up_in),   output(cb_up_acc)>  (EltwiseShape::tiles(48));

``cb_gate_acc`` / ``cb_up_acc`` / ``cb_reduce_*_in`` are all bfloat8_b — the textbook
``eltwise_l1_vs_dest_accumulate`` read-modify-write anti-pattern: the running accumulator is
unpacked, added, and packed back to bfp8 once PER CHILD (4 full L1 round trips of 48 tiles at the
root, plus a bfp8 requantization every step).

This bench isolates JUST that accumulate step (transport — the child unicast — is out of scope;
all children are pre-resident in L1, matching the real op once ``REDUCE_SLOTS >= fan_in``) and
measures three mechanisms at a fixed precision contract (LoFi / approx / fp32_dest_acc_en=False /
dst_full_sync_en=False, i.e. DEST_AUTO_LIMIT = 8 bf16 tiles — never fp32 DEST):

  VARIANT_BASELINE (0)     - the op's CURRENT approach, verbatim: per-child streaming `add`, the
                             accumulator round-trips L1 (unpack + add + pack) every child.
  VARIANT_PACK_L1_ACC (1)  - the PACKER folds each child onto the resident accumulator in L1;
                             the accumulator is never unpacked. One child resident at a time is
                             enough (REDUCE_SLOTS=1 legal, no transport protocol change needed).
  VARIANT_DEST_ACC (2)     - the seed (this core's own local partial) plus all `fan_in` children
                             are summed in a sticky bf16 DEST window (<= 8 tiles/window, since
                             DEST_AUTO_LIMIT=8 here), packed to the accumulator ONCE per window.
                             Needs all `fan_in` children resident AT ONCE (REDUCE_SLOTS=fan_in).

RAW LLK note (permitted inside this isolated bench, see the module docstring on the assignment):
``eltwise_chain``'s own ``L1Accumulation`` OutputSpec field is a MANY:1 reduce primitive — with
L1 accumulation enabled the chain's own "walk" derivation (`eltwise_chain.inl`, `PackTileImpl`)
PINS the pack address (`out_idx = base`, not `base + i_flat`) for every iteration, so an
``eltwise_chain`` call using it would collapse all BLOCK_TILES positions onto ONE tile instead of
preserving them. That is the wrong cardinality for this op's per-position (BLOCK_TILES-wide)
accumulate. VARIANT_PACK_L1_ACC therefore bypasses that OutputSpec field and drives the SAME
underlying hardware register directly with the raw ``pack_reconfig_l1_acc(1)`` / ``(0)`` toggle
around an ordinary (position-ADVANCING) ``copy<>`` call — exactly the mechanism the real op's own
`down` matmul already relies on (`matmul_block_helpers.inl` leaves `packer_l1_acc` set until an
explicit `pack_reconfig_l1_acc(0)`; `moe_fused_swiglu_compute.cpp`'s own comment on this: "neither
the eltwise chain (L1Accumulation::Disabled is a compile-time no-op) ... resets it"). With
``ReservePolicy::None`` and the chain's own L1Accumulation left ``Disabled``, `walk` evaluates
true (`Disabled == Disabled && (None == Upfront || None == None)`), so the address still advances
0..BLOCK_TILES-1 relative to a fixed (manually reserved) base — the raw toggle only changes
whether the PACKER overwrites or adds at each such address.

VARIANT_DEST_ACC does NOT need raw LLK: ``DestReuseBinary`` is a documented, non-pinned
``eltwise_chain`` element (``eltwise_chain.hpp``) that applies an FPU op between a CB operand and
the sticky DEST slot in place — chaining several of them lets one CopyTile(seed) + N
DestReuseBinary(child) + one PackTile(acc) process a <=8-tile window per outer iteration, entirely
through the documented chain API (mirrors `compute_fusion`'s `dstreuse` combine step and
`ttnn/cpp/ttnn/kernel_lib/tests/axes/{dest_reuse_param,fused_chain}.cpp`).
"""

import ttnn

TILE = 32

VARIANT_BASELINE = 0
VARIANT_PACK_L1_ACC = 1
VARIANT_DEST_ACC = 2
VARIANTS = (VARIANT_BASELINE, VARIANT_PACK_L1_ACC, VARIANT_DEST_ACC)
VARIANT_NAMES = {VARIANT_BASELINE: "baseline", VARIANT_PACK_L1_ACC: "pack_l1_acc", VARIANT_DEST_ACC: "dest_acc"}

CB_SEED = 0
CB_CHILD0 = 1
CB_CHILD1 = 2
CB_CHILD2 = 3
CB_CHILD3 = 4
CB_ACC = 5
_CHILD_CBS = (CB_CHILD0, CB_CHILD1, CB_CHILD2, CB_CHILD3)

_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

// moe_fused_swiglu reduce-accumulate isolated bench -- see bench.py's module docstring for the
// full mechanism writeup (baseline / pack_l1_acc / dest_acc) and the raw-LLK justification.
//
// CT args: [VARIANT, FAN_IN, BLOCK_TILES, cb_seed, cb_child0, cb_child1, cb_child2, cb_child3, cb_acc]
void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t FAN_IN = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t cb_seed = get_compile_time_arg_val(3);
    constexpr uint32_t cb_child0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_child1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_child2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_child3 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(8);
    static_assert(FAN_IN >= 1 && FAN_IN <= 4, "fan_in out of range");
    static_assert(VARIANT <= 2, "variant out of range");

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_seed, cb_child0, cb_acc);

    // The seed (this core's own local partial) and every child are pre-resident sharded-L1
    // tensors -- mark them available once, no real DMA (examples/eltwise_l1_vs_dest_accumulate's
    // resident-CB idiom). Transport is a different assigned part; out of scope here.
    cb_reserve_back(cb_seed, BLOCK_TILES);
    cb_push_back(cb_seed, BLOCK_TILES);
    if constexpr (FAN_IN >= 1) {
        cb_reserve_back(cb_child0, BLOCK_TILES);
        cb_push_back(cb_child0, BLOCK_TILES);
    }
    if constexpr (FAN_IN >= 2) {
        cb_reserve_back(cb_child1, BLOCK_TILES);
        cb_push_back(cb_child1, BLOCK_TILES);
    }
    if constexpr (FAN_IN >= 3) {
        cb_reserve_back(cb_child2, BLOCK_TILES);
        cb_push_back(cb_child2, BLOCK_TILES);
    }
    if constexpr (FAN_IN >= 4) {
        cb_reserve_back(cb_child3, BLOCK_TILES);
        cb_push_back(cb_child3, BLOCK_TILES);
    }

    if constexpr (VARIANT == 0) {
        // ---- baseline: the op's CURRENT approach, verbatim shape (moe_fused_swiglu_compute.cpp
        // section 3) -- sequential in-place streaming add. cb_acc round-trips L1 (unpack + FPU
        // add + pack, bfp8 requantized) once per child. `output(cb_acc)` == `input(cb_acc)` is
        // the documented in-place pattern (op_design.md section 6): PerTile wait+pop and PerTile
        // reserve+push on the SAME CB keep the read pointer trailing the write pointer by exactly
        // one full block, so each round overlays the same physical positions.
        CircularBuffer acc_buf(cb_acc);
        acc_buf.reserve_back(BLOCK_TILES);
        copy<input(cb_seed), output(cb_acc, ReservePolicy::None, PushPolicy::None)>(EltwiseShape::tiles(BLOCK_TILES));
        acc_buf.push_back(BLOCK_TILES);
        if constexpr (FAN_IN >= 1) {
            add<input(cb_acc), input(cb_child0), output(cb_acc)>(EltwiseShape::tiles(BLOCK_TILES));
        }
        if constexpr (FAN_IN >= 2) {
            add<input(cb_acc), input(cb_child1), output(cb_acc)>(EltwiseShape::tiles(BLOCK_TILES));
        }
        if constexpr (FAN_IN >= 3) {
            add<input(cb_acc), input(cb_child2), output(cb_acc)>(EltwiseShape::tiles(BLOCK_TILES));
        }
        if constexpr (FAN_IN >= 4) {
            add<input(cb_acc), input(cb_child3), output(cb_acc)>(EltwiseShape::tiles(BLOCK_TILES));
        }
    } else if constexpr (VARIANT == 1) {
        // ---- pack_l1_acc: raw pack_reconfig_l1_acc(1) folds each child onto acc without ever
        // unpacking it (see module docstring for why this bypasses eltwise_chain's own
        // L1Accumulation field). cb_acc is manually reserved ONCE for the whole fold sequence
        // (matching the real op's down-matmul `out_interm_buf.reserve_back(...); <K-block loop>;
        // out_interm_buf.push_back(...)` pattern) and used OUTPUT-ONLY (never read back) in every
        // fold call.
        //
        // TileOffset::Set{0} on the OUTPUT is REQUIRED here, not cosmetic: PackTileImpl's `walk`
        // (eltwise_chain.inl) computes the right per-position address (base + i_flat) whenever
        // Reserve==None, but only PASSES it to the underlying `pack_tile<out_of_order_output>` LLK
        // call when `out_of_order_output = (Offset != Unset) || (L1AccumulationMode != Disabled)`.
        // With L1Accumulation left Disabled at the chain level (the raw toggle drives the hardware
        // register instead) and Offset left Unset, out_of_order_output is FALSE, so `pack_tile`
        // ignores the computed index entirely and uses its OWN internal sequential counter, which
        // resets to 0 at the start of every separate eltwise_chain/copy<> call
        // (tt_metal/hw/inc/api/compute/pack.h: "each call to pack_tile advances the internal write
        // pointer ... reset after cb_push_back") -- confirmed by the FIRST version of this kernel,
        // which silently OVERWROTE acc with the LAST child every time instead of accumulating
        // (measured PCC matched "one random operand vs the full sum" exactly: 0.707 at fan_in=1,
        // 0.444 at fan_in=4, i.e. 1/sqrt(2) and 1/sqrt(5)). Setting TileOffset::Set{0} makes
        // out_of_order_output TRUE, so `pack_tile<true>(dst, cb_acc, base+i_flat)` honors the
        // EXPLICIT index every call -- the same physical tiles are targeted across all FAN_IN
        // separate fold calls -- while `pack_reconfig_l1_acc` (an INDEPENDENT hardware register,
        // per its own docstring: "configures the packer to accumulate ... with the ones already in
        // L1 at a given CB ID and tile index") does the actual add-not-overwrite.
        CircularBuffer acc_buf(cb_acc);
        acc_buf.reserve_back(BLOCK_TILES);
        using AccFoldPack = PackTile<output(
            cb_acc, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, PackRelu::Disabled,
            L1Accumulation::Disabled, DestAccumulation::Disabled, TileOffset::Set)>;
        eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_seed)>{}, AccFoldPack{0});
        pack_reconfig_l1_acc(1);
        if constexpr (FAN_IN >= 1) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_child0)>{}, AccFoldPack{0});
        }
        if constexpr (FAN_IN >= 2) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_child1)>{}, AccFoldPack{0});
        }
        if constexpr (FAN_IN >= 3) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_child2)>{}, AccFoldPack{0});
        }
        if constexpr (FAN_IN >= 4) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_child3)>{}, AccFoldPack{0});
        }
        pack_reconfig_l1_acc(0);
        acc_buf.push_back(BLOCK_TILES);
    } else {
        // ---- dest_acc: seed + FAN_IN children summed in a sticky bf16 DEST window (<=8 tiles,
        // DEST_AUTO_LIMIT under this op's fixed precision contract), packed to acc ONCE per
        // window -- BLOCK_TILES/BLK windows total, each its own tile_regs acquire/commit/pack.
        constexpr uint32_t BLK = (BLOCK_TILES < 8u) ? BLOCK_TILES : 8u;
        static_assert(BLOCK_TILES % BLK == 0, "BLOCK_TILES must divide into <=8-tile DEST windows");
        constexpr auto Up = WaitPolicy::Upfront;
        constexpr auto AtEnd = PopPolicy::AtEnd;
        constexpr auto Blk = OperandKind::Block;
        using SeedLoad = CopyTile<input(cb_seed, Up, AtEnd, Blk)>;
        using AccPack = PackTile<output(cb_acc, ReservePolicy::Upfront, PushPolicy::AtEnd)>;
        if constexpr (FAN_IN == 1) {
            eltwise_chain(
                EltwiseShape::tiles(BLOCK_TILES, BLK),
                SeedLoad{},
                DestReuseBinary<input(cb_child0, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                AccPack{});
        } else if constexpr (FAN_IN == 2) {
            eltwise_chain(
                EltwiseShape::tiles(BLOCK_TILES, BLK),
                SeedLoad{},
                DestReuseBinary<input(cb_child0, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child1, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                AccPack{});
        } else if constexpr (FAN_IN == 3) {
            eltwise_chain(
                EltwiseShape::tiles(BLOCK_TILES, BLK),
                SeedLoad{},
                DestReuseBinary<input(cb_child0, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child1, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child2, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                AccPack{});
        } else {
            eltwise_chain(
                EltwiseShape::tiles(BLOCK_TILES, BLK),
                SeedLoad{},
                DestReuseBinary<input(cb_child0, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child1, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child2, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                DestReuseBinary<input(cb_child3, Up, AtEnd, Blk), BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>{},
                AccPack{});
        }
    }
}
"""

# Fused gate+up (option 5): ONE combined pack_l1_acc call over 2*BLOCK_TILES instead of two
# separate BLOCK_TILES calls, to isolate the per-eltwise_chain-call overhead. Same mechanism as
# VARIANT_PACK_L1_ACC above; NUM_ROLES selects unfused (2 independent role blocks, one launch) vs
# the caller simply doubling BLOCK_TILES for the fused case (no kernel change needed for "fused" --
# see test file).
_FUSE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

// Option 5: pack_l1_acc, gate and up handled as TWO independent role blocks (unfused, 2 separate
// per-role call sequences paying init/reconfig twice) in ONE kernel launch. Compare against the
// SAME mechanism run once over a single combined 2*BLOCK_TILES role (bench.py reuses the main
// kernel above with BLOCK_TILES doubled for that arm).
//
// CT args: [FAN_IN, BLOCK_TILES,
//           cb_seed_a, cb_c0_a, cb_c1_a, cb_c2_a, cb_c3_a, cb_acc_a,
//           cb_seed_b, cb_c0_b, cb_c1_b, cb_c2_b, cb_c3_b, cb_acc_b]
void kernel_main() {
    constexpr uint32_t FAN_IN = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_seed_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_c0_a = get_compile_time_arg_val(3);
    constexpr uint32_t cb_c1_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_c2_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_c3_a = get_compile_time_arg_val(6);
    constexpr uint32_t cb_acc_a = get_compile_time_arg_val(7);
    constexpr uint32_t cb_seed_b = get_compile_time_arg_val(8);
    constexpr uint32_t cb_c0_b = get_compile_time_arg_val(9);
    constexpr uint32_t cb_c1_b = get_compile_time_arg_val(10);
    constexpr uint32_t cb_c2_b = get_compile_time_arg_val(11);
    constexpr uint32_t cb_c3_b = get_compile_time_arg_val(12);
    constexpr uint32_t cb_acc_b = get_compile_time_arg_val(13);
    static_assert(FAN_IN >= 1 && FAN_IN <= 4, "fan_in out of range");

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup(cb_seed_a, cb_c0_a, cb_acc_a);

    auto expose = [](uint32_t cb, uint32_t n) {
        cb_reserve_back(cb, n);
        cb_push_back(cb, n);
    };
    expose(cb_seed_a, BLOCK_TILES);
    expose(cb_seed_b, BLOCK_TILES);
    if constexpr (FAN_IN >= 1) { expose(cb_c0_a, BLOCK_TILES); expose(cb_c0_b, BLOCK_TILES); }
    if constexpr (FAN_IN >= 2) { expose(cb_c1_a, BLOCK_TILES); expose(cb_c1_b, BLOCK_TILES); }
    if constexpr (FAN_IN >= 3) { expose(cb_c2_a, BLOCK_TILES); expose(cb_c2_b, BLOCK_TILES); }
    if constexpr (FAN_IN >= 4) { expose(cb_c3_a, BLOCK_TILES); expose(cb_c3_b, BLOCK_TILES); }

    // Role A (e.g. "gate"), then role B (e.g. "up") -- two independent pack_l1_acc call
    // sequences, each paying its own seed-copy + FAN_IN fold calls' init/reconfig.
    // TileOffset::Set{0} is REQUIRED (see bench.py's main kernel comment on VARIANT==1): it makes
    // `out_of_order_output` true so `pack_tile` honors the explicit (fixed) index every call
    // instead of resetting its own internal sequential counter per call.
    {
        CircularBuffer acc_buf(cb_acc_a);
        acc_buf.reserve_back(BLOCK_TILES);
        using AccFoldPackA = PackTile<output(
            cb_acc_a, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, PackRelu::Disabled,
            L1Accumulation::Disabled, DestAccumulation::Disabled, TileOffset::Set)>;
        eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_seed_a)>{}, AccFoldPackA{0});
        pack_reconfig_l1_acc(1);
        if constexpr (FAN_IN >= 1) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c0_a)>{}, AccFoldPackA{0});
        }
        if constexpr (FAN_IN >= 2) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c1_a)>{}, AccFoldPackA{0});
        }
        if constexpr (FAN_IN >= 3) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c2_a)>{}, AccFoldPackA{0});
        }
        if constexpr (FAN_IN >= 4) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c3_a)>{}, AccFoldPackA{0});
        }
        pack_reconfig_l1_acc(0);
        acc_buf.push_back(BLOCK_TILES);
    }
    {
        CircularBuffer acc_buf(cb_acc_b);
        acc_buf.reserve_back(BLOCK_TILES);
        using AccFoldPackB = PackTile<output(
            cb_acc_b, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, PackRelu::Disabled,
            L1Accumulation::Disabled, DestAccumulation::Disabled, TileOffset::Set)>;
        eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_seed_b)>{}, AccFoldPackB{0});
        pack_reconfig_l1_acc(1);
        if constexpr (FAN_IN >= 1) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c0_b)>{}, AccFoldPackB{0});
        }
        if constexpr (FAN_IN >= 2) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c1_b)>{}, AccFoldPackB{0});
        }
        if constexpr (FAN_IN >= 3) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c2_b)>{}, AccFoldPackB{0});
        }
        if constexpr (FAN_IN >= 4) {
            eltwise_chain(EltwiseShape::tiles(BLOCK_TILES), CopyTile<input(cb_c3_b)>{}, AccFoldPackB{0});
        }
        pack_reconfig_l1_acc(0);
        acc_buf.push_back(BLOCK_TILES);
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(num_tiles):
    """One row of `num_tiles` tiles, height-sharded onto a single core (tiles row-major)."""
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def default_compute_kernel_config():
    """The op's fixed precision contract (moe_fused_swiglu.default_compute_kernel_config()),
    reconstructed here so this bench never touches it: LoFi / approx SFPU / fp16 DEST."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


def make_seed_and_children(device, fan_in, block_tiles, dtype, seed_val=0):
    """Build FAN_IN + 1 torch fp32 tensors (seed, child_0..child_{fan_in-1}) and their device
    (dtype, TILE_LAYOUT, single-core-sharded) counterparts. Returns (tt_seed, tt_children,
    torch_seed, torch_children)."""
    import torch

    torch.manual_seed(seed_val)
    cfg = create_sharded_memory_config(block_tiles)
    seed = torch.randn(TILE, block_tiles * TILE, dtype=torch.float32) * 0.1
    children = [torch.randn(TILE, block_tiles * TILE, dtype=torch.float32) * 0.1 for _ in range(fan_in)]
    tt_seed = ttnn.from_torch(seed, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    tt_children = [
        ttnn.from_torch(c, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg) for c in children
    ]
    return tt_seed, tt_children, seed, children


def create_program_descriptor(tt_seed, tt_children, tt_acc, *, variant, fan_in, block_tiles):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not (1 <= fan_in <= 4):
        raise ValueError(f"fan_in must be 1..4, got {fan_in}")
    if len(tt_children) != fan_in:
        raise ValueError(f"expected {fan_in} child tensors, got {len(tt_children)}")

    compile_time_args = [variant, fan_in, block_tiles, CB_SEED]
    compile_time_args.extend(_CHILD_CBS)  # always 4 slots; unused ones are dummy IDs, never referenced
    compile_time_args.append(CB_ACC)

    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        config=default_compute_kernel_config(),
    )

    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_SEED, tt_seed)]
    for i, t in enumerate(tt_children):
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(_CHILD_CBS[i], t))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_ACC, tt_acc))

    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_reduce_accum(tt_seed, tt_children, *, variant, fan_in, block_tiles, dtype):
    """Allocate the accumulator and run one variant. Returns the ttnn output tensor."""
    device = tt_seed.device()
    acc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, block_tiles * TILE]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(block_tiles),
    )
    descriptor = create_program_descriptor(
        tt_seed, tt_children, acc, variant=variant, fan_in=fan_in, block_tiles=block_tiles
    )
    return ttnn.generic_op([tt_seed, *tt_children, acc], descriptor)


# ---------------------------------------------------------------------------
# Option 5 — fuse gate+up into one pack_l1_acc call (bonus, not part of the main sweep)
# ---------------------------------------------------------------------------


def create_fuse_program_descriptor(
    tt_seed_a, tt_children_a, tt_acc_a, tt_seed_b, tt_children_b, tt_acc_b, *, fan_in, block_tiles
):
    if not (1 <= fan_in <= 4):
        raise ValueError(f"fan_in must be 1..4, got {fan_in}")

    CB_SEED_A, CB_C0_A, CB_C1_A, CB_C2_A, CB_C3_A, CB_ACC_A = 0, 1, 2, 3, 4, 5
    CB_SEED_B, CB_C0_B, CB_C1_B, CB_C2_B, CB_C3_B, CB_ACC_B = 6, 7, 8, 9, 10, 11
    child_cbs_a = (CB_C0_A, CB_C1_A, CB_C2_A, CB_C3_A)
    child_cbs_b = (CB_C0_B, CB_C1_B, CB_C2_B, CB_C3_B)

    compile_time_args = [fan_in, block_tiles, CB_SEED_A, *child_cbs_a, CB_ACC_A, CB_SEED_B, *child_cbs_b, CB_ACC_B]
    compute = ttnn.KernelDescriptor(
        kernel_source=_FUSE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        config=default_compute_kernel_config(),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_SEED_A, tt_seed_a),
        ttnn.cb_descriptor_from_sharded_tensor(CB_SEED_B, tt_seed_b),
    ]
    for i in range(fan_in):
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(child_cbs_a[i], tt_children_a[i]))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(child_cbs_b[i], tt_children_b[i]))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_ACC_A, tt_acc_a))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_ACC_B, tt_acc_b))

    tensors = [tt_seed_a, tt_seed_b, *tt_children_a, *tt_children_b, tt_acc_a, tt_acc_b]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs), tensors
