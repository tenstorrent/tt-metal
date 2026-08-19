// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GNA fused neighborhood reader -- P3 FIRST DRAFT (unvalidated; needs the P5 program-factory CT/RT-arg + CB
// contract and device debug). Implements the "read the region K/V once, reuse across the region's query-
// blocks" dataflow from GNA_FUSED_KERNEL_SPEC.md / gna_neighborhood_compute.SKELETON.cpp.
//
// Per this core's regions:
//   1. Gather the region's K/V-UNION box (bounding box of all the region's query-blocks, dilated by the
//      kernel) ONCE into cb_kv_region -- coalesced W-run gather (reuse neighborhood_gather::gather_range_wrun).
//      This is the read-once step: the shared halo is not re-read per block.
//   2. For each query-block in the region: stream its Q rows into cb_q_block, and its [box_tile_lo, box_tile_hi)
//      sub-range + tile-mask handles (region-relative) into the ctrl CBs for compute.
//
// RT-arg contract (set by P5 program factory, per core):
//   num_regions, then per region: region_kv_page_base, region_box_ntiles, num_qblocks,
//     then per qblock: q_row_base, box_tile_lo, box_tile_hi, mask_row_base
// Host geometry (box_idx / mask_table from gna_gather.py) is baked into these page/tile bases.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "dataflow_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/neighborhood_gather.hpp"

void kernel_main() {
    Noc noc;
    // --- compile-time (P5 assigns exact indices) ---
    constexpr uint32_t heads = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);    // K head-dim in tiles
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);   // V head-dim in tiles
    constexpr uint32_t vol_t = get_compile_time_arg_val(3);  // query-block size in tiles (padded)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_kv_region = get_compile_time_arg_val(5);   // resident region K/V
    constexpr uint32_t cb_q_block = get_compile_time_arg_val(6);     // per-block Q
    constexpr uint32_t cb_qblk_range = get_compile_time_arg_val(7);  // per-block [lo,hi) + mask base ctrl
    constexpr uint32_t cb_gather_stage = get_compile_time_arg_val(8);
    // TensorAccessor for the wrow-paged K/V table + Q table follow (P5)
    constexpr auto q_args = TensorAccessorArgs<9>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_regions = get_arg_val<uint32_t>(argidx++);

    const auto q_reader = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto kv_reader = TensorAccessor(kv_args, kv_addr, /*page=*/0 /*P5: W-row page bytes*/);

    for (uint32_t r = 0; r < num_regions; ++r) {
        const uint32_t region_kv_rows = get_arg_val<uint32_t>(argidx++);  // box-union rows (t,h)-rows
        const uint32_t region_box_ntiles = get_arg_val<uint32_t>(argidx++);
        const uint32_t num_qblocks = get_arg_val<uint32_t>(argidx++);
        const uint32_t region_row_base = get_arg_val<uint32_t>(argidx++);  // first W-row page of the union

        // ---- 1. region K/V-union gathered ONCE into cb_kv_region (coalesced W-runs) ----
        CircularBuffer kvcb(cb_kv_region);
        kvcb.reserve_back(region_box_ntiles);
        // P4/P5: call neighborhood_gather::gather_range_wrun over the region-union box to fill kvcb (K then V),
        // reading each W-row page once. (region geometry from RT args; scatter_row packs into the CB.)
        //   gather_range_wrun(noc, kv_reader, kvcb.get_write_ptr(), cb_gather_stage L1, ...region box..., ...);
        kvcb.push_back(region_box_ntiles);

        // ---- 2. per query-block: stream Q + the ctrl (box-tile range + mask base) ----
        for (uint32_t qb = 0; qb < num_qblocks; ++qb) {
            const uint32_t q_row_base = get_arg_val<uint32_t>(argidx++);
            const uint32_t box_tile_lo = get_arg_val<uint32_t>(argidx++);
            const uint32_t box_tile_hi = get_arg_val<uint32_t>(argidx++);
            const uint32_t mask_row_base = get_arg_val<uint32_t>(argidx++);

            CircularBuffer qcb(cb_q_block);
            qcb.reserve_back(vol_t * DHt);
            for (uint32_t t = 0; t < vol_t * DHt; ++t) {
                noc.async_read(
                    q_reader,
                    CoreLocalMem<uint32_t>(qcb.get_write_ptr() + t * tile_bytes),
                    tile_bytes,
                    {.page_id = q_row_base + t},
                    {});
            }
            noc.async_read_barrier();
            qcb.push_back(vol_t * DHt);

            CircularBuffer ctrl(cb_qblk_range);
            ctrl.reserve_back(1);
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl.get_write_ptr());
            p[0] = box_tile_lo;  // compute visits only [lo,hi) of the resident region K/V
            p[1] = box_tile_hi;
            p[2] = mask_row_base;  // base into cb_tile_mask for this block's tri-state tile-mask
            ctrl.push_back(1);
        }
        kvcb.pop_front(region_box_ntiles);  // region done; free for the next
    }
}
