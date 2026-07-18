// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/pool_kernels_common.hpp>

#define ENABLE_DEBUG_PRINT 0  // [DEBUG] test DM-core DPRINT capture (remove after)

#if ENABLE_DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

// Read kernel data for normal max/average pooling (without indices)
template <
    uint32_t in_nblocks_c,
    uint32_t in_cb_id,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_w_padded,
    uint32_t in_nbytes_leftover,
    uint32_t in_c,
    uint32_t max_sticks_for_reduction,
    uint32_t total_elems_to_reduce,
    bool is_avg_pool,
    bool wide_reduction,
    uint32_t clear_value_cb_id,
    uint32_t in_cb_ntiles,
    uint32_t in_nbytes_c,
    uint32_t shard_width_bytes,
    bool is_large_kernel,
    bool last_tile_is_partial,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool zero_pages,
    uint32_t in_cb_sz,
    uint32_t bf16_init_value,
    bool force_max_tiles_per_reduction_4>
ALWI void read_kernel_with_top_left_index(uint32_t ind, uint32_t in_l1_read_base_addr) {
    constexpr uint32_t BYTES_PER_ELEM = 2;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr uint32_t MAX_TILES_PER_REDUCTION =
        (force_max_tiles_per_reduction_4 || (is_avg_pool && is_large_kernel)) ? 4 : 8;
    constexpr uint32_t MAX_BYTES_PER_REDUCTION = MAX_TILES_PER_REDUCTION * TILE_WIDTH * BYTES_PER_ELEM;
    constexpr uint32_t in_ntiles_c = (in_c + TILE_WIDTH - 1) / TILE_WIDTH;
    constexpr uint32_t num_tilized_rows =
        wide_reduction ? (in_cb_sz / (MAX_TILES_PER_REDUCTION * TILE_WIDTH)) : (in_cb_sz / (in_ntiles_c * TILE_WIDTH));
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     (kernel_h * kernel_w) <= 16 && !last_tile_is_partial;

    DataflowBuffer in_cb(in_cb_id);
    DataflowBuffer clear_cb(clear_value_cb_id);
    Noc noc;
    UnicastEndpoint self_ep;

    uint32_t max_write_inc = wide_reduction ? MAX_BYTES_PER_REDUCTION : in_nbytes_leftover;
    for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
        uint32_t read_bytes = in_nbytes_c;
        if constexpr (wide_reduction) {
            read_bytes =
                (c_i == in_nblocks_c - 1) ? in_nbytes_c - c_i * MAX_BYTES_PER_REDUCTION : MAX_BYTES_PER_REDUCTION;
        }

        in_cb.reserve_back(1);
        uint32_t write_offset = 0;
        uint32_t processed_sticks = 0;
        // page zeroing is only necessary for tiled block output format so that scale is not affected by
        // junk/padding data
        if constexpr (zero_pages) {
            if (c_i == in_nblocks_c - 1 && last_tile_is_partial) {
                zero_out_page(noc, in_cb);
            }
        }
        // When the CB intentionally holds more rows than the kernel window (medium kernels,
        // FACE_WIDTH < kernel_size_hw < TILE_HEIGHT), the rows in
        // [total_elems_to_reduce, num_tilized_rows) are never overwritten by the async_reads
        // below and would otherwise contribute junk to the reduce. Fill only that tail region
        // with the init value -- the leading rows will be fully overwritten by process_h().
        if constexpr (!is_large_kernel) {
            if constexpr (num_tilized_rows > total_elems_to_reduce) {
                constexpr uint32_t row_stride_elems =
                    wide_reduction ? (MAX_TILES_PER_REDUCTION * TILE_WIDTH) : (in_ntiles_c * TILE_WIDTH);
                constexpr uint32_t tail_offset_bytes = total_elems_to_reduce * row_stride_elems * BYTES_PER_ELEM;
                constexpr uint32_t tail_elems = (num_tilized_rows - total_elems_to_reduce) * row_stride_elems;
                fill_with_val(
                    in_cb.get_write_ptr() + tail_offset_bytes, tail_elems, static_cast<uint16_t>(bf16_init_value));
#ifdef ARCH_QUASAR
                // Quasar sim coherency: write back the CPU-store tail fill so compute's TL1 read of in_cb's
                // pad rows sees the init value (not stale L1). Same reason as the window-copy write-back.
                flush_l2_cache_range(
                    static_cast<uintptr_t>(in_cb.get_write_ptr() + tail_offset_bytes),
                    static_cast<size_t>(tail_elems) * 2);
#endif
            }
        }
        for (uint32_t h = 0; h < kernel_h; ++h) {
            auto process_h = [&](uint32_t w_offset, uint32_t w_multiple) __attribute__((always_inline)) {
                const uint32_t stick_offset = ind + w_offset + h * dilation_h * in_w_padded;
                const uint32_t read_offset =
                    in_l1_read_base_addr + (stick_offset * shard_width_bytes + c_i * MAX_BYTES_PER_REDUCTION);
#ifdef ARCH_QUASAR
                // [DIAG cache-vs-data RESULT] The invalidate above proved coherency is NOT the cause: with the
                // invalidate before this DPRINT, POOLSRC still shows 0.4375 -> the data at base is genuinely
                // wrong (stick ~447 / row13), not stale cache. Reverted the read invalidate; the real bug is
                // the input base / data-layout (in_shard_cb.get_read_ptr() misaligned with the input's stick0).
                // [DIAG 64c reconfig escape] Dump the SOURCE the reader gathers for the first few windows:
                // base + stick_offset + the actual read address + the source values. golden out_stick0 ~0.032
                // (input row1); if src[] here already reads ~0.44 (row13) the reader/base is wrong; if src[]
                // is correct (~0.03) but the output is 0.4375, the bug is downstream (compute reduce).
                if (c_i == 0 && h == 0 && w_offset == 0 && ind <= 8) {
                    volatile tt_l1_ptr uint16_t* sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(read_offset);
                    DPRINT(
                        "POOLSRC ind={} soff={} roff={} base={} sw={} inwp={} src0={} src1={} src2={} src3={}\n",
                        (uint32_t)ind,
                        (uint32_t)stick_offset,
                        (uint32_t)read_offset,
                        (uint32_t)in_l1_read_base_addr,
                        (uint32_t)shard_width_bytes,
                        (uint32_t)in_w_padded,
                        bf16_t(sv[0]),
                        bf16_t(sv[1]),
                        bf16_t(sv[2]),
                        bf16_t(sv[3]));
                }
#endif
#ifdef ARCH_QUASAR
                // Quasar sim: a local self-loopback NOC read (self_ep, src_coord==dst_coord) into in_cb drops
                // data / reads stale SRAM. The window gather is a same-core L1->L1 copy, so do it with a direct
                // RISC copy for reliable data. read_offset and the in_cb write ptr are L1-aligned (>=16B) and
                // read_bytes is L1-aligned, so a uint32 word copy is safe. (NOTE: the 64c-after-32c failure is
                // NOT this path -- CPU and NOC reads BOTH see the same wrong TL1 data at base, so the input
                // tensor's L1 data is genuinely stale after 32c, a host-upload/L1-alloc issue, not the gather.)
                {
                    volatile tt_l1_ptr uint32_t* rp_src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_offset);
                    volatile tt_l1_ptr uint32_t* rp_dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_cb.get_write_ptr() + write_offset);
                    const uint32_t rp_nwords = (read_bytes * w_multiple) >> 2;
                    for (uint32_t rp_i = 0; rp_i < rp_nwords; ++rp_i) {
                        rp_dst[rp_i] = rp_src[rp_i];
                    }
                    // Write back the CPU-store copy to TL1 so the compute unpack (which reads in_cb from TL1)
                    // sees it. Without this, compute reduces stale/zero in_cb.
                    flush_l2_cache_range(
                        reinterpret_cast<uintptr_t>(rp_dst), static_cast<size_t>(read_bytes * w_multiple));
                }
#else
                noc.async_read(
                    self_ep,
                    in_cb,
                    read_bytes * w_multiple,
                    experimental::local_addr(read_offset),
                    {.offset_bytes = write_offset});
#endif
                // if compute is using tilize_reconfig we will only untilize the needed number of tiles rather
                // than the entire MAX_TILES_PER_REDUCTION, thus we use a different offset for the write address
                if constexpr (tilize_reconfig) {
                    write_offset += read_bytes * w_multiple;
                } else {
                    write_offset += max_write_inc * w_multiple;
                }
                processed_sticks += w_multiple;
                if constexpr (is_large_kernel) {
                    if ((processed_sticks % max_sticks_for_reduction) == 0 ||
                        processed_sticks == total_elems_to_reduce) {
                        noc.async_read_barrier();
                        in_cb.push_back(1);
                        in_cb.reserve_back(1);
                        write_offset = 0;
                        // If next is last chunk, fill whole buffer with the init_value. note for max pool we do
                        // not need to fill the CB for the partial chunk since as long as we have N>1 chunks we
                        // are guaranteed that the junk data remaining from chunk N-1 will fill the entire CB and
                        // cannot contain values greater than the max value, and if we have N=1 chunks we already
                        // initialized the entire CB with the init value, but for avg pool we need to fill the
                        // entire CB with the init value since the junk data will contribute to the average.
                        if constexpr (is_avg_pool) {
                            // clear the in CB
                            if ((total_elems_to_reduce - processed_sticks) < max_sticks_for_reduction &&
                                processed_sticks != total_elems_to_reduce) {
#ifdef ARCH_QUASAR
                                // QSR avg_pool coherency fix (same class as the once-at-init clears above):
                                // clear_out_tiles copies clear_value_cb -> in_cb via a NoC self-loopback read,
                                // which is UNRELIABLE on the Quasar sim (drops / reads stale SRAM). This clear
                                // runs right before the PARTIAL last reduction chunk (e.g. window 49 = 32 + 17);
                                // if it drops, the padding rows [remaining, TILE_HEIGHT) of the freshly reserved
                                // in_cb entry hold stale L1 that the reduce sums into the average -> per-channel
                                // means shifted up and DECORRELATED from the golden (low PCC). 0 is the avg (sum)
                                // additive identity, so an SRAM-coherent NoC zero-write of the just-reserved
                                // in_cb entry produces the identical clear without the loopback. get_entry_size()
                                // is exactly the one-page region clear_out_tiles(..., in_cb_ntiles) covered.
                                noc.async_write_zeros(in_cb, in_cb.get_entry_size());
                                noc.write_zeros_l1_barrier();
#else
                                clear_out_tiles<clear_value_cb_id>(noc, in_cb, clear_cb, in_cb_ntiles);
#endif
                            }
                        }
                    }
                }
            };

            // Case where in_nbytes_leftover and in_nbytes_c is different is when we are dealing with
            // tesnors that have last tile as partial. Cb page size is multiple of tile but when the last
            // tile is partial we have to read the smaller stick width. Therefore we need to write out the next
            // stick right below the previous one and this is when increment of the write pointer and the read
            // stick size is not compliant.
            bool use_contiguous_read = !wide_reduction && in_nbytes_leftover == in_nbytes_c &&
                                       dilation_w == 1;  // read entire row as one chunk (only if no width dilation)
            if constexpr (is_large_kernel) {
                bool whole_row_remaining =
                    kernel_w <= max_sticks_for_reduction - (processed_sticks % max_sticks_for_reduction);
                use_contiguous_read &= whole_row_remaining;
            }

            if (use_contiguous_read) {
                process_h(0, kernel_w);
            } else {  // read rows stick by stick with dilation
                for (uint32_t w = 0; w < kernel_w; ++w) {
                    process_h(w * dilation_w, 1);
                }
            }
        }
        if constexpr (!is_large_kernel) {
            noc.async_read_barrier();
            in_cb.push_back(1);
        }
    }
}

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
#if ENABLE_DEBUG_PRINT == 1
    DPRINT("READER_ENTER (data-movement kernel_main reached)\n");
#endif
    constexpr uint32_t reader_nindices = get_arg(args::reader_nindices);
    constexpr uint32_t kernel_h = get_arg(args::kernel_h);
    constexpr uint32_t kernel_w = get_arg(args::kernel_w);

    constexpr int32_t pad_w = get_arg(args::pad_w);

    // channel size in bytes
    constexpr uint32_t in_nbytes_leftover = get_arg(args::in_nbytes_leftover);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_arg(args::in_w);

    constexpr uint32_t in_c = get_arg(args::in_c);

    constexpr uint32_t split_reader = get_arg(args::split_reader);
    constexpr uint32_t reader_id = get_arg(args::reader_id);

    constexpr uint32_t bf16_scalar = get_arg(args::bf16_scalar);
    constexpr uint32_t bf16_init_value = get_arg(args::bf16_init_value);

    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t in_cb_sz = get_arg(args::in_cb_sz);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);
    constexpr uint32_t ceil_pad_w = get_arg(args::ceil_pad_w);

    // CB ids now come from Metal 2.0 DFB bindings. Split-reader uses per-reader input/scalar
    // DFBs bound under the same accessor names, so the kernel references one name regardless
    // of reader_id (the host binds the right DFB per reader KernelSpec).
    constexpr uint32_t in_cb_id = dfb::in_cb;
    // [DEBUG scratch->DM] this reader's scratch CB (reader0->scratch_cb_0, reader1->scratch_cb_1; the
    // factory routes dfb::scratch_cb per reader). Compute produces it; we consume it and NoC-copy row 0
    // into out_shard_cb (borrowed OUTPUT view), working around the broken narrow pack.
    constexpr uint32_t scratch_cb_id = dfb::scratch_cb;
    constexpr uint32_t out_shard_cb_id = dfb::out_shard_cb;
    constexpr uint32_t out_row_bytes = get_arg(args::out_row_bytes);
    constexpr uint32_t in_shard_cb_id = dfb::in_shard_cb;
    constexpr uint32_t in_reader_indices_cb_id = dfb::reader_indices_cb;
    constexpr uint32_t in_scalar_cb_id = dfb::in_scalar_cb;
    constexpr uint32_t clear_value_cb_id = dfb::clear_value_cb;
    constexpr bool is_avg_pool = (bool)get_arg(args::pool_type_is_avg);
    // QSR: cap tiles-per-pack to 4 (matches compute + host in_nblocks_c); see factory comment (PACR0 bounds).
    constexpr bool force_max_tiles_per_reduction_4 = (bool)get_arg(args::force_max_tiles_per_reduction_4);
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
    // The avg-pool scalar config DFB + tensor::config only exist when !one_scalar_per_core; the
    // host emits HAS_CONFIG to this kernel's defines exactly then. Gate every dfb::config_cb /
    // tensor::config reference: `if constexpr (!one_scalar_per_core)` is not enough since the
    // discarded branch still name-looks-up the (then-undeclared) tokens.
#ifdef HAS_CONFIG
    constexpr uint32_t config_cb_id = dfb::config_cb;
    constexpr uint32_t config_page_size = get_arg(args::config_page_size);
#endif
    constexpr uint32_t in_nbytes_c = get_arg(args::in_nbytes_c);
    constexpr uint32_t shard_width_bytes = get_arg(args::shard_width_bytes);
    constexpr uint32_t multi_buffering_factor = get_arg(args::multi_buffering_factor);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr bool zero_pages = (bool)get_arg(args::zero_pages);
    constexpr uint32_t config_in_dram = get_arg(args::config_in_dram);
    constexpr uint32_t reader_page_size = get_arg(args::reader_page_size);

    constexpr bool use_split_reader = split_reader;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    // The per-reader scalar DFB selection that legacy code did here (in_scalar_cb_id_1 for
    // reader1 when !one_scalar_per_core) is now done on the host: each reader KernelSpec binds
    // its own scalar DFB under accessor name "in_scalar_cb", so in_scalar_cb_id == dfb::in_scalar_cb.

    constexpr uint32_t window_size_hw = kernel_h * kernel_w;
    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT ? window_size_hw : FACE_HEIGHT;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_WIDTH || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;
    // we only need to initialize the in_cb if we will not fill each reduction chunk with valid data
    constexpr bool need_to_initialize_in_cb =
        (remaining_elems && face_r_dim == FACE_HEIGHT && (num_faces_in_input_tile == 4 || last_tile_is_partial) &&
         interm_reduction_chunks <= multi_buffering_factor);
    constexpr uint32_t in_cb_ntiles = in_cb_sz / (TILE_WIDTH * TILE_HEIGHT);  // only use the non-multi buffering size

    DataflowBuffer clear_value_cb(clear_value_cb_id);
    DataflowBuffer in_scalar_cb(in_scalar_cb_id);
    DataflowBuffer in_shard_cb(in_shard_cb_id);
    DataflowBuffer scratch_cb(scratch_cb_id);      // [DEBUG scratch->DM]
    DataflowBuffer out_shard_cb(out_shard_cb_id);  // [DEBUG scratch->out] borrowed OUTPUT view (NoC dest)
    // Compute reserves/pushes the WHOLE scratch CB per output stick; wait/pop the same whole-CB count so
    // the single-tile scratch serializes and we never read a partially/overlapping-written tile.
    constexpr uint32_t scratch_npages = get_arg(args::scratch_npages);
    DataflowBuffer reader_indices_cb(in_reader_indices_cb_id);
#ifdef HAS_CONFIG
    DataflowBuffer config_cb(config_cb_id);
#endif

    // QSR max_pool fix: for a partial-face window (face_r_dim < 16, e.g. 3x3 -> 9), need_to_initialize_in_cb
    // is false, but the quasar reduce reads the FULL 16-row face while the reader fills only the populated
    // rows -> the unwritten face rows leak stale L1 into the max (value inflation; masked only when the L1
    // residue happens to be <= the data). Force the -inf pre-clear for MAX pool. -inf is the max identity,
    // so pre-clearing can never change a correct max; the once-at-init clear persists across the in_cb ring
    // because the reader never overwrites those rows. (Real fix: make the quasar reduce respect face_r_dim.)
    constexpr bool force_max_clear = !is_avg_pool;
    // fill the clear cb
    if constexpr (is_avg_pool || need_to_initialize_in_cb || force_max_clear) {
        if constexpr (reader_id == 0) {
            fill_with_val(clear_value_cb.get_write_ptr(), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);
            clear_value_cb.push_back(1);
        }
        if constexpr (reader_id == 1) {
            clear_value_cb.wait_front(1);
        }
        // for average pool clear out tiles runs in loop, no need to initialize here
        if constexpr (!is_avg_pool || !is_large_kernel) {
            if constexpr (is_avg_pool) {
                // QSR avg_pool coherency fix (same class as the MAX fix below): the original pre-clear
                // (clear_out_tiles) copies clear_value_cb -> in_cb via a NoC self-loopback read, which is
                // UNRELIABLE on the Quasar sim (drops / reads stale SRAM) and HANGS the small-kernel avg
                // path. 0 is the avg additive (sum) identity -- padding rows must contribute nothing to the
                // sum -- so the SRAM-coherent NoC zero-write produces the identical clear without the
                // loopback. Once-at-init persists across the in_cb ring because the reader only overwrites
                // the window rows (the unwritten tail rows stay 0 and reduce to a no-op in the sum).
                DataflowBuffer icb_clear(in_cb_id);
                Noc clear_noc;
                clear_noc.async_write_zeros(icb_clear, icb_clear.get_entry_size() * multi_buffering_factor);
                clear_noc.write_zeros_l1_barrier();
            } else {
                // QSR max_pool coherency fix: the pre-clear above (clear_out_tiles) uses a NoC
                // self-loopback read from clear_value_cb into in_cb. On the Quasar sim that self-loopback
                // read is unreliable (drops / reads stale SRAM), so in_cb's rows beyond the window are
                // left holding stale L1. The compute reduces the FULL 4-face (32-row) tile, so that stale
                // L1 leaks into the MAX (e.g. one reader's output pinned to a spurious max). Clear in_cb
                // with the SRAM-coherent NoC zero-write instead (the same primitive FIX #1 / the halo pad
                // use). Compute the region from the DFB object (get_local_cb_interface / get_tile_size are
                // stale for Metal-2.0 DFBs). For MAX the reduce identity must be <= every real window
                // element; the pooled inputs here are non-negative (resnet stem maxpool is post-ReLU), so 0
                // is a safe identity and the once-at-init clear persists across the in_cb ring (the reader
                // only overwrites the window rows). The unwritten tail rows then reduce to a no-op.
                DataflowBuffer icb_clear(in_cb_id);
                Noc clear_noc;
#ifdef ARCH_QUASAR
                // [DIAG — clear-overrun / stale-wr_ptr probe, revert after]: this once-at-init clear writes
                // clrsz = esz*mbf bytes starting at get_write_ptr() (= the DM wr_ptr). It overruns adjacent L1
                // if clrsz > alloc (mbf > nent) OR if wr_ptr is stale from a prior op (not reset to the ring
                // base per launch -- the suspected Bug-1 reconfig escape). Compare across a lone 64c run vs a
                // 64c-after-32c run: wr changing => stale wr_ptr; clrsz>alloc => size overrun. (The 0xDEADBEEF
                // poison was reverted -- a bf16 value can't fault the RISC; the poison fault was this overrun
                // clobbering a control region with garbage instead of benign zeros.)
                DPRINT(
                    "POOLCLR wr={} esz={} mbf={} clrsz={} nent={} alloc={}\n",
                    (uint32_t)icb_clear.get_write_ptr(),
                    (uint32_t)icb_clear.get_entry_size(),
                    (uint32_t)multi_buffering_factor,
                    (uint32_t)(icb_clear.get_entry_size() * multi_buffering_factor),
                    (uint32_t)icb_clear.get_total_num_entries(),
                    (uint32_t)(icb_clear.get_total_num_entries() * icb_clear.get_entry_size()));
#endif
                clear_noc.async_write_zeros(icb_clear, icb_clear.get_entry_size() * multi_buffering_factor);
                clear_noc.write_zeros_l1_barrier();
            }
        }
    }

    // initialize the scalar CB
    if constexpr (reader_id == 0 && one_scalar_per_core) {
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        fill_with_val(in_scalar_cb.get_write_ptr(), FACE_WIDTH, bf16_scalar >> 16);
#ifdef ARCH_QUASAR
        // Quasar sim coherency: the reduce scalar is a CPU-store fill through the DM L1/L2 cache, but the
        // compute reduce reads it directly from TL1. Without write-back compute multiplies by a STALE scalar
        // -> wrong reduce magnitude (the /TILE_HEIGHT scale on the const-channel test) and, if the stale value
        // varies per reduce, decorrelated output (low PCC). Mirrors the in_cb / scratch->out write-backs.
        flush_l2_cache_range(static_cast<uintptr_t>(in_scalar_cb.get_write_ptr()), static_cast<size_t>(FACE_WIDTH) * 2);
#endif
        in_scalar_cb.push_back(1);
    }
    const uint32_t core_nhw_index = get_arg(args::core_nhw_index);

    const uint32_t in_l1_read_base_addr = in_shard_cb.get_read_ptr();
    if constexpr (config_in_dram) {
        if (reader_id == 0) {
            // Inlined load_config_tensor_if_in_dram: the reader-indices tensor flows in via its
            // Metal 2.0 TensorBinding (tensor::reader_indices) instead of a CTA-baked DRAM address.
            Noc cfg_noc;
            const auto reader_indices_accessor = TensorAccessor(tensor::reader_indices);
            cfg_noc.async_read(
                reader_indices_accessor, reader_indices_cb, reader_page_size, {.page_id = core_nhw_index}, {});
            cfg_noc.async_read_barrier();
            reader_indices_cb.push_back(1);
        } else {
            reader_indices_cb.wait_front(1);
        }
    }
    uint32_t reader_indices_l1_addr = reader_indices_cb.get_read_ptr();
    volatile tt_l1_ptr uint32_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_l1_addr);

    uint32_t segments_counter = 1;
    constexpr uint32_t total_elems_to_reduce = kernel_h * kernel_w;

    volatile tt_l1_ptr uint16_t* config_ptr;
    uint32_t scalar_index = 0;
    uint32_t scalar_start;
    uint32_t scalar_value;
    uint32_t scalar_end;
    uint32_t counter = reader_id;
    // HAS_CONFIG <=> !one_scalar_per_core (host-emitted). Gated rather than `if constexpr` because
    // the body references dfb::config_cb / tensor::config, which are only declared when HAS_CONFIG.
#ifdef HAS_CONFIG
    {
        uint32_t config_l1_addr = config_cb.get_read_ptr();
        if constexpr (config_in_dram) {
            if (reader_id == 0) {
                // Inlined load_config_tensor_if_in_dram: the scalar config tensor flows in via its
                // Metal 2.0 TensorBinding (tensor::config) instead of a CTA-baked DRAM address.
                Noc cfg_noc;
                const auto config_accessor = TensorAccessor(tensor::config);
                cfg_noc.async_read(config_accessor, config_cb, config_page_size, {.page_id = core_nhw_index}, {});
                cfg_noc.async_read_barrier();
                config_cb.push_back(1);
            } else {
                config_cb.wait_front(1);
            }
        }
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[0];
        scalar_value = config_ptr[1];
        scalar_end = config_ptr[2];
    }
#endif

    uint16_t num_segments = reader_indices_ptr[0] & 0xffff;
    bool first_row_value = reader_id == 0 || !use_split_reader;

    // [#47797 DEBUG] If POOL hangs at waypoint R, dump the loop-control values. A garbage num_segments
    // (e.g. unwritten reader_indices config) or stride_w==0 makes while(num_segments--)/the inner stride
    // loop spin forever. Compare these against the host sliding-window config for this pool.
    DPRINT(
        "POOL rdr id={} nseg={} strW={} kH={} kW={}\n",
        (uint32_t)reader_id,
        (uint32_t)num_segments,
        (uint32_t)stride_w,
        (uint32_t)kernel_h,
        (uint32_t)kernel_w);

    // This reader's output-stick counter. With split reader, reader0 writes even output rows, reader1
    // writes odd, so global row = 2*counter + reader_id.
    uint32_t out_stick_counter = 0;

    while (num_segments--) {
        uint32_t start_end_segment = reader_indices_ptr[segments_counter++];
        uint16_t start = start_end_segment & 0xffff;
        uint16_t end = start_end_segment >> 16;
        DPRINT("POOL seg start={} end={}\n", (uint32_t)start, (uint32_t)end);  // [#47797 DEBUG]

        if (!first_row_value) {
            start += stride_w;
            first_row_value = true;
        }

        constexpr uint32_t stride_multiple = use_split_reader ? 2 : 1;
        for (uint16_t ind = start; ind <= end; ind += stride_multiple * stride_w) {
            if constexpr (!one_scalar_per_core) {
                fill_scalar<
                    one_scalar_per_core,
                    in_scalar_cb_id,
                    reader_nindices,
                    use_split_reader,
                    multi_buffering_factor>(
                    in_scalar_cb, scalar_start, scalar_end, scalar_value, scalar_index, counter, config_ptr);
            }
            read_kernel_with_top_left_index<
                in_nblocks_c,
                in_cb_id,
                kernel_h,
                kernel_w,
                in_w_padded,
                in_nbytes_leftover,
                in_c,
                max_sticks_for_reduction,
                total_elems_to_reduce,
                is_avg_pool,
                wide_reduction,
                clear_value_cb_id,
                in_cb_ntiles,
                in_nbytes_c,
                shard_width_bytes,
                is_large_kernel,
                last_tile_is_partial,
                dilation_h,
                dilation_w,
                zero_pages,
                in_cb_sz,
                bf16_init_value,
                force_max_tiles_per_reduction_4>(ind, in_l1_read_base_addr);
#if ENABLE_DEBUG_PRINT == 1
            // [DIAG] Peek THIS reader's just-filled input CB (reader is producer; on DM get_read_ptr still
            // points at the base page it filled, before compute pops). Tilized face0 row0 = first 16
            // channels of the window's first row -> should read 1..16 for the deterministic input. If
            // reader1's in_cb_1 reads 0 while reader0's in_cb_0 reads 1..16, the split reader1 input feed
            // is the bug (not the pack). Only the first stick is reliable (rd_ptr stays at base after).
            if (out_stick_counter == 0) {
                DataflowBuffer in_cb_peek(in_cb_id);
                volatile tt_l1_ptr uint16_t* ip =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_cb_peek.get_read_ptr());
                // face0 = tilized rows 0..15, each row = 16 channels; ip[r*16+0] = row r channel 0.
                // Window rows (the 9 sticks) should read the input value (channel 0 -> 1); cleared/tail
                // rows read the pool identity (-inf). Shows whether reader1 fills its window rows at all.
                DPRINT(
                    "INCB rdr={} ind={} in_cb_base={} face0 col0 rows0..15: ",
                    (uint32_t)reader_id,
                    (uint32_t)ind,
                    in_cb_peek.get_read_ptr());
                for (uint32_t r = 0; r < 16; ++r) {
                    DPRINT("{} ", bf16_t(ip[r * 16]));
                }
                DPRINT("\n");
            }
#endif
            // [DEBUG scratch->out workaround] We just fed the input for this output stick; compute reduces
            // it and packs the CORRECT full-tile reduced DEST into our scratch CB (row 0 = all channels).
            // wait_front blocks until that push (ordering for free via the SPSC credit). Then, from this DM
            // core (the only reliable L1 path on the sim), NoC-copy scratch row 0 -> the output tensor at
            // this stick's row, bypassing the broken narrow pack entirely. Then release the scratch page.
            //
            // OUTPUT_TILED (TILE output layout): compute packs straight into the real out_cb (borrowed
            // from the output tensor, via pre_tilize_cb -> tilize_block -> out_cb) and never produces
            // scratch_cb_0/1 in that mode (see compute_pool_2d.cpp's `if constexpr (is_output_tiled)`
            // branch). This whole scratch-consume/NoC-copy workaround exists only to route around the
            // ROW_MAJOR path's broken narrow pack, so it must be skipped here -- otherwise this wait_front
            // blocks forever on a push that will never come (the actual bug behind this fix).
#ifndef OUTPUT_TILED
            scratch_cb.wait_front(scratch_npages);
            {
                const uint32_t global_stick =
                    use_split_reader ? (2u * out_stick_counter + reader_id) : out_stick_counter;
                const uint32_t scratch_row0_addr = scratch_cb.get_read_ptr();  // untilized row 0 = the result
                // Scratch and the borrowed output shard are BOTH local L1 on this core, so the reduced row 0
                // -> output-stick copy is a local L1->L1 move. Do it with a direct pointer copy rather than a
                // NoC self-loopback async_read: on HW both are correct, but the sim's per-stick self-loopback
                // read under multi-core load silently drops/duplicates sticks (the zero-write + adjacent-dup
                // artifacts). A straight L1 copy is race-free and HW-faithful.
                //
                // COHERENCY: the compute packer wrote this stick's reduced row directly to Tensix L1 (TL1),
                // bypassing this DM core's private L1 D$ / shared L2. The scratch CB is single-buffered, so
                // its L1 line address is constant across sticks: after the first read caches it, every later
                // read hits the STALE cached copy (all sticks would read stick 0's result). On HW the reader
                // must invalidate any address another agent wrote before reading it (invalidate_l1_cache() is
                // a no-op on Quasar DM). Invalidate the scratch row's L2+L1D lines so the load re-fetches the
                // freshly packed data from TL1. The prior NoC-read path avoided this because the NoC engine
                // reads TL1 directly (non-snooping), never through the DM cache. Arch-split: Quasar (tt-2xx)
                // has invalidate_l2_cache_range; WH/BH have no L2, so invalidate_l1_cache() (equivalent effect).
#ifdef ARCH_QUASAR
                invalidate_l2_cache_range(scratch_row0_addr, out_row_bytes);
#else
                invalidate_l1_cache();
#endif
                const uint32_t out_dst_addr = out_shard_cb.get_write_ptr() + global_stick * out_row_bytes;
                volatile tt_l1_ptr uint32_t* src_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_row0_addr);
                volatile tt_l1_ptr uint32_t* dst_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_dst_addr);
                for (uint32_t w = 0; w < out_row_bytes >> 2; ++w) {
                    dst_w[w] = src_w[w];
                }
                // Write-back the copied output row to TL1 so the host device->host read-back (and any NoC
                // consumer) sees it, not this DM core's cached copy. Without this the output stick keeps its
                // pre-kernel stale L1 -> the got.max=2.0 leak. Write-side analog of the scratch invalidate
                // above and the in_cb write-back in read_kernel_with_top_left_index. (Quasar tt-2xx L2; WH/BH
                // have no L2 so the write is already visible to the NoC engine.)
#ifdef ARCH_QUASAR
                flush_l2_cache_range(reinterpret_cast<uintptr_t>(dst_w), static_cast<size_t>(out_row_bytes));
#endif
                // DEBUG (compute-vs-read locator): dump the scratch row-0 (reduced result) the reader sees,
                // limited to the first few global sticks to avoid flooding/crashing the dprint server.
                // Distinct sensible values per stick => compute/reduce/pack is fine and the bug is in the
                // out-copy/assembly; constant/garbage (e.g. 2.0) => compute-side or a fixed/stale read.
                if (global_stick < 4u) {
                    volatile tt_l1_ptr uint16_t* rp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_row0_addr);
                    DPRINT(
                        "SCRATCH2OUT rdr={} gstick={} rdptr={} npages={} row0[0..7]: ",
                        (uint32_t)reader_id,
                        global_stick,
                        scratch_row0_addr,
                        (uint32_t)scratch_npages);
                    for (uint32_t j = 0; j < 8; ++j) {
                        DPRINT("{} ", bf16_t(rp[j]));
                    }
                    DPRINT(" [60..63]: ");
                    for (uint32_t j = 60; j < 64; ++j) {
                        DPRINT("{} ", bf16_t(rp[j]));
                    }
                    DPRINT("\n");
                }
            }
            scratch_cb.pop_front(scratch_npages);
#endif  // !OUTPUT_TILED
            out_stick_counter++;
            if (use_split_reader && ind == end) {
                first_row_value = false;
            }
        }
    }
}  // kernel_main()
