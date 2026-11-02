// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ring_attention_slice_readiness.hpp"
#include "ring_attention_kv_window.hpp"
#include "ring_attention_all_gather_metadata.hpp"
#include "ring_attention_rank_mapping.hpp"
#include <algorithm>
#include <cstdint>
#include <utility>

using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

enum CompileTimeArg : uint32_t {
    kMyTransportRank,
    kReservedPacketHeaderCbId,
    kCbOutputId,
    kPacketSizeInPages,
    kOutputPageSize,
    kNumTargetsForwardDirection,
    kNumTargetsBackwardDirection,
    kFuseOp,
    kTopology,
    kNumInputs,
    kDirection,
    kUnicastRouteArg0,
    kUnicastRouteArg1,
    kHasMetadata,
    kCbMetaId,
    kNumLinks,
    kSplitForwardingEnabled,
    kFullMeshRankMapping,
    kSnakeOrientation,
    kMeshRows,
    kMeshCols,
    kPartialReadinessEnabled,
    kOutputBankOwnedSchedule,
    kNumDramBanks,
    // Slice readiness (ring_attention_slice_readiness.hpp): K-split band count (0 = off) and K chunk tile rows.
    kSliceReadinessBands,
    kSliceReadinessChunkTiles,
    // Rotating KV window (ring_attention_kv_window.hpp): parts per direction at the largest slice (0 = off), tile rows
    // of such a part, the consumers' progress semaphore and the scratch CB a link-0 writer polls it into.
    kKvWindowParts,
    kKvWindowCapRows,
    kKvWindowProgressSem,
    kKvWindowScratchCb,
    kNumFixedCompileTimeArgs,
};

constexpr uint32_t my_transport_rank = get_compile_time_arg_val(kMyTransportRank);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(kReservedPacketHeaderCbId);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(kCbOutputId);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(kPacketSizeInPages);
constexpr uint32_t output_page_size = get_compile_time_arg_val(kOutputPageSize);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(kNumTargetsForwardDirection);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(kNumTargetsBackwardDirection);
constexpr bool fuse_op = get_compile_time_arg_val(kFuseOp);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(kTopology));
constexpr uint32_t num_inputs = get_compile_time_arg_val(kNumInputs);
constexpr bool direction = get_compile_time_arg_val(kDirection);  // 1 is forward, 0 is backward
constexpr uint32_t unicast_route_arg0 = get_compile_time_arg_val(kUnicastRouteArg0);
constexpr uint32_t unicast_route_arg1 = get_compile_time_arg_val(kUnicastRouteArg1);
// Trace-safe metadata path: when set, the writer recomputes the gather extent (valid_pages) on-device
// from kv_actual_isl[0] (a 1-element uint32 DRAM tensor) so it stays matched to the reader's on-device
// recompute (else they desync under a placeholder host logical_n). When false neither this nor the
// metadata accessor is emitted.
constexpr bool has_metadata = get_compile_time_arg_val(kHasMetadata);
constexpr uint32_t cb_meta_id = get_compile_time_arg_val(kCbMetaId);
constexpr uint32_t num_links = get_compile_time_arg_val(kNumLinks);
constexpr bool split_forwarding_enabled = get_compile_time_arg_val(kSplitForwardingEnabled);
constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(kFullMeshRankMapping);
constexpr auto snake_orientation =
    static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(kSnakeOrientation));
constexpr uint32_t mesh_rows = get_compile_time_arg_val(kMeshRows);
constexpr uint32_t mesh_cols = get_compile_time_arg_val(kMeshCols);
constexpr bool partial_readiness_enabled = get_compile_time_arg_val(kPartialReadinessEnabled);
constexpr bool output_bank_owned_schedule = get_compile_time_arg_val(kOutputBankOwnedSchedule);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(kNumDramBanks);
constexpr uint32_t slice_readiness_bands = get_compile_time_arg_val(kSliceReadinessBands);
constexpr uint32_t slice_readiness_chunk_tiles = get_compile_time_arg_val(kSliceReadinessChunkTiles);
constexpr bool slice_readiness_enabled = slice_readiness_bands > 0;
constexpr uint32_t kv_window_parts = get_compile_time_arg_val(kKvWindowParts);
constexpr uint32_t kv_window_cap_rows = get_compile_time_arg_val(kKvWindowCapRows);
constexpr uint32_t kv_window_progress_sem = get_compile_time_arg_val(kKvWindowProgressSem);
constexpr uint32_t kv_window_scratch_cb = get_compile_time_arg_val(kKvWindowScratchCb);
constexpr bool kv_window = kv_window_parts > 0;
// Full slices a direction carries (its writers' local slice and full relays; the diametric half comes on top).
constexpr uint32_t kv_window_full_slices(bool dir) {
    return dir ? num_targets_backward_direction : num_targets_forward_direction - 1;
}
constexpr uint32_t kv_window_units =
    kv_window_full_slices(direction) * ttnn::ring_attention_all_gather::kReadinessParts;
static_assert(!kv_window || (slice_readiness_enabled && split_forwarding_enabled && fuse_op));
static_assert(
    !slice_readiness_enabled || (output_bank_owned_schedule && !partial_readiness_enabled),
    "slice readiness needs the bank-owned schedule and excludes midpoint partial readiness");

static_assert(!partial_readiness_enabled || num_inputs == 1, "partial readiness requires one gathered input");
static_assert(
    !partial_readiness_enabled || !split_forwarding_enabled,
    "partial readiness and split forwarding are mutually exclusive");
static_assert(
    !partial_readiness_enabled || (num_targets_forward_direction > 0 && num_targets_backward_direction > 0),
    "partial readiness requires both ring directions");

// Rotating KV window (ring_attention_kv_window.hpp), writer side. A writer lands a unit only once the downstream has a
// free slot for it: the downstream's link-0 writer of the other direction adds credits to this core's otherwise unused
// semaphore word of the other direction, rotating units in the low half and diametric parts in the high half.
struct KvWindowCredits {
    uint32_t addr = 0;
    uint32_t value() const {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(this->addr);
    }
};

// Link 0's writer also returns the credits of the other direction's window, upstream over its own fabric connection
// (it sends toward that window's upstream). It polls the progress word of every consumer of the window: the SDPA cores
// the gather signals and that direction's gather readers, which relay from it. A slot is free once the slowest of them
// has moved past the unit in it.
struct KvWindowMaster {
    bool active = false;
    uint32_t window_dir = 0;
    uint32_t units = 0;  // rotating units per direction
    uint32_t rot_slots = 0;
    uint32_t diam_slots = 0;
    uint32_t rot_sent = 0;
    uint32_t diam_sent = 0;
    uint32_t credit_addr = 0;  // the credit word on the upstream writers
    uint32_t num_link_cores = 0;
    uint32_t link_cores_arg = 0;  // runtime arg index of the window direction's worker cores, x | y << 16 each
    uint32_t progress_addr = 0;
    uint32_t scratch = 0;
    const OpSignalerT<true>* sdpa = nullptr;
    volatile PACKET_HEADER_TYPE* pkt_hdr = nullptr;
    tt::tt_fabric::WorkerToFabricEdmSender* connection = nullptr;

    bool done() const {
        return this->rot_sent == this->units && this->diam_sent == ttnn::ring_attention_all_gather::kReadinessParts;
    }

    // Polls every consumer and returns the freed slots' credits; the writer calls it whenever it waits.
    void service(Noc& noc) {
        if (!this->active || this->done()) {
            return;
        }
        namespace kvw = ttnn::ring_attention_all_gather;
        const uint32_t num_sdpa = this->sdpa->num_fused_op_cores_to_signal;
        const uint32_t count = num_sdpa + this->num_link_cores;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t x, y;
            if (i < num_sdpa) {
                x = this->sdpa->signal_core_x(i);
                y = this->sdpa->signal_core_y(i);
            } else {
                const uint32_t packed = get_arg_val<uint32_t>(this->link_cores_arg + i - num_sdpa);
                x = packed & 0xFFFF;
                y = packed >> 16;
            }
            noc_async_read(get_noc_addr(x, y, this->progress_addr), this->scratch + i * L1_ALIGNMENT, sizeof(uint32_t));
        }
        noc_async_read_barrier();
        uint32_t rot_free = this->units;
        uint32_t diam_free = kvw::kReadinessParts;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t progress = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(this->scratch + i * L1_ALIGNMENT);
            rot_free = std::min(rot_free, kvw::kv_window_field(progress, this->window_dir));
            diam_free = std::min(diam_free, kvw::kv_window_field(progress, 2));
        }
        const uint32_t rot_target = std::min(rot_free + this->rot_slots, this->units);
        const uint32_t diam_target = std::min(diam_free + this->diam_slots, kvw::kReadinessParts);
        if (rot_target <= this->rot_sent && diam_target <= this->diam_sent) {
            return;
        }
        const uint32_t credits = (rot_target - this->rot_sent) | (diam_target - this->diam_sent)
                                                                     << kvw::kKvWindowDiamCreditShift;
        for (uint32_t link = 0; link < this->num_link_cores; ++link) {
            const uint32_t packed = get_arg_val<uint32_t>(this->link_cores_arg + link);
            this->pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(packed & 0xFFFF, packed >> 16, this->credit_addr, 0), credits});
            this->connection->wait_for_empty_write_slot();
            this->connection->send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(this->pkt_hdr), sizeof(PACKET_HEADER_TYPE));
        }
        this->rot_sent = rot_target;
        this->diam_sent = diam_target;
    }
};

constexpr auto kNoIdle = [] {};

// `idle` runs while the packet CB is empty (the window's credit master keeps returning credits meanwhile).
template <typename AddrGenType, typename FabricSender, typename IdleFn = decltype(kNoIdle)>
FORCE_INLINE void write_bank_owned_slices(
    CircularBuffer& cb_output,
    AddrGenType& output_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender& fabric_direction_connection,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride,
    const IdleFn& idle = kNoIdle) {
    ring_attention_all_gather::BankOwnedPacketSchedule<num_dram_banks> schedule(
        output_page_base, valid_pages, first_bank, bank_stride, packet_size_in_pages);
    uint32_t first_page_offset = 0;
    uint32_t batch = 0;
    while (schedule.next_packet(first_page_offset, batch)) {
        if constexpr (kv_window) {
            while (!cb_output.pages_available_at_front(packet_size_in_pages)) {
                idle();
            }
        }
        cb_output.wait_front(packet_size_in_pages);
        fabric_write_unidir(
            output_page_base + first_page_offset,
            output_addrgen,
            pkt_hdr,
            fabric_direction_connection,
            cb_output.get_read_ptr(),
            batch * output_page_size);
        cb_output.pop_front(packet_size_in_pages);
    }
}

FORCE_INLINE void discard_bank_owned_slices(
    CircularBuffer& cb_output,
    uint32_t output_page_base,
    uint32_t valid_pages,
    uint32_t first_bank,
    uint32_t bank_stride) {
    for (uint32_t bank = first_bank; bank < num_dram_banks; bank += bank_stride) {
        const uint32_t page_count =
            ring_attention_all_gather::get_bank_owned_slice(output_page_base, valid_pages, bank, num_dram_banks)
                .page_count;
        for (uint32_t pages_discarded = 0; pages_discarded < page_count; pages_discarded += packet_size_in_pages) {
            cb_output.wait_front(packet_size_in_pages);
            cb_output.pop_front(packet_size_in_pages);
        }
    }
}

// A full slice in readiness parts (ring_attention_slice_readiness.hpp), in the reader's prefetch order: part p of every
// input and batch-head, then a marker, for every part but the last (the caller's completion marker closes it).
// Window mode (kv_window): part p lands packed in the downstream's window from tile row window_row(p), after
// before_part(p) has its credit; bank_offset / bank_stride select this direction's half of a split diametric slice.
template <typename AddrGens, typename FabricSender, typename SignalFn, typename RowFn, typename PartFn, typename IdleFn>
FORCE_INLINE void write_slice_in_parts(
    Noc& noc,
    CircularBuffer& cb_output,
    AddrGens& output_addrgens,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    FabricSender* fabric_direction_connection,
    const SignalFn& signal_output_ready,
    const std::array<uint32_t, num_inputs>& input_tensor_Wt,
    const std::array<uint32_t, num_inputs>& input_tensor_Ht,
    const std::array<uint32_t, num_inputs>& output_tensor_Wt,
    const std::array<uint32_t, num_inputs>& output_tensor_Ht,
    const std::array<uint32_t, num_inputs>& input_batch_head_count,
    const std::array<uint32_t, num_inputs>& input_valid_pages,
    const std::array<uint32_t, num_inputs>& worker_link,
    uint32_t slice_tensor_rank,
    uint32_t num_chunks,
    uint32_t bank_offset,
    uint32_t bank_stride,
    const RowFn& window_row,
    const PartFn& before_part,
    const IdleFn& idle) {
    for (uint32_t part = 0; part < ttnn::ring_attention_all_gather::kReadinessParts; ++part) {
        if constexpr (kv_window) {
            before_part(part);
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            const uint32_t row_pages = input_tensor_Wt[input_idx];
            const uint32_t input_pages_per_batch_head = row_pages * input_tensor_Ht[input_idx];
            const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
            const uint32_t valid_rows = input_valid_pages[input_idx] / row_pages;
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                uint32_t output_page_base =
                    bh_idx * output_pages_per_batch_head + slice_tensor_rank * input_pages_per_batch_head;
                // Window mode: the part's ranges pack from the slot's first row, in send order.
                uint32_t slot_row = 0;
                if constexpr (kv_window) {
                    output_page_base = (bh_idx * output_tensor_Ht[input_idx] + window_row(part)) * row_pages;
                }
                for (uint32_t range = 0; range < slice_readiness_bands; ++range) {
                    const auto rows = ttnn::ring_attention_all_gather::readiness_rows(
                        num_chunks,
                        slice_readiness_bands,
                        slice_readiness_chunk_tiles,
                        valid_rows,
                        part,
                        range,
                        slice_tensor_rank);
                    if (rows.end <= rows.begin) {
                        continue;
                    }
                    const uint32_t first_page = output_page_base + (kv_window ? slot_row : rows.begin) * row_pages;
                    slot_row += rows.end - rows.begin;
                    if (fabric_direction_connection != nullptr) {
                        write_bank_owned_slices(
                            cb_output,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            first_page,
                            (rows.end - rows.begin) * row_pages,
                            worker_link[input_idx] + bank_offset,
                            bank_stride,
                            idle);
                    } else {
                        discard_bank_owned_slices(
                            cb_output,
                            first_page,
                            (rows.end - rows.begin) * row_pages,
                            worker_link[input_idx] + bank_offset,
                            bank_stride);
                    }
                }
            }
        }
        if (part + 1 < ttnn::ring_attention_all_gather::kReadinessParts) {
            // Flush this part's writes before its marker.
            noc.async_write_barrier();
            signal_output_ready();
        }
    }
}

void kernel_main() {
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, kNumFixedCompileTimeArgs + num_inputs>();
    constexpr uint32_t kMetaArgsOffset = has_metadata
                                             ? std::get<num_inputs - 1>(outputs_args).next_compile_time_args_offset()
                                             : (kNumFixedCompileTimeArgs + num_inputs);
    constexpr auto meta_args = TensorAccessorArgs<kMetaArgsOffset>();
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // KEEP IN SYNC with kWriterRuntimeArgHeaderCount/kTensorDescriptorFieldCount and field offsets in
    // ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp. Fused consumers patch
    // selected fields through that shared host-side layout.
    uint32_t arg_idx = 0;
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    // The other direction's semaphore, unused on this core: the window's credit word (kv_window).
    [[maybe_unused]] const uint32_t kv_window_credit_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_tensor_Wt;
    std::array<uint32_t, num_inputs> input_tensor_Ht;
    std::array<uint32_t, num_inputs> output_tensor_Wt;
    std::array<uint32_t, num_inputs> output_tensor_Ht;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;
    std::array<uint32_t, num_inputs> input_valid_pages;
    std::array<uint32_t, num_inputs> worker_link;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        // input_batch_base is reader-only. The writer always targets output
        // slot 0, so it reads the arg here only for alignment.
        (void)get_arg_val<uint32_t>(arg_idx++);
        // valid_pages_per_batch_head (slot 6): clamp the gather to the logical_n-valid slab prefix (must
        // match the reader's clamp so cb_output producer/consumer page counts stay aligned). Default
        // (full input) leaves the range unchanged.
        const uint32_t valid_pages = get_arg_val<uint32_t>(arg_idx++);
        worker_link[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        if constexpr (has_metadata) {
            (void)get_arg_val<uint32_t>(arg_idx++);  // input_cache_batch_extent: reader-only
        }
        input_valid_pages[input_idx] = valid_pages;
        const auto link_page_range =
            ring_attention_all_gather::compute_link_page_range(valid_pages, num_links, worker_link[input_idx]);
        input_tile_id_start[input_idx] = link_page_range.start;
        input_tile_id_end[input_idx] = link_page_range.end;
    }

    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);

    // Trace-safe metadata path: recompute the gather extent (valid_pages) on-device from kv_actual_isl[0]
    // so it matches the reader's recompute even when the host logical_n is a placeholder. The
    // kv_actual_isl DRAM address and chunk_local_tiles are the next two runtime args (after the
    // output-buffer addrs, before the fabric args). Identical formula to the all-gather reader / host
    // compute_gather_valid_Ht.
    if constexpr (has_metadata) {
        // kv_actual_isl is a 1-element uint32 DRAM tensor (was metadata[1]); read its page 0.
        const uint32_t kv_actual_isl_addr = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t chunk_local_tiles = get_arg_val<uint32_t>(arg_idx++);
        Noc meta_noc;
        CircularBuffer cb_meta(cb_meta_id);
        // Shared read protocol (async_read page 0 -> barrier -> invalidate_l1_cache -> volatile load). The
        // invalidate is required, not cosmetic: this tensor is at a fixed DRAM address the host refreshes in
        // place between trace replays, so a cached L1 line would return the prior chunk's kv_actual_isl and
        // silently clamp the gather to the wrong prefix.
        const uint32_t kv_actual = trace_metadata::read_metadata_scalar_u32(
            meta_noc, meta_args, kv_actual_isl_addr, cb_meta.get_write_ptr());  // kv_actual_isl (tile-aligned)
        uint32_t cache_local_tile_rows = input_tensor_Ht[0];
        for (uint32_t input_idx = 1; input_idx < num_inputs; ++input_idx) {
            cache_local_tile_rows = std::min(cache_local_tile_rows, input_tensor_Ht[input_idx]);
        }
        // Same formula the reader uses -- both MUST clamp to the same slab prefix or the cb_output
        // producer/consumer page counts drift (see the header's KEEP IN SYNC note).
        const uint32_t gather_valid_Ht = ring_attention_all_gather::compute_gather_valid_Ht(
            kv_actual, chunk_local_tiles, ring_size, cache_local_tile_rows);
        ring_attention_all_gather::update_link_page_ranges_for_gather_extent(
            gather_valid_Ht,
            num_links,
            input_tensor_Wt,
            input_valid_pages,
            worker_link,
            input_tile_id_start,
            input_tile_id_end);
    }

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
    /* Args for overlapped all gather */
    OpSignalerT</*packed_signal_coords=*/true> op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignalerT<true>(arg_idx);
    }
    // Window mode: [count, the window direction's worker cores x | y << 16] after the signaler args, link 0 only.
    constexpr uint32_t kvw_parts = ttnn::ring_attention_all_gather::kReadinessParts;
    [[maybe_unused]] KvWindowCredits kv_credits;
    [[maybe_unused]] KvWindowMaster kv_master;
    // This launch's slots: the window cut into parts of its own slices' size (every input has the same rows).
    [[maybe_unused]] const ttnn::ring_attention_all_gather::KvWindowLayout kv_layout =
        kv_window ? ttnn::ring_attention_all_gather::kv_window_layout(
                        input_valid_pages[0] / input_tensor_Wt[0],
                        slice_readiness_bands,
                        slice_readiness_chunk_tiles,
                        kv_window_parts,
                        kv_window_cap_rows)
                  : ttnn::ring_attention_all_gather::KvWindowLayout{};
    if constexpr (kv_window) {
        kv_master.rot_slots = kv_layout.rot_slots;
        kv_master.diam_slots = kv_layout.diam_slots;
        kv_credits.addr = kv_window_credit_sem;
        kv_master.num_link_cores = get_arg_val<uint32_t>(arg_idx++);
        kv_master.link_cores_arg = arg_idx;
        kv_master.active = kv_master.num_link_cores > 0;
        kv_master.window_dir = direction ? 0 : 1;
        kv_master.units = kv_window_full_slices(!direction) * kvw_parts;
        kv_master.credit_addr = out_ready_sem;
        kv_master.progress_addr = get_semaphore(kv_window_progress_sem);
        kv_master.scratch = get_write_ptr(kv_window_scratch_cb);
        kv_master.sdpa = &op_signaler_sender;
    }

    Noc noc_obj;
    CircularBuffer cb_packet_header(reserved_packet_header_cb_id);
    CircularBuffer cb_output(cb_output_id);

    // packet header cb
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_addr = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    cb_packet_header.reserve_back(1);
    auto packet_header_buffer_seminc = cb_packet_header.get_write_ptr();
    cb_packet_header.push_back(1);
    if constexpr (kv_window) {
        cb_packet_header.reserve_back(1);
        kv_master.pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_packet_header.get_write_ptr());
        cb_packet_header.push_back(1);
    }

    // pre-populate packet headers
    constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info = {
        .dst_mesh_id = static_cast<uint16_t>(unicast_route_arg0),
        .dst_chip_id = static_cast<uint16_t>(unicast_route_arg1)};

    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr, unicast_route_info);

    fabric_connection.open();
    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection =
        fabric_connection.is_logically_connected() ? (direction == 1 ? &fabric_connection.get_backward_connection()
                                                                     : &fabric_connection.get_forward_connection())
                                                   : nullptr;
    constexpr uint32_t num_targets_in_direction =
        direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;
    if constexpr (kv_window) {
        ccl_routing_utils::fabric_set_line_unicast_route(kv_master.pkt_hdr, unicast_route_info);
        kv_master.connection = fabric_direction_connection;
    }
    namespace kvw = ttnn::ring_attention_all_gather;
    // Window mode: the master returns credits while this writer waits, for credits of its own or for packets.
    const auto kv_idle = [&]() {
        if constexpr (kv_window) {
            kv_master.service(noc_obj);
        }
    };
    // Unit `first_unit + part` of this direction may land once the downstream has credited it.
    const auto kv_rot_credit = [&](uint32_t first_unit) {
        return [&, first_unit](uint32_t part) {
            while ((kv_credits.value() & 0xFFFF) < first_unit + part + 1) {
                kv_idle();
            }
        };
    };
    const auto kv_rot_row = [&](uint32_t first_unit) {
        return [&, first_unit](uint32_t part) { return kv_layout.rot_row(direction, first_unit + part); };
    };

    const uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});
    ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_sem_inc, unicast_route_info);
    const auto signal_output_ready = [&]() {
        if constexpr (num_targets_in_direction) {
            fabric_direction_connection->wait_for_empty_write_slot();
            fabric_direction_connection->send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
    };

    uint32_t slice_writes = 0;
    constexpr uint32_t my_tensor_rank =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            my_transport_rank, mesh_rows, mesh_cols, snake_orientation);

    if constexpr (slice_readiness_enabled) {
        if constexpr (fuse_op && direction == 1) {
            // The "local slice available" signal, sent up front: the consumer only counts it in its direction-0
            // thresholds, and sent last it would delay the first arriving slice's part-0 threshold.
            op_signaler_sender.synchronize_workers_and_signal_op(my_transport_rank);
        }
        // The local slice in readiness parts.
        write_slice_in_parts(
            noc_obj,
            cb_output,
            output_addrgens,
            pkt_hdr,
            num_targets_in_direction ? fabric_direction_connection : nullptr,
            signal_output_ready,
            input_tensor_Wt,
            input_tensor_Ht,
            output_tensor_Wt,
            output_tensor_Ht,
            input_batch_head_count,
            input_valid_pages,
            worker_link,
            my_tensor_rank,
            ttnn::ring_attention_all_gather::readiness_split_chunks(
                input_valid_pages[0] / input_tensor_Wt[0], slice_readiness_chunk_tiles),
            0,
            num_links,
            kv_rot_row(0),
            kv_rot_credit(0),
            kv_idle);
    } else {
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            /**
             * Write out the local slice to forward and backward devices
             * Note that it is not copied to local output buffer. This is because
             * the fused op (RingJointAttention) reads from the input buffer directly
             * when accessing the local slice. This is a performance optimization
             * to remove startup latency from the fused op.
             */

            const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
            const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
            if constexpr (output_bank_owned_schedule) {
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                    const uint32_t output_page_base =
                        bh_idx * output_pages_per_batch_head + my_tensor_rank * input_pages_per_batch_head;
                    if constexpr (num_targets_in_direction) {
                        if constexpr (partial_readiness_enabled) {
                            const uint32_t first_pages = ring_attention_all_gather::midpoint_prefix_pages(
                                input_valid_pages[input_idx], input_tensor_Wt[input_idx]);
                            write_bank_owned_slices(
                                cb_output,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                output_page_base,
                                first_pages,
                                worker_link[input_idx],
                                num_links);
                            signal_output_ready();
                            write_bank_owned_slices(
                                cb_output,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                output_page_base + first_pages,
                                input_valid_pages[input_idx] - first_pages,
                                worker_link[input_idx],
                                num_links);
                        } else {
                            write_bank_owned_slices(
                                cb_output,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                output_page_base,
                                input_valid_pages[input_idx],
                                worker_link[input_idx],
                                num_links);
                        }
                    } else {
                        discard_bank_owned_slices(
                            cb_output,
                            output_page_base,
                            input_valid_pages[input_idx],
                            worker_link[input_idx],
                            num_links);
                    }
                }
            } else {
                uint32_t tile_id_start = my_tensor_rank * input_tensor_Wt[input_idx];
                uint32_t pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                uint32_t row_offset =
                    (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                uint32_t tiles_read = input_tile_id_start[input_idx];
                uint32_t tiles_to_read = input_tile_id_end[input_idx];
                if (gather_dim == 3) {
                    tile_id_start = my_tensor_rank * input_tensor_Wt[input_idx];
                } else {
                    tile_id_start = my_tensor_rank * input_pages_per_batch_head;
                }

                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    while (tiles_read < tiles_to_read) {
                        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                        cb_output.wait_front(packet_size_in_pages);
                        size_t l1_read_addr = cb_output.get_read_ptr();

                        uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;

                        pages_read_in_row++;
                        if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                            row_offset += output_tensor_Wt[input_idx];
                            pages_read_in_row = 0;
                        }

                        if (num_pages_to_read == 2) {
                            uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;

                            if constexpr (num_targets_in_direction) {
                                scatter_fabric_write_unidir(
                                    tile_id,
                                    second_tile_id,
                                    output_addrgens[input_idx],
                                    pkt_hdr,
                                    *fabric_direction_connection,
                                    l1_read_addr,
                                    output_page_size);
                            }

                            pages_read_in_row++;
                            if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                                row_offset += output_tensor_Wt[input_idx];
                                pages_read_in_row = 0;
                            }
                        } else {
                            ASSERT(num_pages_to_read == 1);

                            if constexpr (num_targets_in_direction) {
                                fabric_write_unidir(
                                    tile_id,
                                    output_addrgens[input_idx],
                                    pkt_hdr,
                                    *fabric_direction_connection,
                                    l1_read_addr,
                                    output_page_size);
                            }
                        }

                        tiles_read += num_pages_to_read;
                        cb_output.pop_front(packet_size_in_pages);
                    }
                    tile_id_start += output_pages_per_batch_head;
                    tiles_read = input_tile_id_start[input_idx];
                    tiles_to_read = input_tile_id_end[input_idx];
                    pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
                    row_offset =
                        (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                }
            }
        }
    }

    noc_obj.async_write_barrier();
    // increment locally
    if constexpr (fuse_op && direction == 1 && !slice_readiness_enabled) {
        /**
         * Synchronize and signal that the local tensor slice is available
         *
         * While the fused op will not wait on this "local write done" increment,
         * the fused op signaler will account for it in future waits.
         */
        op_signaler_sender.synchronize_workers_and_signal_op(my_transport_rank);
    }

    // The completion marker is the only marker in the legacy protocol and the second marker in
    // partial-readiness mode. num_hops=1 is correct for both supported Fabric topologies.
    signal_output_ready();

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if constexpr (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    // On an even ring the terminal relayed slice (the downstream neighbor's diametric shard) is relayed
    // half per direction. The gate is a host-derived compile-time flag (see top of file).
    if (split_forwarding_enabled && direction == 1) {
        writes_expected++;
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times

        int slice_transport_rank_signed;
        uint32_t slice_transport_rank;
        if constexpr (direction == 1) {
            slice_transport_rank_signed = my_transport_rank + slice_writes + 1;
            slice_transport_rank = (slice_transport_rank_signed >= (int)ring_size)
                                       ? slice_transport_rank_signed - ring_size
                                       : slice_transport_rank_signed;
        } else {
            slice_transport_rank_signed = my_transport_rank - slice_writes - 1;
            slice_transport_rank = (slice_transport_rank_signed < 0) ? ring_size + slice_transport_rank_signed
                                                                     : slice_transport_rank_signed;
        }
        const bool is_split_forwarded_slice = split_forwarding_enabled && (slice_writes == writes_expected - 1);
        const uint32_t slice_tensor_rank =
            ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                slice_transport_rank, mesh_rows, mesh_cols, snake_orientation);
        if (slice_readiness_enabled && (kv_window || !is_split_forwarded_slice)) {
            // A full relayed slice in readiness parts. Window mode sends the diametric half in parts too, into the
            // downstream's diametric slots: this direction's banks of each.
            write_slice_in_parts(
                noc_obj,
                cb_output,
                output_addrgens,
                pkt_hdr,
                fabric_direction_connection,
                signal_output_ready,
                input_tensor_Wt,
                input_tensor_Ht,
                output_tensor_Wt,
                output_tensor_Ht,
                input_batch_head_count,
                input_valid_pages,
                worker_link,
                slice_tensor_rank,
                ttnn::ring_attention_all_gather::readiness_split_chunks(
                    input_valid_pages[0] / input_tensor_Wt[0], slice_readiness_chunk_tiles),
                is_split_forwarded_slice ? direction * num_links : 0,
                is_split_forwarded_slice ? 2 * num_links : num_links,
                [&](uint32_t part) {
                    return is_split_forwarded_slice ? kv_layout.diam_row(part)
                                                    : kv_rot_row((slice_writes + 1) * kvw_parts)(part);
                },
                [&](uint32_t part) {
                    if (is_split_forwarded_slice) {
                        while ((kv_credits.value() >> kvw::kKvWindowDiamCreditShift) < part + 1) {
                            kv_idle();
                        }
                    } else {
                        kv_rot_credit((slice_writes + 1) * kvw_parts)(part);
                    }
                },
                kv_idle);
            signal_output_ready();
            slice_writes++;
            continue;
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            const uint32_t input_pages_per_batch_head = input_tensor_Wt[input_idx] * input_tensor_Ht[input_idx];
            const uint32_t output_pages_per_batch_head = output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
            if constexpr (output_bank_owned_schedule) {
                const uint32_t split_factor = is_split_forwarded_slice ? 2 : 1;
                const uint32_t first_bank =
                    worker_link[input_idx] + (is_split_forwarded_slice ? direction * num_links : 0);
                const uint32_t bank_stride = num_links * split_factor;
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; ++bh_idx) {
                    const uint32_t output_page_base =
                        bh_idx * output_pages_per_batch_head + slice_tensor_rank * input_pages_per_batch_head;
                    if constexpr (partial_readiness_enabled) {
                        const uint32_t first_pages = ring_attention_all_gather::midpoint_prefix_pages(
                            input_valid_pages[input_idx], input_tensor_Wt[input_idx]);
                        write_bank_owned_slices(
                            cb_output,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            output_page_base,
                            first_pages,
                            first_bank,
                            bank_stride);
                        signal_output_ready();
                        write_bank_owned_slices(
                            cb_output,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            output_page_base + first_pages,
                            input_valid_pages[input_idx] - first_pages,
                            first_bank,
                            bank_stride);
                    } else {
                        write_bank_owned_slices(
                            cb_output,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            output_page_base,
                            input_valid_pages[input_idx],
                            first_bank,
                            bank_stride);
                    }
                }
            } else {
                const uint32_t total_pages = input_tile_id_end[input_idx] - input_tile_id_start[input_idx];
                const uint32_t num_packets = (total_pages + packet_size_in_pages - 1) / packet_size_in_pages;
                const uint32_t first_half_pages = (num_packets / 2) * packet_size_in_pages;
                uint32_t relay_start = input_tile_id_start[input_idx];
                uint32_t relay_end = input_tile_id_end[input_idx];
                if (is_split_forwarded_slice) {
                    if constexpr (direction == 0) {
                        relay_end = relay_start + first_half_pages;
                    } else {
                        relay_start += first_half_pages;
                    }
                }

                uint32_t tiles_read = relay_start;
                uint32_t tiles_to_read = relay_end;
                uint32_t tile_id_start = slice_tensor_rank * input_tensor_Wt[input_idx];
                const uint32_t slice_Wt = input_tensor_Wt[input_idx];
                const uint32_t stride_Wt = output_tensor_Wt[input_idx];
                uint32_t row_offset = (relay_start / slice_Wt) * stride_Wt;
                uint32_t pages_read_in_row = relay_start % slice_Wt;

                if (gather_dim == 3) {
                    tile_id_start = slice_tensor_rank * slice_Wt;
                } else {
                    tile_id_start = slice_tensor_rank * input_pages_per_batch_head;
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                    while (tiles_read < tiles_to_read) {
                        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                        cb_output.wait_front(packet_size_in_pages);
                        size_t l1_read_addr = cb_output.get_read_ptr();
                        uint32_t first_tile_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                        if (num_pages_to_read == 2) {
                            uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                            pages_read_in_row++;
                            if (pages_read_in_row >= slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = 0;
                            }

                            scatter_fabric_write_unidir(
                                first_tile_id,
                                second_tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        } else {
                            ASSERT(num_pages_to_read == 1);
                            fabric_write_unidir(
                                first_tile_id,
                                output_addrgens[input_idx],
                                pkt_hdr,
                                *fabric_direction_connection,
                                l1_read_addr,
                                output_page_size);
                        }

                        tiles_read += num_pages_to_read;
                        cb_output.pop_front(packet_size_in_pages);
                    }
                    tile_id_start += output_pages_per_batch_head;
                    tiles_read = relay_start;
                    tiles_to_read = relay_end;
                    row_offset = (relay_start / slice_Wt) * stride_Wt;
                    pages_read_in_row = relay_start % slice_Wt;
                }
            }
        }

        signal_output_ready();

        slice_writes++;
    }

    if constexpr (kv_window) {
        // Return every credit the other direction's window still owes upstream, then leave this core's credit word as
        // the launch found it: subtract exactly what this launch used (the next launch's grants may already be in).
        while (kv_master.active && !kv_master.done()) {
            kv_master.service(noc_obj);
        }
        noc_semaphore_inc(
            get_noc_addr(kv_credits.addr), 0u - (kv_window_units | kvw_parts << kvw::kKvWindowDiamCreditShift));
    }

    // Drain in-flight writes BEFORE closing the EDM connections.
    noc_obj.async_atomic_barrier();
    noc_obj.async_write_barrier();

    fabric_connection.close();
}
