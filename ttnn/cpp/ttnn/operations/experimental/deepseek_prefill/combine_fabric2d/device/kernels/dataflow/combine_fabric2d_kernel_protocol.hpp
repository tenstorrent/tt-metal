// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Compile-time argument protocol between the program factory and the two kernels, plus the wire format of
// a ring slot's metadata tail. Both sides address every argument by name from the enums below, so no index
// is ever written down twice.
//
// The host fills a fixed-size array by name (`pack()`) and the kernels read by name (`ProducerCtArgs` /
// `ReaderCtArgs`), which is why this is not a byte stream: kernel-side values must be `constexpr` to drive
// `constexpr` branches and template parameters, and C++17 offers no constexpr way to reinterpret bytes as
// a struct.
//
// The reader's variable-length blocks follow its scalars, in this order:
//   [ReaderCtArg::Count ..)                      schedule, `schedule_len` words
//   [schedule end ..)                            assignments, 4 words each
//   [assignments end ..)                         TensorAccessorArgs, chained
// The producer has scalars only: every address it writes arrives per token in the slot's metadata tail.

namespace cmbf2d {

enum class ProducerCtArg : uint32_t {
    NumL1Slots,
    TokenSizeBytes,
    SlotTailBytes,
    PeerChipId,
    PeerMeshId,
    RingAddr,
    PktHdrRingAddr,
    PktHdrDrainAddr,
    DrainSinkAddr,
    Batch,
    FilledAddr,
    FreedAddr,
    FwdSemNocX,
    FwdSemNocY,
    FwdSemAddr,
    Count,
};

enum class ReaderCtArg : uint32_t {
    NumL1Slots,
    TokenSizeBytes,
    SlotTailBytes,
    Batch,
    RingAddr,
    FilledAddr,
    FreedAddr,
    DramInBaseAddr,
    DramOutBaseAddr,
    DramFwdBaseAddr,
    FwdChunksPerQuarter,
    FwdPagesPerChunk,
    MyQuarter,
    NumIncomingChunks,
    FwdSemAddr,
    NbrChipId,
    NumAssignments,
    ScheduleLen,
    DramMetaBaseAddr,
    DramCountsBaseAddr,
    DramRegionBaseAddr,
    DramExpertOffsetsBaseAddr,
    NumRoutedExperts,
    ExpertsPerChip,
    MyExpertBase,
    NumExpertsPerTok,
    DispatchGroupSize,
    LocalSplitCount,
    MyRow,
    ControlAddr,
    MetaPrefetchCap,
    Count,
};

// Words per assignment in the reader's assignment block: [dst_chip_id, dst_row, split_idx, split_count].
constexpr uint32_t ASSIGNMENT_WORDS = 4;
// Marks a schedule entry as "relay forwarding chunk k" rather than "own assignment k".
constexpr uint32_t SCHED_FWD = 0x80000000u;

// Ring slot = token + metadata tail. 64 rather than the 32 in use keeps the slot stride DRAM-aligned
// (14336 + 64 = 64 * 225), which lets a fabric write target a (token_size + 64)-byte forwarding page
// directly.
constexpr uint32_t SLOT_TAIL_BYTES = 64;

// Tail words, as uint64_t indices from (slot_base + token_size_bytes). The reader fills them, the producer
// consumes them. All uint64_t so the producer needs no sub-word loads.
constexpr uint32_t TAIL_FINAL_ADDR = 0;  // destination DRAM address on the FINAL destination chip
constexpr uint32_t TAIL_DST_CHIP = 1;    // final destination chip id; SENTINEL_DST_CHIP marks a sentinel
constexpr uint32_t TAIL_CMD = 2;
constexpr uint32_t TAIL_THIS_ADDR = 3;  // the address THIS hop writes to

constexpr uint64_t CMD_END = 0;  // end of stream; the slot carries no token
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;

// A sentinel carries no usable token; it marks the end of a forwarding chunk. UINT64_MAX can never collide
// with a real chip id.
constexpr uint64_t SENTINEL_DST_CHIP = UINT64_MAX;

#ifdef KERNEL_BUILD

#define CMBF2D_CT(kind, name) get_compile_time_arg_val(static_cast<uint32_t>(kind::name))

// A kernel is passed one role's args only, and a `static constexpr` initialiser is evaluated as soon as its
// enclosing class is complete, so reading the other role's indices would fail the CT-arg range assert. Each
// kernel therefore defines its role before including this header.
#ifdef CMBF2D_PRODUCER_KERNEL
struct ProducerCtArgs {
    static constexpr uint32_t num_l1_slots = CMBF2D_CT(ProducerCtArg, NumL1Slots);
    static constexpr uint32_t token_size_bytes = CMBF2D_CT(ProducerCtArg, TokenSizeBytes);
    static constexpr uint32_t slot_tail_bytes = CMBF2D_CT(ProducerCtArg, SlotTailBytes);
    static constexpr uint32_t peer_chip_id = CMBF2D_CT(ProducerCtArg, PeerChipId);
    static constexpr uint32_t peer_mesh_id = CMBF2D_CT(ProducerCtArg, PeerMeshId);
    static constexpr uint32_t ring_addr = CMBF2D_CT(ProducerCtArg, RingAddr);
    static constexpr uint32_t pkt_hdr_ring_addr = CMBF2D_CT(ProducerCtArg, PktHdrRingAddr);
    static constexpr uint32_t pkt_hdr_drain_addr = CMBF2D_CT(ProducerCtArg, PktHdrDrainAddr);
    static constexpr uint32_t drain_sink_addr = CMBF2D_CT(ProducerCtArg, DrainSinkAddr);
    static constexpr uint32_t batch = CMBF2D_CT(ProducerCtArg, Batch);
    static constexpr uint32_t filled_addr = CMBF2D_CT(ProducerCtArg, FilledAddr);
    static constexpr uint32_t freed_addr = CMBF2D_CT(ProducerCtArg, FreedAddr);
    static constexpr uint32_t fwd_sem_noc_x = CMBF2D_CT(ProducerCtArg, FwdSemNocX);
    static constexpr uint32_t fwd_sem_noc_y = CMBF2D_CT(ProducerCtArg, FwdSemNocY);
    static constexpr uint32_t fwd_sem_addr = CMBF2D_CT(ProducerCtArg, FwdSemAddr);

    static constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;
};
#endif

#ifdef CMBF2D_READER_KERNEL
struct ReaderCtArgs {
    static constexpr uint32_t num_l1_slots = CMBF2D_CT(ReaderCtArg, NumL1Slots);
    static constexpr uint32_t token_size_bytes = CMBF2D_CT(ReaderCtArg, TokenSizeBytes);
    static constexpr uint32_t slot_tail_bytes = CMBF2D_CT(ReaderCtArg, SlotTailBytes);
    static constexpr uint32_t batch = CMBF2D_CT(ReaderCtArg, Batch);
    static constexpr uint32_t ring_addr = CMBF2D_CT(ReaderCtArg, RingAddr);
    static constexpr uint32_t filled_addr = CMBF2D_CT(ReaderCtArg, FilledAddr);
    static constexpr uint32_t freed_addr = CMBF2D_CT(ReaderCtArg, FreedAddr);
    static constexpr uint32_t dram_in_base_addr = CMBF2D_CT(ReaderCtArg, DramInBaseAddr);
    static constexpr uint32_t dram_out_base_addr = CMBF2D_CT(ReaderCtArg, DramOutBaseAddr);
    static constexpr uint32_t dram_fwd_base_addr = CMBF2D_CT(ReaderCtArg, DramFwdBaseAddr);
    static constexpr uint32_t fwd_chunks_per_quarter = CMBF2D_CT(ReaderCtArg, FwdChunksPerQuarter);
    static constexpr uint32_t fwd_pages_per_chunk = CMBF2D_CT(ReaderCtArg, FwdPagesPerChunk);
    static constexpr uint32_t my_quarter = CMBF2D_CT(ReaderCtArg, MyQuarter);
    static constexpr uint32_t num_incoming_chunks = CMBF2D_CT(ReaderCtArg, NumIncomingChunks);
    static constexpr uint32_t fwd_sem_addr = CMBF2D_CT(ReaderCtArg, FwdSemAddr);
    static constexpr uint32_t nbr_chip_id = CMBF2D_CT(ReaderCtArg, NbrChipId);
    static constexpr uint32_t num_assignments = CMBF2D_CT(ReaderCtArg, NumAssignments);
    static constexpr uint32_t schedule_len = CMBF2D_CT(ReaderCtArg, ScheduleLen);
    static constexpr uint32_t dram_meta_base_addr = CMBF2D_CT(ReaderCtArg, DramMetaBaseAddr);
    static constexpr uint32_t dram_counts_base_addr = CMBF2D_CT(ReaderCtArg, DramCountsBaseAddr);
    static constexpr uint32_t dram_region_base_addr = CMBF2D_CT(ReaderCtArg, DramRegionBaseAddr);
    static constexpr uint32_t dram_expert_offsets_base_addr = CMBF2D_CT(ReaderCtArg, DramExpertOffsetsBaseAddr);
    static constexpr uint32_t num_routed_experts = CMBF2D_CT(ReaderCtArg, NumRoutedExperts);
    static constexpr uint32_t experts_per_chip = CMBF2D_CT(ReaderCtArg, ExpertsPerChip);
    static constexpr uint32_t my_expert_base = CMBF2D_CT(ReaderCtArg, MyExpertBase);
    static constexpr uint32_t num_experts_per_tok = CMBF2D_CT(ReaderCtArg, NumExpertsPerTok);
    static constexpr uint32_t dispatch_group_size = CMBF2D_CT(ReaderCtArg, DispatchGroupSize);
    static constexpr uint32_t local_split_count = CMBF2D_CT(ReaderCtArg, LocalSplitCount);
    static constexpr uint32_t my_row = CMBF2D_CT(ReaderCtArg, MyRow);
    static constexpr uint32_t control_addr = CMBF2D_CT(ReaderCtArg, ControlAddr);
    static constexpr uint32_t meta_prefetch_cap = CMBF2D_CT(ReaderCtArg, MetaPrefetchCap);

    static constexpr uint32_t schedule_base = static_cast<uint32_t>(ReaderCtArg::Count);
    static constexpr uint32_t assignment_base = schedule_base + schedule_len;
    static constexpr uint32_t accessor_base = assignment_base + ASSIGNMENT_WORDS * num_assignments;
    static constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;
};
#endif

#undef CMBF2D_CT

#else  // host

#include <array>
#include <vector>

// Host mirror: fill by name, then `pack()` into the kernel's compile-time arg vector. Fields are
// deliberately unordered here — position comes from the enum, not from declaration order.
template <typename Arg, Arg CountValue>
class CtArgPacker {
public:
    uint32_t& operator[](Arg a) { return words_[static_cast<uint32_t>(a)]; }

    void append_to(std::vector<uint32_t>& ct_args) const {
        ct_args.insert(ct_args.end(), words_.begin(), words_.end());
    }

private:
    std::array<uint32_t, static_cast<uint32_t>(CountValue)> words_{};
};

using ProducerCtArgPacker = CtArgPacker<ProducerCtArg, ProducerCtArg::Count>;
using ReaderCtArgPacker = CtArgPacker<ReaderCtArg, ReaderCtArg::Count>;

#endif  // KERNEL_BUILD

}  // namespace cmbf2d
