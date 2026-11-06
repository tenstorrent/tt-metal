// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// push_rows_block pieces shared by the two pushing RISCs (reader, writer).
//
// A token core stages a range of stage_tiles hidden tiles of its tokens' rows, untilized: a token's staged row is
// stage_tiles*64 contiguous bytes. The range is its own slice c = [c*ht, (c+1)*ht), or (whole-piece push) the
// output pieces it pushes, gathered from its group peers. Output row r lives in output tile row r/32, which is cut
// into k pieces of w tiles; shard s = (r/32)*k + p sits on core s % num_cores in its local block s / num_cores. Each
// pushed part of a row (host _push_args) is one NoC write into the owner's cb_gathered_rows block, at the row's
// row-major offset inside the 32 x w-tile block that the owner later tilizes in place.

#pragma once

#include <cstdint>
#include <string_view>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ckernel.h"
#include "nomic_moe_route_dispatch_w_layout.hpp"

namespace route_dispatch {

constexpr uint32_t kTile = 32;

// Owner-side landing: the row pushes are the only posted writes that land on a worker core during the op (every other
// write, the dispatcher's included, is non-posted), so the sum of both NIUs' posted-write slave counters, minus its
// value at kernel start, counts the push writes landed here. NIU_SLV_POSTED_WR_REQ_RECEIVED moves only once a request's
// data is in L1 (measured, perf_experiments/r32_owner_land/race_probe; REQ_STARTED does not). A core's owned blocks
// receive exactly one push write per row (slot rows and pads cover [0, rows) once per piece; a piece split by row
// parity is still one write a row), so all nb blocks have landed at 32 * nb.
// Raw NIU register reads: no kernel_lib helper exposes the slave counters. Replacing the coordinator barrier (Mcast
// helper go + landing atomics) measured 8x512 25.6 -> 24.5 us (Perf R3.2, r32_owner_land).
inline uint32_t owner_land_count() {
    return NOC_STATUS_READ_REG(0, NIU_SLV_POSTED_WR_REQ_RECEIVED) +
           NOC_STATUS_READ_REG(1, NIU_SLV_POSTED_WR_REQ_RECEIVED);
}
constexpr uint32_t kTileRowBytes = 64;  // one row of one bf16 tile column (32 elements)
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kNumExperts = 8;
// Slot-rows page: slots k-major (2 x rps*32 words; packed expert | rank << 4 after select_block, the output row
// after push_rows_block), then the region header.
constexpr uint32_t kHeaderStarts = 0;
constexpr uint32_t kHeaderTotals = kNumExperts;
constexpr uint32_t kHeaderEnd = 2 * kNumExperts;
constexpr uint32_t kHeaderStaging = kHeaderEnd + 1;  // L1 address of the untilized shard (cb_x_rows)
// Published by the reader before the table lands (Perf R3.3: a selecting writer scans the table itself).
// kHeaderTableFlag: read by a coordinator token core's writer only (the coordinator sends the table, nothing lands).
constexpr uint32_t kHeaderReaderCounts = kHeaderEnd + 2;    // 2 words: the reader's byte-lane counts
constexpr uint32_t kHeaderTable = kHeaderReaderCounts + 2;  // L1 address of the count table
constexpr uint32_t kHeaderTableFlag = kHeaderTable + 1;     // 1 once the table has landed
// Table scan halves (scan_rows_packed): packed sums[4], packed prefixes[4], then a done flag, one block a RISC.
constexpr uint32_t kScanFlag = 8;
constexpr uint32_t kHeaderScanReader = kHeaderTableFlag + 1;
constexpr uint32_t kHeaderScanWriter = kHeaderScanReader + kScanFlag + 1;
// Set by the reader once the untilized shard is in cb_x_rows (it waits for it after the assign, not before the table)
constexpr uint32_t kHeaderStagingReady = kHeaderScanWriter + kScanFlag + 1;
constexpr uint32_t kHeaderWords = kHeaderStagingReady + 1;
constexpr uint32_t kRankShift = 4;
constexpr uint32_t kExpertMask = (1u << kRankShift) - 1;
// Reader -> writer select handshake page (cb_select_go): the slot page address.
constexpr uint32_t kGoSlotPage = 0;
// Writer -> reader counts page (cb_writer_counts): 2 count words, then the drain word.
constexpr uint32_t kHandshakeWords = 3;
// fp32 tile: 1024 words; column c of row r sits in face 0 (r < 16) or face 2 (r >= 16).
constexpr uint32_t kTileWords = 1024;
constexpr uint32_t kFaceWords = 256;
inline uint32_t face_word(uint32_t row) { return row < 16 ? row * 16 : 2 * kFaceWords + (row - 16) * 16; }

// A RISC-V store can retire before L1 has processed it, and the NoC reads L1 as a separate client. L1 serves this
// core's requests in order, so a blocking load of a word just stored returns only once every earlier store landed.
inline void drain_stores(volatile tt_l1_ptr uint32_t* scratch_word) {
    *scratch_word = 0;
    (void)ckernel::load_blocking(scratch_word);
}

inline uint32_t coord_x(uint32_t packed) { return packed >> 16; }
inline uint32_t coord_y(uint32_t packed) { return packed & 0xFFFF; }

constexpr uint32_t round_up_to_tile(uint32_t rows) { return (rows + kTile - 1) / kTile * kTile; }

// Count-table scan: packed u16-lane sums (8 experts in 4 words; a total is at most 2T < 65536, so no lane carries) of
// rows [lo, hi) into sums[0..3] and of rows [lo, clamp(group_of, lo, hi)) into sums[4..7]. The scan is L1 load-latency
// bound (~4.5 cycles a load), so a selecting writer scans the upper half of the rows in parallel with the reader
// (Perf R3.3, perf_experiments/r33_table_assign).
template <typename Words>
inline void scan_rows_packed(
    const tt_l1_ptr uint32_t* rows_words, uint32_t lo, uint32_t hi, uint32_t group_of, Words sums) {
    asm volatile("" ::: "memory");
    const uint32_t mid = group_of < lo ? lo : (group_of > hi ? hi : group_of);
    uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    uint32_t g = lo;
#pragma GCC unroll 2
    for (; g < mid; ++g) {
        const tt_l1_ptr uint32_t* row = rows_words + 4 * g;
        b0 += row[0];
        b1 += row[1];
        b2 += row[2];
        b3 += row[3];
    }
    uint32_t t0 = b0, t1 = b1, t2 = b2, t3 = b3;
#pragma GCC unroll 2
    for (; g < hi; ++g) {
        const tt_l1_ptr uint32_t* row = rows_words + 4 * g;
        t0 += row[0];
        t1 += row[1];
        t2 += row[2];
        t3 += row[3];
    }
    sums[0] = t0, sums[1] = t1, sums[2] = t2, sums[3] = t3;
    sums[4] = b0, sums[5] = b1, sums[6] = b2, sums[7] = b3;
}

// Totals / before (prefix over the lower groups) from one or two scan_rows_packed blocks (b may be null).
template <typename WordsA, typename WordsB>
inline void combine_scans(WordsA a, WordsB b, uint32_t (&totals)[kNumExperts], uint32_t (&before)[kNumExperts]) {
    for (uint32_t word = 0; word < 4; ++word) {
        const uint32_t t = a[word] + (b ? b[word] : 0);
        const uint32_t p = a[4 + word] + (b ? b[4 + word] : 0);
        totals[2 * word] = t & 0xFFFF;
        totals[2 * word + 1] = t >> 16;
        before[2 * word] = p & 0xFFFF;
        before[2 * word + 1] = p >> 16;
    }
}

inline void wait_flag(const volatile tt_l1_ptr uint32_t* flag) {
    while (*flag == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
}

// The count table as ONE posted multicast to the grid (source excluded): num_groups count rows, then a valid row whose
// first word the coordinator sets to 1 (every receiver clears it at kernel start). A receiver waits for its NIU0
// posted-write slave counter (landing accounting) AND the valid word: a peer that already has the table may push a row
// here before the multicast lands, and that push alone moves the counter. The valid word is the last word of the one
// packet, so the rows before it have landed once it reads 1.
// No path reservation: a reserved broadcast waits until every router of the tree has granted it, ~400 cycles on the
// 11x10 grid. The NoC leaves deadlock freedom of unreserved broadcasts to software: NOC0 carries two (W channel 0 and
// this table), both from the coordinator's reader on one command buffer and VC, so they leave in order and never hold
// parts of each other's tree. No kernel_lib helper multicasts posted or unreserved. 8x512 21.6 -> 21.35 us
// (perf_experiments/r41_table_mcast).
inline uint64_t table_mcast_dst(uint32_t addr, uint32_t coords_arg, uint32_t num_cores) {
    const uint32_t first = get_common_arg_val<uint32_t>(coords_arg);
    const uint32_t last = get_common_arg_val<uint32_t>(coords_arg + num_cores - 1);
    return get_noc_multicast_addr(coord_x(first), coord_y(first), coord_x(last), coord_y(last), addr, noc_index);
}

inline void send_table(uint64_t dst, uint32_t addr, uint32_t bytes, uint32_t num_dests) {
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_index, dst, addr, bytes);
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
    }
    ncrisc_noc_fast_write<noc_mode>(
        noc_index, write_cmd_buf, addr, dst, bytes, NOC_MULTICAST_WRITE_VC, true, false, num_dests, false, true);
}

inline uint32_t table_land_base() { return NOC_STATUS_READ_REG(0, NIU_SLV_POSTED_WR_REQ_RECEIVED); }

inline volatile tt_l1_ptr uint32_t* table_valid_word(uint32_t table_addr, uint32_t num_groups, uint32_t row_bytes) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_addr + num_groups * row_bytes);
}

inline void wait_table(uint32_t base, const volatile tt_l1_ptr uint32_t* valid) {
    while (NOC_STATUS_READ_REG(0, NIU_SLV_POSTED_WR_REQ_RECEIVED) - base < 1) {
    }
    wait_flag(valid);
}

// Region header from the scanned table: starts / totals / end, and the first row per expert of a RISC whose tokens
// follow `prior` (byte-lane counts of the earlier tokens of the core) in the group's token order.
inline void assign_regions(
    volatile tt_l1_ptr uint32_t* header,
    const uint32_t (&totals)[kNumExperts],
    const uint32_t (&before)[kNumExperts],
    uint64_t prior,
    uint32_t (&base)[kNumExperts]) {
    uint32_t region_start = 0;
    for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
        base[expert] = region_start + before[expert] + (static_cast<uint32_t>(prior >> (expert * 8)) & 0xFF);
        header[kHeaderStarts + expert] = region_start;
        header[kHeaderTotals + expert] = totals[expert];
        region_start += round_up_to_tile(totals[expert]);
    }
    header[kHeaderEnd] = region_start;
}

// W slices (G of ht fp32 tiles), faces 0 / 2 only in the compact layout of nomic_moe_route_dispatch_w_layout.hpp.
// Slices go out on RISC c % slice_nocs (0: reader / NOC0, 1: writer / NOC1; host W_SLICE_NOCS); every slice of a NoC
// from the sender of its first slice (channel c % slice_nocs, core w_sender_top - channel), then the channel's flag
// (semaphore sem_w_flag0 + channel, every core). Every receiver waits on the RISC whose NoC carries its slice.
//
// Raw multicasts, not the Mcast helper: unreserved (a reserved broadcast waits for every router of the tree to grant
// it, ~400 cycles on the 11x10 grid) and the flag rides the same command buffer and VC right behind the data, so it
// lands after the data on each receiver with no barrier between them. Deadlock freedom of unreserved broadcasts is
// left to software: one broadcast source a NoC (NOC1: W channel 1; NOC0: the coordinator's reader, which sends W
// channel 0 and later the count table on one command buffer and VC). No kernel_lib helper multicasts unreserved.
// The data's burst chunks are linked and the flag closes the sequence: unlinked chunks each re-arbitrate the tree
// (G = 1, 48 KB in 16 KB bursts: W landed ~950 cycles later, l1_interleaved 2x512 / 4x256 +4.6% / +5.3%).
// Perf R4.2, perf_experiments/r42_w_arrival.
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kMaxSlices = 3;
constexpr uint32_t kReaderRisc = 0;
constexpr uint32_t kWriterRisc = 1;
constexpr uint32_t kPartialFaceOffsets[] = {0, 2 * kWFaceBytes};  // faces 0 and 2 (columns 0..15) of an fp32 tile
constexpr uint32_t kWEntryBytes = 2 * kWFaceBytes;                // faces 0 / 2 of one K tile

// The whole grid (coords_arg: its cores' packed NoC coordinates in core order), in `noc`'s routing order.
inline uint64_t grid_mcast_dst(uint8_t noc, uint32_t coords_arg, uint32_t num_cores, uint32_t addr) {
    const uint32_t first = get_common_arg_val<uint32_t>(coords_arg);
    const uint32_t last = get_common_arg_val<uint32_t>(coords_arg + num_cores - 1);
    return noc == 0 ? get_noc_multicast_addr(coord_x(first), coord_y(first), coord_x(last), coord_y(last), addr, noc)
                    : get_noc_multicast_addr(coord_x(last), coord_y(last), coord_x(first), coord_y(first), addr, noc);
}

// Non-posted (the posted-write slave counters count only pushes / count rows), unreserved, source excluded; every
// burst chunk carries `linked`.
inline void w_mcast(uint8_t noc, uint64_t dst, uint32_t src, uint32_t bytes, uint32_t num_dests, bool linked) {
    while (bytes > 0) {
        const uint32_t chunk = bytes > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : bytes;
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst, src, chunk);
        while (!noc_cmd_buf_ready(noc, write_cmd_buf)) {
        }
        ncrisc_noc_fast_write<noc_mode>(
            noc, write_cmd_buf, src, dst, chunk, NOC_MULTICAST_WRITE_VC, true, linked, num_dests, false, false);
        src += chunk;
        dst += chunk;
        bytes -= chunk;
    }
}

#ifdef KERNEL_COMPILE_TIME_ARG_MAP
// receive_weight_slice on the RISC whose NoC carries slice `member`: it is cb_router_weight's producer on this core
// (one producer a core; the other RISC never touches the CB). A core that sent the slice itself does not wait.
template <uint32_t G, uint32_t slice_nocs, uint32_t risc>
FORCE_INLINE void receive_weight_slice(
    CircularBuffer& router_weight, uint32_t weight_tiles, uint32_t member, uint32_t sem_w_flag0, bool sent_here) {
    const uint32_t channel = member % slice_nocs;
    if (channel != risc) {
        return;
    }
    router_weight.reserve_back(weight_tiles);
    if (!sent_here) {
        Semaphore<>(sem_w_flag0 + channel).wait(1);
    }
    router_weight.push_back(weight_tiles);
}

// The sender core of channel `channel` (core w_sender_top - channel): faces 0 / 2 of every slice the channel carries,
// read from DRAM into their compact slots (consecutive), then multicast. Both RISCs of the sender core read, each over
// its own NoC (one RISC issuing every 1 KB read was the bound, perf_experiments/r32_w_early): the RISC off the
// channel's NoC reads the first half of the K tile pairs and raises sem_w_local; the RISC on it reads the rest, then
// multicasts the channel and its flag. Both RISCs of a sender core must call this. The multicaster leaves its acks to
// the kernel's final write barrier unless `ack_now`.
template <uint32_t G, uint32_t ht, uint32_t slice_nocs, uint32_t risc, typename Accessor>
FORCE_INLINE void deliver_weight_slice(
    const Noc& noc,
    const Accessor& weight,
    uint32_t weight_base,
    uint32_t channel,
    uint32_t sem_w_local,
    uint32_t sem_w_flag0,
    uint32_t coords_arg,
    uint32_t num_cores,
    bool ack_now) {
    static_assert(slice_nocs == 1 || slice_nocs == 2, "W slices go out on the reader's NoC, or alternate over both");
    static_assert(ht % 2 == 0, "W is read in pairs of K tiles");
    if (channel >= slice_nocs) {
        return;
    }
    const uint8_t noc_id = noc.get_noc_id();
    const bool mcaster = channel == risc;
    const uint32_t entries = w_channel_slices(channel, G, slice_nocs) * ht;  // (slot, K tile), slot-major
    const uint32_t split = entries / 4 * 2;  // the first half of the pairs: the other RISC's share
    const uint32_t channel_addr = weight_base + w_slot(channel, G, slice_nocs) * ht * kWEntryBytes;
    for (uint32_t entry = mcaster ? split : 0; entry < (mcaster ? entries : split); ++entry) {
        const uint32_t slot = entry / ht;
        const uint32_t tile = entry % ht;
        const uint32_t dst = channel_addr + (slot * 2 * ht + w_face_tile(0, ht, tile)) * kWFaceBytes;
        const uint64_t src = weight.get_noc_addr((channel + slot * slice_nocs) * ht + tile, 0, noc_id);
        noc_async_read(src, dst, kWFaceBytes, noc_id);
        noc_async_read(src + 2 * kWFaceBytes, dst + 2 * kWFaceBytes, kWFaceBytes, noc_id);
    }
    Semaphore<> other_half(sem_w_local);
    if (!mcaster) {
        noc.async_read_barrier();
        other_half.set(1);
        return;
    }
    const uint64_t dst = grid_mcast_dst(noc_id, coords_arg, num_cores, 0);
    other_half.wait(1);
    noc.async_read_barrier();
    w_mcast(noc_id, dst | channel_addr, channel_addr, entries * kWEntryBytes, num_cores - 1, true);
    // The flag's source: this core's own cell (the multicast excludes its source). A blocking load of the word just
    // stored returns once the store reached L1, before the NoC reads it.
    const uint32_t flag_addr = get_semaphore(sem_w_flag0 + channel);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    *flag = 1;
    (void)ckernel::load_blocking(flag);
    w_mcast(noc_id, dst | flag_addr, flag_addr, 4, num_cores - 1, false);
    if (ack_now) {
        noc.async_write_barrier();
    }
}
#endif

// exchange_partials_block, one partial tile (Perf R3.3): to every other member d of the group the faces of tile `tile`
// of this member's region whose halves d computes (w_layout.hpp half_at_rank), then the landed-tile count (the
// constant at flag_src) to this member's counter flag_dst on d. The count goes out on the same NoC, command buffer and
// VC as the faces, so it lands after them (no barrier, no atomic). Sent by the writer only, tile by tile as compute
// packs them: sending each tile as soon as it is packed moved 8x512 push start 14.6k -> 13.6k cycles
// (perf_experiments/r33_partials_window).
template <uint32_t G>
FORCE_INLINE void send_tile_partials(
    const Noc& noc,
    uint32_t region,
    uint32_t coords_arg,
    uint32_t member,
    HalfSplit split,
    uint32_t tile,
    uint32_t flag_src,
    uint32_t flag_dst) {
    for (uint32_t peer = 0; peer < G; ++peer) {
        if (peer == member) {
            continue;
        }
        const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg + peer);
        for (uint32_t pos = peer; pos < split.hn; pos += G) {
            const uint32_t half = half_at_rank(pos, split);
            if (half / 2 != tile) {
                continue;
            }
            const uint32_t addr = region + tile * kFp32TileBytes + kPartialFaceOffsets[half % 2];
            noc.async_write(
                CoreLocalMem<uint32_t>(addr),
                UnicastEndpoint{},
                kWFaceBytes,
                {},
                {.noc_x = coord_x(coord), .noc_y = coord_y(coord), .addr = addr});
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(flag_src),
            UnicastEndpoint{},
            4,
            {},
            {.noc_x = coord_x(coord), .noc_y = coord_y(coord), .addr = flag_dst});
    }
}

// Streamed all-gather of the code faces. Code flags page (cb_code_flags, same address on every
// core): word 4 * m = passes of member m landed here, word 4 * (kMaxSlices + j) = the constant j + 1 (the source of
// the flag write after pass j). A pass's face write and its flag write go out on the same NoC, command buffer and VC
// to the same core, so they land in order (no barrier). No kernel_lib helper streams an intra-group all-gather (the
// mcast helper is one-to-many and barriers per send), so these are plain Noc writes (Perf R3.1, r3_softmax_split).
constexpr uint32_t kFlagStrideWords = 4;  // 16 B: L1-to-L1 NoC writes keep the low address bits

constexpr uint32_t kSemPartials[kMaxSlices] = {
    get_named_compile_time_arg_val("sem_partials0"),
    get_named_compile_time_arg_val("sem_partials1"),
    get_named_compile_time_arg_val("sem_partials2")};

// The constants 1..count after the counters: NoC write sources for landed-pass and landed-tile counts. Both DM RISCs
// write them so neither waits for the other; the blocking load orders the stores before any NoC read of them.
inline void write_flag_constants(volatile tt_l1_ptr uint32_t* flags, uint32_t count) {
    for (uint32_t pass = 0; pass < count; ++pass) {
        flags[(kMaxSlices + pass) * kFlagStrideWords] = pass + 1;
    }
    (void)ckernel::load_blocking(&flags[(kMaxSlices + count - 1) * kFlagStrideWords]);
}

inline void init_code_flags(volatile tt_l1_ptr uint32_t* flags, uint32_t max_passes) {
    for (uint32_t member = 0; member < kMaxSlices; ++member) {
        flags[member * kFlagStrideWords] = 0;
    }
    write_flag_constants(flags, max_passes);
}

// One DM RISC's side of the code all-gather: it sends each pass of this member (cb_code_stage page j, packed by
// compute) to members [first, last) as soon as compute pushed it. pump() never blocks.
template <uint32_t G>
class CodePump {
public:
    CodePump(
        uint32_t stage_cb,
        uint32_t scores_base,
        uint32_t flags_base,
        uint32_t member,
        uint32_t coords_arg,
        uint32_t first,
        uint32_t last,
        HalfSplit split) :
        stage_(stage_cb),
        scores_base_(scores_base),
        flags_base_(flags_base),
        member_(member),
        coords_arg_(coords_arg),
        first_(first),
        last_(last),
        split_(split),
        passes_(first < last ? member_passes(member, G, split) : 0) {}

    void pump(const Noc& noc) {
        while (sent_ < passes_ && stage_.pages_available_at_front(sent_ + 1)) {
            send(noc, sent_++);
        }
    }

    void finish(const Noc& noc) {
        while (sent_ < passes_) {
            pump(noc);
        }
    }

private:
    void send(const Noc& noc, uint32_t pass) {
        const uint32_t half = half_at_rank(member_ + pass * G, split_);
        const uint32_t face = kPartialFaceOffsets[half % 2];
        const uint32_t src = stage_.get_read_ptr() + pass * kFp32TileBytes + face;
        const uint32_t dst = scores_base_ + (half / 2) * kFp32TileBytes + face;
        const uint32_t flag_src = flags_base_ + (kMaxSlices + pass) * kFlagStrideWords * 4;
        const uint32_t flag_dst = flags_base_ + member_ * kFlagStrideWords * 4;
        for (uint32_t peer = first_; peer < last_; ++peer) {
            const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg_ + peer);
            noc.async_write(
                CoreLocalMem<uint32_t>(src),
                UnicastEndpoint{},
                kWFaceBytes,
                {},
                {.noc_x = coord_x(coord), .noc_y = coord_y(coord), .addr = dst});
            noc.async_write(
                CoreLocalMem<uint32_t>(flag_src),
                UnicastEndpoint{},
                4,
                {},
                {.noc_x = coord_x(coord), .noc_y = coord_y(coord), .addr = flag_dst});
        }
    }

    CircularBuffer stage_;
    uint32_t scores_base_;
    uint32_t flags_base_;
    uint32_t member_;
    uint32_t coords_arg_;
    uint32_t first_;
    uint32_t last_;
    HalfSplit split_;
    uint32_t passes_;
    uint32_t sent_ = 0;
};

// Blocks until half `half`'s code face has landed here, pumping this RISC's own sends meanwhile.
template <uint32_t G>
inline void wait_code_half(
    const Noc& noc, CodePump<G>& pump, const volatile tt_l1_ptr uint32_t* flags, uint32_t half, HalfSplit split) {
    const uint32_t pos = half_rank(half, split);
    const uint32_t owner = pos % G;
    const uint32_t need = pos / G + 1;
    pump.pump(noc);
    while (flags[owner * kFlagStrideWords] < need) {
        invalidate_l1_cache();
        pump.pump(noc);
    }
    invalidate_l1_cache();  // the code rows read next landed over the NoC
}

// Push RT args (host _push_args) and a piece's row filter (host ROWS_ALL / ROWS_EVEN / ROWS_ODD).
constexpr uint32_t kPushArgStageLo = 0;
constexpr uint32_t kPushArgPieces = 1;
constexpr uint32_t kPushArgFirstPiece = 2;
constexpr uint32_t kPushPieceWords = 5;
constexpr uint32_t kRowsAll = 0;

// Posted writes: the owner proves landing (owner_land_count), so the source needs no ack. At most bytes_in_flight of
// pushes not yet sent by the NIU: past a bytes threshold the NoC tips into congestion collapse (measured on
// nomic_moe_dispatch; uncapped posted pushes end 1-2k cycles later, r32_owner_land). The cap is counted in writes of
// this RISC's average piece, so a member with more, smaller writes keeps as many bytes in flight as the others.
// Credits: the sent counter is read only when the last reading's credits are spent.
// The per-write software paces the push (Perf R4.1: zero credit stalls), so it is kept minimal: the destination
// (owner coordinate register value, block address, VC command buffer) of each piece is computed once per output tile
// row and cached per walk stream (one entry per expert: a RISC's rows of an expert ascend through its region, so a
// stream changes tile row at most once every 32 rows), the piece loop is unrolled over the host's max pieces a member
// pushes, and the row parity filter compiles out unless some piece is split by parity.
// The diet alone delivers unfairly (faster issue, slower last landing). Each write's static VC is keyed on the owner's
// index relative to this core, (owner - self) & 1 -> VC 1 / VC 3, which spreads one source's writes over two VCs by
// destination and keeps the diet a win on every shape measured (Perf R4.3, r43_push_vc_rel: 8x512 -3.7..-5.3%; a
// global key such as owner & 1 or owner & 3 regressed other shapes, r42_push_vc). Each VC has its own command buffer,
// programmed once in start(), so a write adds no register store for its VC.
template <uint32_t w, uint32_t k, uint32_t max_pieces, bool row_parity, uint32_t num_cores, uint32_t bytes_in_flight>
class RowPusher {
    static_assert(max_pieces >= 1 && max_pieces <= k, "a member pushes at most k pieces a row");
    static constexpr uint32_t kVcCmdBufs[2] = {write_cmd_buf, write_reg_cmd_buf};
    static constexpr uint32_t kPushVcs[2] = {1, 3};
    static constexpr uint32_t kRowBytes = w * kTileRowBytes;

public:
    static constexpr uint32_t kPadStream = kNumExperts;  // pad rows; streams 0..7 are the experts' slot rows
    static constexpr uint32_t kStreams = kNumExperts + 1;

    // A stream's current output tile row: per piece, the owner's NOC_RET_ADDR_COORDINATE value and the address of
    // the row-0 slice in the owner's block (shard s = tile_row * k + piece on core s % num_cores, block s / num_cores).
    struct Stream {
        uint32_t tile_row;
        uint32_t coord[max_pieces];
        uint32_t dst[max_pieces];
        uint32_t cmd_buf[max_pieces];  // the command buffer of the write's VC
    };

    // pieces_arg: the RISC's push RT args (host _push_args): first staged tile, piece count, then k pieces of
    // (piece, src offset in the staged row, dst offset in the piece, bytes, rows).
    // streams: kStreams entries, owned by the caller (an array member would keep the whole pusher in memory, the
    // per-write scalars included, since the streams are indexed at run time).
    RowPusher(uint32_t pieces_arg, uint32_t gathered_base, uint32_t coords_arg, Stream* streams, uint32_t self) :
        streams_(streams), gathered_base_(gathered_base), coords_arg_(coords_arg), self_(self) {
        num_pieces_ = get_arg_val<uint32_t>(pieces_arg + kPushArgPieces);
        uint32_t bytes = 0;
#pragma GCC unroll 4
        for (uint32_t i = 0; i < max_pieces; ++i) {
            if (i >= num_pieces_) {
                break;
            }
            const uint32_t arg = pieces_arg + kPushArgFirstPiece + i * kPushPieceWords;
            piece_[i] = get_arg_val<uint32_t>(arg);
            src_[i] = get_arg_val<uint32_t>(arg + 1);
            dst_[i] = get_arg_val<uint32_t>(arg + 2);
            bytes_[i] = get_arg_val<uint32_t>(arg + 3);
            rows_[i] = get_arg_val<uint32_t>(arg + 4);
            bytes += bytes_[i];
        }
        const uint32_t writes = bytes > 0 ? bytes_in_flight * num_pieces_ / bytes : 0;
        writes_in_flight_ = writes > 0 ? writes : 1;
        for (uint32_t stream = 0; stream < kStreams; ++stream) {
            streams_[stream].tile_row = ~0u;
        }
    }

    bool idle() const { return num_pieces_ == 0; }

    // Programs the write command buffer's fixed fields; call before a run of push() with no other write between.
    void start() {
        static_assert(kVcCmdBufs[0] != kVcCmdBufs[1], "one command buffer a push VC");
        for (uint32_t vc = 0; vc < 2; ++vc) {
            ncrisc_noc_write_set_state<true, false>(
                noc_index, kVcCmdBufs[vc], get_noc_addr(my_x[noc_index], my_y[noc_index], 0), 0, kPushVcs[vc]);
        }
        // Posted writes issued earlier on this NIU but not yet sent would make the in-flight count wrap.
        noc_async_posted_writes_flushed(noc_index);
        sent_base_ = NOC_STATUS_READ_REG(noc_index, NIU_MST_POSTED_WR_REQ_SENT);
        issued_ = 0;
    }

    // One row slice (src_row: its staging address) to output row `row` of walk stream `stream`.
    __attribute__((always_inline)) inline void push(uint32_t src_row, uint32_t row, uint32_t stream) {
#ifdef ABLATE_PUSH_PIECES  // /perf-measure: the per-piece address math and writes stubbed (the slot walk kept)
        asm volatile("" ::"r"(src_row), "r"(row), "r"(stream));
        return;
#endif
        Stream& s = streams_[stream];
        const uint32_t tile_row = row / kTile;
        if (s.tile_row != tile_row) [[unlikely]] {
            refill(s, tile_row);
        }
        const uint32_t row_offset = (row % kTile) * kRowBytes;
#pragma GCC unroll 4
        for (uint32_t i = 0; i < max_pieces; ++i) {
            if (i >= num_pieces_) {
                break;
            }
            if constexpr (row_parity) {
                if (rows_[i] != kRowsAll && (row & 1) + 1 != rows_[i]) {
                    continue;
                }
            }
            write(s.cmd_buf[i], s.coord[i], src_row + src_[i], s.dst[i] + row_offset, bytes_[i]);
        }
    }

    // Accounts the issued writes in the counter a posted-write flush waits on.
    void finish() {
        noc_posted_writes_num_issued[noc_index] += issued_;
#ifdef ABLATE_MEASURE_STALLS
        DeviceTimestampedData("push_writes", issued_);
        DeviceTimestampedData("push_stall_credit", stall_credit_);
        DeviceTimestampedData("push_stalls_credit", stalls_credit_);
        DeviceTimestampedData("push_stall_cmdbuf", stall_cmd_);
#endif
    }

private:
    __attribute__((always_inline)) inline void refill(Stream& s, uint32_t tile_row) {
        s.tile_row = tile_row;
#pragma GCC unroll 4
        for (uint32_t i = 0; i < max_pieces; ++i) {
            if (i >= num_pieces_) {
                break;
            }
            const uint32_t shard = tile_row * k + piece_[i];
            const uint32_t block = shard / num_cores;
            const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg_ + (shard - block * num_cores));
            s.coord[i] =
                static_cast<uint32_t>(get_noc_addr(coord_x(coord), coord_y(coord), 0) >> NOC_ADDR_COORD_SHIFT) &
                NOC_COORDINATE_MASK;
            s.dst[i] = gathered_base_ + block * w * kBf16TileBytes + dst_[i];
            const uint32_t owner = shard - block * num_cores;
            s.cmd_buf[i] = kVcCmdBufs[(owner + num_cores - self_) & 1];
        }
    }

    __attribute__((always_inline)) inline void write(
        uint32_t cmd_buf, uint32_t coord, uint32_t src, uint32_t dst, uint32_t bytes) {
        if (credits_ == 0) {
            uint32_t outstanding;
#ifdef ABLATE_MEASURE_STALLS
            outstanding = sent_base_ + issued_ - NOC_STATUS_READ_REG(noc_index, NIU_MST_POSTED_WR_REQ_SENT);
            if (outstanding >= writes_in_flight_) {
                const uint32_t t0 = clk();
                ++stalls_credit_;
#endif
                while (
                    (outstanding = sent_base_ + issued_ - NOC_STATUS_READ_REG(noc_index, NIU_MST_POSTED_WR_REQ_SENT)) >=
                    writes_in_flight_) {
                }
#ifdef ABLATE_MEASURE_STALLS
                stall_credit_ += clk() - t0;
            }
#endif
            credits_ = writes_in_flight_ - outstanding;
        }
        --credits_;
#ifdef ABLATE_PUSH_WRITES
        asm volatile("" ::"r"(cmd_buf), "r"(dst), "r"(coord), "r"(src), "r"(bytes));
        return;
#endif
        ++issued_;
        // Stateful one-packet write: only the target coordinate, addresses and length change per write.
#ifdef ABLATE_MEASURE_STALLS
        if (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
            const uint32_t t0 = clk();
            while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
            }
            stall_cmd_ += clk() - t0;
        }
#endif
        while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
        ncrisc_noc_write_with_state<DM_DEDICATED_NOC, true, false, false, true>(noc_index, cmd_buf, src, dst, bytes);
    }

#ifdef ABLATE_MEASURE_STALLS
    static uint32_t clk() { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
    uint32_t stall_credit_ = 0, stalls_credit_ = 0, stall_cmd_ = 0;
#endif
    Stream* streams_;
    uint32_t piece_[max_pieces]{};
    uint32_t src_[max_pieces]{};
    uint32_t dst_[max_pieces]{};
    uint32_t bytes_[max_pieces]{};
    uint32_t rows_[max_pieces]{};
    uint32_t num_pieces_ = 0;
    uint32_t writes_in_flight_ = 1;
    uint32_t credits_ = 0;
    uint32_t sent_base_ = 0;
    uint32_t issued_ = 0;
    uint32_t gathered_base_;
    uint32_t coords_arg_;
    uint32_t self_;
};

// select_block for tokens [token_lo, token_hi) of the core, from the code tiles of softmax8 (top-2 decided on the
// SFPU, ties to the lower expert): e1 = ctz(code), e2 = ctz(code with its lowest bit cleared) % 8. Writes
// expert | rank << kRankShift per slot (rank: among this range's slots of that expert). The weights (w1 / w2 in
// columns 2 / 4 of the token's row) are left to fill_weight_tiles, off the select's critical path.
// Returns the range's per-expert counts, 8 bits each (a core holds <= 128 tokens). wait_tile(half) is called before
// the first token of each 16-token half (the halves arrive one at a time).
// Loads and stores are non-volatile between compiler barriers: a code tile is static once waited for, and the
// callers' drain_stores orders the stores before any NoC read.

// Word offset of token `token`'s row: tile token/32, face 0 or 2, row token%16 of the face.
inline uint32_t token_row_offset(uint32_t token) {
    return ((token >> 5) << 10) | ((token & 16) << 5) | ((token & 15) << 4);
}

union ExpertCounts {
    uint8_t byte[kNumExperts];
    uint32_t word[2];
};

template <uint32_t rps, typename WaitTile>
inline uint64_t select_tokens(
    const volatile tt_l1_ptr uint32_t* code_words_v,
    volatile tt_l1_ptr uint32_t* slots_v,
    uint32_t token_lo,
    uint32_t token_hi,
    WaitTile&& wait_tile) {
    constexpr uint32_t tokens_per_shard = rps * kTile;
    const tt_l1_ptr uint32_t* code_words = const_cast<const tt_l1_ptr uint32_t*>(code_words_v);
    tt_l1_ptr uint32_t* slots = const_cast<tt_l1_ptr uint32_t*>(slots_v);
    ExpertCounts cnt{};
    const auto emit = [&](uint32_t token, uint32_t code) {
        const uint32_t first = __builtin_ctz(code);
        const uint32_t second = __builtin_ctz(code & (code - 1)) & 7;
        slots[token] = first | (static_cast<uint32_t>(cnt.byte[first]++) << kRankShift);
        slots[tokens_per_shard + token] = second | (static_cast<uint32_t>(cnt.byte[second]++) << kRankShift);
    };
    uint32_t token = token_lo;
    while (token < token_hi) {
        wait_tile(token / 16);  // per 16-token half (the loop starts each half group here)
        asm volatile("" ::: "memory");
        const uint32_t half_end = (token | 15) + 1;
        const uint32_t group_end = half_end < token_hi ? half_end : token_hi;
        uint32_t offset = token_row_offset(token);
        // The code words are freshly invalidated L1 lines (one a token row) and the select is load-latency bound, not
        // emit bound: 8 independent loads in flight before the emits (Perf R4.2, perf_experiments/r42_select_chain).
        for (; token + 8 <= group_end; token += 8, offset += 16 * 8) {
            uint32_t codes[8];
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                codes[i] = code_words[offset + 16 * i];
            }
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                emit(token + i, codes[i]);
            }
        }
        for (; token + 1 < group_end; token += 2, offset += 32) {  // two loads in flight before the stores
            const uint32_t first = code_words[offset];
            const uint32_t second = code_words[offset + 16];
            emit(token, first);
            emit(token + 1, second);
        }
        if (token < group_end) {
            emit(token, code_words[offset]);
            ++token;
        }
    }
    asm volatile("" ::: "memory");
    return (uint64_t{cnt.word[1]} << 32) | cnt.word[0];
}

// write_combine_weights_block's source: column 0 of the combine_weights tiles this RISC writes (tiles first,
// first + step, ... of the core's k-major 2 x core_tiles at tiles_base, zeroed beforehand), copied from w1 / w2
// (columns 2 / 4) of the code rows at code_base. Called in the RISC's wait for the count table, so the select on the
// critical path stores no weights. Tile-padding tokens stay 0.
template <uint32_t rps>
inline void fill_weight_tiles(
    uint32_t code_base, uint32_t tiles_base, uint32_t core_tiles, uint32_t core_tokens, uint32_t first, uint32_t step) {
    const tt_l1_ptr uint32_t* code = reinterpret_cast<const tt_l1_ptr uint32_t*>(code_base);
    tt_l1_ptr uint32_t* tiles = reinterpret_cast<tt_l1_ptr uint32_t*>(tiles_base);
    asm volatile("" ::: "memory");
    for (uint32_t write = first; write < 2 * core_tiles; write += step) {
        const uint32_t choice = write / core_tiles;
        const uint32_t tile = write - choice * core_tiles;
        const uint32_t lo = tile * kTile;
        const uint32_t hi = lo + kTile < core_tokens ? lo + kTile : core_tokens;
        tt_l1_ptr uint32_t* dst = tiles + choice * rps * kTileWords;
        const tt_l1_ptr uint32_t* src = code + 2 + 2 * choice;
        uint32_t token = lo;
        for (; token + 8 <= hi; token += 8) {  // 8 rows of one face: 8 loads in flight before the stores
            const uint32_t offset = token_row_offset(token);
            uint32_t words[8];
#pragma GCC unroll 8
            for (uint32_t row = 0; row < 8; ++row) {
                words[row] = src[offset + 16 * row];
            }
#pragma GCC unroll 8
            for (uint32_t row = 0; row < 8; ++row) {
                dst[offset + 16 * row] = words[row];
            }
        }
        for (; token < hi; ++token) {
            const uint32_t offset = token_row_offset(token);
            dst[offset] = src[offset];
        }
    }
    asm volatile("" ::: "memory");
}

inline uint32_t count_of(uint64_t counts, uint32_t expert) {
    return static_cast<uint32_t>(counts >> (expert * 8)) & 0xFF;
}

// Rows that hold no slot ("pad rows", any x row is a valid copy): each region's tail up to its 32-row boundary,
// then [regions_end, rows). Pad p of that concatenation belongs to push unit p % num_units (a unit: one DM RISC's
// tokens of one token group), so the pads spread evenly over every pushing RISC whatever the routing. Pushes the
// pads of unit `unit`, the j-th of them carrying the RISC's token j % risc_tokens (row slices from `staging`).
template <uint32_t row_slice_bytes, typename Pusher>
__attribute__((always_inline)) inline void push_pad_rows(
    Pusher& pusher,
    const volatile tt_l1_ptr uint32_t* header,
    uint32_t rows,
    uint32_t unit,
    uint32_t num_units,
    uint32_t staging,
    uint32_t risc_tokens,
    uint32_t& j) {
    uint32_t pad = unit;     // next pad of this unit
    uint32_t run_first = 0;  // pad index of the current run's first row
    for (uint32_t run = 0; run <= kNumExperts; ++run) {
        uint32_t lo, hi;
        if (run < kNumExperts) {
            const uint32_t start = header[kHeaderStarts + run];
            const uint32_t total = header[kHeaderTotals + run];
            lo = start + total;
            hi = start + round_up_to_tile(total);
        } else {
            lo = header[kHeaderEnd];
            hi = rows;
        }
        const uint32_t run_end = run_first + (hi - lo);
        for (; pad < run_end; pad += num_units) {
            pusher.push(staging + j * row_slice_bytes, lo + (pad - run_first), Pusher::kPadStream);
            j = j + 1 == risc_tokens ? 0 : j + 1;
        }
        run_first = run_end;
    }
}

// Slot rows of tokens [lo, hi): each packed slot becomes its output row base[expert] + rank, stored back into the
// slot page when keep_rows (combine_index is written from it).
template <bool keep_rows, uint32_t row_slice_bytes, uint32_t tokens_per_shard, typename Pusher>
__attribute__((always_inline)) inline void push_slot_rows(
    Pusher& pusher,
    volatile tt_l1_ptr uint32_t* rows_page,
    const uint32_t (&base)[kNumExperts],
    uint32_t src,
    uint32_t lo,
    uint32_t hi) {
    for (uint32_t t = lo; t < hi; ++t, src += row_slice_bytes) {
        const uint32_t first = rows_page[t];
        const uint32_t second = rows_page[tokens_per_shard + t];
        const uint32_t first_expert = first & kExpertMask;
        const uint32_t second_expert = second & kExpertMask;
        const uint32_t first_row = base[first_expert] + (first >> kRankShift);
        const uint32_t second_row = base[second_expert] + (second >> kRankShift);
        if constexpr (keep_rows) {  // combine_index source (leader only)
            rows_page[t] = first_row;
            rows_page[tokens_per_shard + t] = second_row;
        }
        pusher.push(src, first_row, first_expert);
        pusher.push(src, second_row, second_expert);
    }
}

// Pushes the slot rows and the pad rows of the RISC's tokens [token_lo, token_hi) (local to the core), whose row
// slices start at `staging`. rows_page: the slot page (k-major slots, then the header). The RISC pushes the pads of
// units [first_unit, first_unit + owned_units) of num_units, pad j of a unit carrying the RISC's token
// token_lo + j % (token_hi - token_lo). The slot walk starts at token token_lo + walk_offset % (token_hi - token_lo)
// and wraps (perf lamp L7: cores out of lockstep on the owners).
// noinline: the pusher stays a local of this frame, so its state is never reloaded around the NoC register stores.
template <uint32_t stage_tiles, uint32_t tokens_per_shard, typename Pusher>
__attribute__((noinline)) void push_rows_block(
    uint32_t pieces_arg,
    uint32_t gathered_base,
    uint32_t coords_arg,
    volatile tt_l1_ptr uint32_t* rows_page,
    const uint32_t (&base)[kNumExperts],
    uint32_t staging,
    uint32_t token_lo,
    uint32_t token_hi,
    uint32_t rows,
    uint32_t first_unit,
    uint32_t owned_units,
    uint32_t num_units,
    uint32_t walk_offset,
    bool keep_rows,
    uint32_t self) {
    constexpr uint32_t row_slice_bytes = stage_tiles * kTileRowBytes;
#ifdef ABLATE_PUSH  // /perf-measure: the whole push (software and writes) stubbed
    return;
#endif
    typename Pusher::Stream streams[Pusher::kStreams];
    Pusher pusher(pieces_arg, gathered_base, coords_arg, streams, self);
    pusher.start();
    if (pusher.idle()) {  // no pieces: only the leader's combine_index rows to store
        if (keep_rows) {
            push_slot_rows<true, row_slice_bytes, tokens_per_shard>(
                pusher, rows_page, base, staging, token_lo, token_hi);
        }
        pusher.finish();
        return;
    }
    const uint32_t risc_tokens = token_hi - token_lo;
    const uint32_t walk_start = token_lo + walk_offset % risc_tokens;
    const uint32_t walk_src = staging + (walk_start - token_lo) * row_slice_bytes;
    if (keep_rows) {
        push_slot_rows<true, row_slice_bytes, tokens_per_shard>(
            pusher, rows_page, base, walk_src, walk_start, token_hi);
        push_slot_rows<true, row_slice_bytes, tokens_per_shard>(pusher, rows_page, base, staging, token_lo, walk_start);
    } else {
        push_slot_rows<false, row_slice_bytes, tokens_per_shard>(
            pusher, rows_page, base, walk_src, walk_start, token_hi);
        push_slot_rows<false, row_slice_bytes, tokens_per_shard>(
            pusher, rows_page, base, staging, token_lo, walk_start);
    }
    for (uint32_t unit = first_unit; unit < first_unit + owned_units; ++unit) {
        uint32_t j = 0;
        push_pad_rows<row_slice_bytes>(
            pusher, rows_page + 2 * tokens_per_shard, rows, unit, num_units, staging, risc_tokens, j);
    }
    pusher.finish();
}

// write_combine_index_block for the RISC's tokens [token_lo, token_hi): one segment of page k per choice.
template <uint32_t tokens_per_shard, typename Accessor>
inline void write_combine_index(
    const Noc& noc,
    const Accessor& combine_index,
    uint32_t page_base,
    uint32_t first_token,
    uint32_t token_lo,
    uint32_t token_hi) {
    for (uint32_t choice = 0; choice < 2; ++choice) {
        noc.async_write(
            CoreLocalMem<uint32_t>(page_base + (choice * tokens_per_shard + token_lo) * 4),
            combine_index,
            (token_hi - token_lo) * 4,
            {},
            {.page_id = choice, .offset_bytes = (first_token + token_lo) * 4});
    }
}

// Static VC of the combine_weights DRAM writes. The row pushes are on VC 1 (= NOC_UNICAST_WRITE_VC) and VC 3
// (RowPusher), every other unicast write on NOC_UNICAST_WRITE_VC; a 4 KB write held at a DRAM endpoint on VC 1 blocks
// the pushes queued behind it in the routers on its path (head-of-line). The idle member's weights writes land mid-push
// and stalled the pushers behind them for 1-1.6k cycles (cmd-buf-ready stalls), which set the push makespan (Perf R4.1,
// r41_push_balance). Carve-out: with DRAM `dispatched` the weights stay on NOC_UNICAST_WRITE_VC: there VC 2 measured
// 2x512 +3.3..6.8%, 4x256 +1.7..2.2%, 4x512 +0.9..3.0% (the tail is the DRAM block writes, not the push).
template <bool out_dram>
constexpr uint32_t kWeightsVc = out_dram ? NOC_UNICAST_WRITE_VC : 2;

// write_combine_weights_block, one tile (column 0 = the selection's weights) per call: tile `write` of the core's
// k-major tiles, staged rps tiles a choice at `tiles_base`. A member writes tiles first, first + step, ...: the
// leader all, or (WEIGHTS_SPREAD) every G-th, alternated over the two DM RISCs when both hold tokens so the 1 MB of
// DRAM writes travels on both NoCs.
template <uint32_t rps, uint32_t t_pad, bool out_dram, typename Accessor>
inline void write_weight_tiles(
    const Noc& noc,
    const Accessor& combine_weights,
    uint32_t tiles_base,
    uint32_t first_token,
    uint32_t core_tiles,
    uint32_t first,
    uint32_t step) {
    for (uint32_t write = first; write < 2 * core_tiles; write += step) {
        const uint32_t choice = write / core_tiles;
        const uint32_t tile = write - choice * core_tiles;
        noc.async_write<NocOptions::CUSTOM_VC>(
            CoreLocalMem<uint32_t>(tiles_base + (choice * rps + tile) * kFp32TileBytes),
            combine_weights,
            kFp32TileBytes,
            {},
            {.page_id = choice * (t_pad / kTile) + first_token / kTile + tile},
            {.vc = kWeightsVc<out_dram>});
    }
}

}  // namespace route_dispatch
