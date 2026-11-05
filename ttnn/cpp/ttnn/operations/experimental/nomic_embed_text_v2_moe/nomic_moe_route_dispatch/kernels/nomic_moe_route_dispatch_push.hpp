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
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "nomic_moe_route_dispatch_w_layout.hpp"

namespace route_dispatch {

constexpr uint32_t kTile = 32;
constexpr uint32_t kTileRowBytes = 64;  // one row of one bf16 tile column (32 elements)
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kNumExperts = 8;
// Slot-rows page: slots k-major (2 x rps*32 words; packed expert | rank << 4 after select_block, the output row
// after push_rows_block), then the region header.
constexpr uint32_t kHeaderStarts = 0;
constexpr uint32_t kHeaderTotals = kNumExperts;
constexpr uint32_t kHeaderEnd = 2 * kNumExperts;
constexpr uint32_t kHeaderStaging = kHeaderEnd + 1;     // L1 address of the untilized shard (cb_x_rows)
constexpr uint32_t kHeaderWriterBase = kHeaderEnd + 2;  // first row of each expert for the writer's tokens
constexpr uint32_t kHeaderWords = kHeaderWriterBase + kNumExperts;
constexpr uint32_t kRankShift = 4;
constexpr uint32_t kExpertMask = (1u << kRankShift) - 1;
// Reader -> writer select handshake page (cb_select_go): the slot page address, the weight tiles address.
constexpr uint32_t kGoSlotPage = 0;
constexpr uint32_t kGoWeights = 1;
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

// W slices (G of ht fp32 tiles), faces 0 / 2 only in the compact layout of nomic_moe_route_dispatch_w_layout.hpp.
// Slices go out on RISC c % slice_nocs (0: reader / NOC0, 1: writer / NOC1; host W_SLICE_NOCS). Full-grid multicasts
// on one NoC serialize per command, not per byte, so every slice of a NoC goes out as ONE multicast from the sender
// of its first slice (channel c % slice_nocs, core w_sender_top - channel). Every receiver waits on the RISC whose
// NoC carries its slice.
constexpr uint32_t kFp32TileBytes = 4096;
constexpr uint32_t kMaxSlices = 3;
constexpr uint32_t kReaderRisc = 0;
constexpr uint32_t kWriterRisc = 1;
constexpr std::string_view kWeightCt[kMaxSlices] = {"w0_ct_offset", "w1_ct_offset", "w2_ct_offset"};
constexpr std::string_view kWeightRt[kMaxSlices] = {"w0_rt_offset", "w1_rt_offset", "w2_rt_offset"};
constexpr uint32_t kPartialFaceOffsets[] = {0, 2 * kWFaceBytes};  // faces 0 and 2 (columns 0..15) of an fp32 tile

#ifdef KERNEL_COMPILE_TIME_ARG_MAP
template <uint32_t C, uint32_t G, uint32_t slice_nocs, uint32_t risc>
FORCE_INLINE void send_weight_channel(const Noc& noc, uint32_t channel, uint32_t addr, uint32_t bytes) {
    if constexpr (C < slice_nocs && C < G) {
        if constexpr (C == risc) {
            if (channel == C) {
                constexpr auto pipe = dataflow_kernel_lib::McastArgs<
                    get_named_compile_time_arg_val(kWeightCt[C]),
                    get_named_compile_time_arg_val(kWeightRt[C])>();
                pipe.sender(noc).send(addr, addr, bytes);
                return;
            }
        }
        send_weight_channel<C + 1, G, slice_nocs, risc>(noc, channel, addr, bytes);
    }
}

template <uint32_t C, uint32_t G, uint32_t slice_nocs, uint32_t risc>
FORCE_INLINE void receive_weight_channel(const Noc& noc, uint32_t channel) {
    if constexpr (C < slice_nocs && C < G) {
        if constexpr (C == risc) {
            if (channel == C) {
                constexpr auto pipe = dataflow_kernel_lib::McastArgs<
                    get_named_compile_time_arg_val(kWeightCt[C]),
                    get_named_compile_time_arg_val(kWeightRt[C])>();
                pipe.receiver(noc).receive();
                return;
            }
        }
        receive_weight_channel<C + 1, G, slice_nocs, risc>(noc, channel);
    }
}

// receive_weight_slice on the RISC whose NoC carries slice `member`: it is cb_router_weight's producer on this core
// (one producer a core; the other RISC never touches the CB). A core that sent the slice itself does not wait.
template <uint32_t G, uint32_t slice_nocs, uint32_t risc>
FORCE_INLINE void receive_weight_slice(
    const Noc& noc, CircularBuffer& router_weight, uint32_t weight_tiles, uint32_t member, bool sent_here) {
    const uint32_t channel = member % slice_nocs;
    if (channel != risc) {
        return;
    }
    router_weight.reserve_back(weight_tiles);
    if (!sent_here) {
        receive_weight_channel<0, G, slice_nocs, risc>(noc, channel);
    }
    router_weight.push_back(weight_tiles);
}

// The sender of channel `channel` on this RISC (core w_sender_top - channel): faces 0 / 2 of every slice the channel
// carries, read from DRAM into their compact slots (consecutive), then one multicast. No-op on the other RISC.
template <uint32_t G, uint32_t ht, uint32_t slice_nocs, uint32_t risc, typename Accessor>
FORCE_INLINE void deliver_weight_slice(const Noc& noc, const Accessor& weight, uint32_t weight_base, uint32_t channel) {
    static_assert(slice_nocs == 1 || slice_nocs == 2, "W slices go out on the reader's NoC, or alternate over both");
    if (channel != risc || channel >= slice_nocs) {
        return;
    }
    const uint32_t slices = w_channel_slices(channel, G, slice_nocs);
    const uint32_t channel_addr = weight_base + w_slot(channel, G, slice_nocs) * ht * 2 * kWFaceBytes;
    for (uint32_t slice = channel, slot = 0; slot < slices; slice += slice_nocs, ++slot) {
        for (uint32_t tile = 0; tile < ht; ++tile) {
            const uint32_t dst = channel_addr + (slot * 2 * ht + w_face_tile(0, ht, tile)) * kWFaceBytes;
            const uint64_t src = weight.get_noc_addr(slice * ht + tile, 0, noc.get_noc_id());
            noc_async_read(src, dst, kWFaceBytes, noc.get_noc_id());
            noc_async_read(src + 2 * kWFaceBytes, dst + 2 * kWFaceBytes, kWFaceBytes, noc.get_noc_id());
        }
    }
    noc.async_read_barrier();
    send_weight_channel<0, G, slice_nocs, risc>(noc, channel, channel_addr, slices * ht * 2 * kWFaceBytes);
}
#endif

// exchange_partials_block, one direction: faces 0 and 2 (expert columns 0..15) of the member's rps partial tiles at
// `region` to the same address on members [first, last) of its group (core coordinates at common arg coords_arg).
template <uint32_t rps>
FORCE_INLINE void send_partial_faces(
    const Noc& noc, uint32_t region, uint32_t coords_arg, uint32_t first, uint32_t last) {
    for (uint32_t peer = first; peer < last; ++peer) {
        const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg + peer);
        for (uint32_t tile = 0; tile < rps; ++tile) {
            for (const uint32_t face : kPartialFaceOffsets) {
                const uint32_t addr = region + tile * kFp32TileBytes + face;
                noc.async_write(
                    CoreLocalMem<uint32_t>(addr),
                    UnicastEndpoint{},
                    kWFaceBytes,
                    {},
                    {.noc_x = coord_x(coord), .noc_y = coord_y(coord), .addr = addr});
            }
        }
    }
}

// Push RT args (host _push_args) and a piece's row filter (host ROWS_ALL / ROWS_EVEN / ROWS_ODD).
constexpr uint32_t kPushArgStageLo = 0;
constexpr uint32_t kPushArgPieces = 1;
constexpr uint32_t kPushArgFirstPiece = 2;
constexpr uint32_t kPushPieceWords = 5;
constexpr uint32_t kRowsAll = 0;

struct PushPiece {
    uint32_t piece;
    uint32_t src;
    uint32_t dst;
    uint32_t bytes;
    uint32_t rows;  // kRowsAll, or 1 + the parity of the rows this member pushes (two members split the piece)
};

// The RISC's push pieces (host _push_args); returns their count.
template <uint32_t k>
inline uint32_t load_push_pieces(uint32_t pieces_arg, PushPiece (&pieces)[k]) {
    const uint32_t num_pieces = get_arg_val<uint32_t>(pieces_arg + kPushArgPieces);
    for (uint32_t i = 0; i < num_pieces; ++i) {
        const uint32_t arg = pieces_arg + kPushArgFirstPiece + i * kPushPieceWords;
        pieces[i] = {
            get_arg_val<uint32_t>(arg),
            get_arg_val<uint32_t>(arg + 1),
            get_arg_val<uint32_t>(arg + 2),
            get_arg_val<uint32_t>(arg + 3),
            get_arg_val<uint32_t>(arg + 4)};
    }
    return num_pieces;
}

// At most bytes_in_flight of pushes outstanding: past a bytes threshold the NoC tips into congestion collapse
// (measured on nomic_moe_dispatch). The cap is counted in writes of this RISC's average piece, so a member with
// more, smaller writes keeps as many bytes in flight as the others. Credits: the ack counter is read only when the last
// reading's credits are spent. The push is RISC-issue-bound, so the per-write path keeps its state in registers: the
// piece loop is unrolled over k, and the issued-write count stays local until finish() adds it to the barrier counters
// once.
template <uint32_t w, uint32_t k, uint32_t num_cores, uint32_t bytes_in_flight, uint32_t cmd_bufs>
class RowPusher {
    static_assert(cmd_bufs == 1 || cmd_bufs == 2, "pushes use the write and the write-reg command buffers");
    static constexpr uint32_t kCmdBufs[2] = {write_cmd_buf, write_reg_cmd_buf};

public:
    // pieces_arg: the RISC's push RT args (host _push_args): first staged tile, piece count, then k pieces of
    // (piece, src offset in the staged row, dst offset in the piece, bytes, rows).
    RowPusher(uint32_t pieces_arg, uint32_t gathered_base, uint32_t coords_arg) :
        gathered_base_(gathered_base), coords_arg_(coords_arg) {
        num_pieces_ = load_push_pieces(pieces_arg, pieces_);
        uint32_t bytes = 0;
        for (uint32_t i = 0; i < num_pieces_; ++i) {
            bytes += pieces_[i].bytes;
        }
        const uint32_t writes = bytes > 0 ? bytes_in_flight * num_pieces_ / bytes : 0;
        writes_in_flight_ = writes > 0 ? writes : 1;
    }

    // Programs the write command buffer's fixed fields; call before a run of push() with no other write between.
    void start() {
        for (uint32_t buf = 0; buf < cmd_bufs; ++buf) {
            ncrisc_noc_write_set_state<false, false>(
                noc_index, kCmdBufs[buf], get_noc_addr(my_x[noc_index], my_y[noc_index], 0), 0, NOC_UNICAST_WRITE_VC);
        }
        acked_base_ = noc_nonposted_writes_acked[noc_index];
        issued_ = 0;
    }

    // One row slice (src_row: its staging address) to output row `row`.
    __attribute__((always_inline)) inline void push(uint32_t src_row, uint32_t row) {
        const uint32_t tile_row = row / kTile;
        const uint32_t row_dst = gathered_base_ + (row % kTile) * w * kTileRowBytes;
#pragma GCC unroll 3
        for (uint32_t i = 0; i < k; ++i) {
            if (i >= num_pieces_) {
                break;
            }
            const PushPiece& piece = pieces_[i];
            if (piece.rows != kRowsAll && (row & 1) + 1 != piece.rows) {
                continue;
            }
            const uint32_t shard = tile_row * k + piece.piece;
            const uint32_t block = shard / num_cores;
            const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg_ + (shard - block * num_cores));
            const uint32_t dst = row_dst + block * w * kBf16TileBytes + piece.dst;
            if (credits_ == 0) {
                uint32_t outstanding;
                while ((outstanding = acked_base_ + issued_ -
                                      NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED)) >= writes_in_flight_) {
                }
                credits_ = writes_in_flight_ - outstanding;
            }
            --credits_;
#ifdef ABLATE_PUSH_WRITES
            asm volatile("" ::"r"(dst), "r"(coord));
            continue;
#endif
            ++issued_;
            // Stateful one-packet write: only the target coordinate, addresses and length change per write.
            // Round robin over cmd_bufs command buffers (2 measured slower than 1 on 8x512: the poll is not the cost).
            const uint32_t cmd_buf = kCmdBufs[issued_ % cmd_bufs];
            while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
            }
            NOC_CMD_BUF_WRITE_REG(
                noc_index,
                cmd_buf,
                NOC_RET_ADDR_COORDINATE,
                static_cast<uint32_t>(get_noc_addr(coord_x(coord), coord_y(coord), 0) >> NOC_ADDR_COORD_SHIFT) &
                    NOC_COORDINATE_MASK);
            ncrisc_noc_write_with_state<DM_DEDICATED_NOC, false, false, false, true>(
                noc_index, cmd_buf, src_row + piece.src, dst, piece.bytes);
        }
    }

    // Accounts the issued writes in the counters a write barrier waits on.
    void finish() const {
        noc_nonposted_writes_num_issued[noc_index] += issued_;
        noc_nonposted_writes_acked[noc_index] += issued_;
    }

private:
    PushPiece pieces_[k]{};
    uint32_t num_pieces_ = 0;
    uint32_t writes_in_flight_ = 1;
    uint32_t credits_ = 0;
    uint32_t acked_base_ = 0;
    uint32_t issued_ = 0;
    uint32_t gathered_base_;
    uint32_t coords_arg_;
};

// select_block for tokens [token_lo, token_hi) of the core, from the code tiles of softmax8 (top-2 decided on the
// SFPU, ties to the lower expert): e1 = ctz(code), e2 = ctz(code with its lowest bit cleared) % 8, w1 / w2 in
// columns 2 / 4 of the token's row. Writes expert | rank << kRankShift per slot (rank: among this range's slots of
// that expert) and, when weight_words is set, the two weights into column 0 of the k-major combine_weights tiles.
// Returns the range's per-expert counts, 8 bits each (a core holds <= 128 tokens). wait_tile(tile) is called before
// the first token of each code tile (the tiles are published one at a time).
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
    volatile tt_l1_ptr uint32_t* weight_words_v,
    uint32_t token_lo,
    uint32_t token_hi,
    WaitTile&& wait_tile) {
    constexpr uint32_t tokens_per_shard = rps * kTile;
    const tt_l1_ptr uint32_t* code_words = const_cast<const tt_l1_ptr uint32_t*>(code_words_v);
    tt_l1_ptr uint32_t* slots = const_cast<tt_l1_ptr uint32_t*>(slots_v);
    tt_l1_ptr uint32_t* weight_words = const_cast<tt_l1_ptr uint32_t*>(weight_words_v);
    ExpertCounts cnt{};
    struct Row {
        uint32_t code, w1, w2;
    };
    const auto load = [&](uint32_t offset) {
        return Row{code_words[offset], code_words[offset + 2], code_words[offset + 4]};
    };
    const auto emit = [&](uint32_t token, uint32_t offset, const Row& row) {
        const uint32_t first = __builtin_ctz(row.code);
        const uint32_t second = __builtin_ctz(row.code & (row.code - 1)) & 7;
        slots[token] = first | (static_cast<uint32_t>(cnt.byte[first]++) << kRankShift);
        slots[tokens_per_shard + token] = second | (static_cast<uint32_t>(cnt.byte[second]++) << kRankShift);
        if (weight_words != nullptr) {
            weight_words[offset] = row.w1;
            weight_words[rps * kTileWords + offset] = row.w2;
        }
    };
    uint32_t token = token_lo;
    while (token < token_hi) {
        if (token == token_lo || (token & (kTile - 1)) == 0) {
            wait_tile(token / kTile);
            asm volatile("" ::: "memory");
        }
        const uint32_t half_end = (token | 15) + 1;
        const uint32_t group_end = half_end < token_hi ? half_end : token_hi;
        uint32_t offset = token_row_offset(token);
        for (; token + 1 < group_end; token += 2, offset += 32) {  // two loads in flight before the stores
            const Row first = load(offset);
            const Row second = load(offset + 16);
            emit(token, offset, first);
            emit(token + 1, offset + 16, second);
        }
        if (token < group_end) {
            emit(token, offset, load(offset));
            ++token;
        }
    }
    asm volatile("" ::: "memory");
    return (uint64_t{cnt.word[1]} << 32) | cnt.word[0];
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
    uint32_t risc_tokens) {
    uint32_t pad = unit;     // next pad of this unit
    uint32_t run_first = 0;  // pad index of the current run's first row
    uint32_t j = 0;
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
        for (; pad < run_end; pad += num_units, ++j) {
            pusher.push(staging + (j % risc_tokens) * row_slice_bytes, lo + (pad - run_first));
        }
        run_first = run_end;
    }
}

// Pushes the slot rows and the pad rows of the RISC's tokens [token_lo, token_hi) (local to the core), whose row
// slices start at `staging`. Each packed slot becomes its output row base[expert] + rank, stored back into the
// slot page when keep_rows (combine_index is written from it). rows_page: the slot page (k-major slots, then the
// header). The RISC pushes the pads of units [first_unit, first_unit + owned_units) of num_units, pad j of a unit
// carrying the RISC's token token_lo + j % (token_hi - token_lo). The slot walk starts at token
// token_lo + walk_offset % (token_hi - token_lo) and wraps (perf lamp L7: cores out of lockstep on the owners).
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
    bool keep_rows) {
    constexpr uint32_t row_slice_bytes = stage_tiles * kTileRowBytes;
    Pusher pusher(pieces_arg, gathered_base, coords_arg);
    pusher.start();
    const uint32_t risc_tokens = token_hi - token_lo;
    const uint32_t walk_start = token_lo + walk_offset % risc_tokens;
    for (uint32_t step = 0, t = walk_start; step < risc_tokens; ++step, t = t + 1 == token_hi ? token_lo : t + 1) {
        const uint32_t src = staging + (t - token_lo) * row_slice_bytes;
        const uint32_t first = rows_page[t];
        const uint32_t second = rows_page[tokens_per_shard + t];
        const uint32_t first_row = base[first & kExpertMask] + (first >> kRankShift);
        const uint32_t second_row = base[second & kExpertMask] + (second >> kRankShift);
        if (keep_rows) {  // combine_index source (leader only)
            rows_page[t] = first_row;
            rows_page[tokens_per_shard + t] = second_row;
        }
        pusher.push(src, first_row);
        pusher.push(src, second_row);
    }
    for (uint32_t unit = first_unit; unit < first_unit + owned_units; ++unit) {
        push_pad_rows<row_slice_bytes>(
            pusher, rows_page + 2 * tokens_per_shard, rows, unit, num_units, staging, risc_tokens);
    }
    pusher.finish();
}

// Per-owner counted arrival (op_design.md lamp L3): owner block b of a core counts the 64 B row granules landed in it
// on semaphore sem_arrived + b, and is complete at kTile * w. A pushing RISC tallies the rows it wrote per output tile
// row (runs of rows in one tile row merge into one increment a piece) and, once its writes are acked, adds each
// block's granules with one NoC atomic: about one atomic per (expert region tile row, piece) and per pad, not per
// write.
template <uint32_t k, uint32_t num_cores>
class ArrivalCounter {
public:
    ArrivalCounter(const Noc& noc, uint32_t pieces_arg, uint32_t coords_arg, uint32_t sem_arrived) :
        noc_(noc), coords_arg_(coords_arg), sem_arrived_(sem_arrived) {
        num_pieces_ = load_push_pieces(pieces_arg, pieces_);
    }

    bool pushes() const { return num_pieces_ > 0; }

    // Rows [lo, hi) were pushed.
    void add_rows(uint32_t lo, uint32_t hi) {
        while (lo < hi) {
            const uint32_t tile_row = lo / kTile;
            const uint32_t end = (tile_row + 1) * kTile < hi ? (tile_row + 1) * kTile : hi;
            if (tile_row != tile_row_) {
                flush();
                tile_row_ = tile_row;
            }
            even_rows_ += (end + 1) / 2 - (lo + 1) / 2;
            odd_rows_ += end / 2 - lo / 2;
            lo = end;
        }
    }

    // push_pad_rows visitor: pad row `row` was pushed.
    void push(uint32_t /*src_row*/, uint32_t row) { add_rows(row, row + 1); }

    // Issues the pending tile row's increments.
    void flush() {
        for (uint32_t i = 0; i < num_pieces_; ++i) {
            const PushPiece& piece = pieces_[i];
            const uint32_t rows =
                piece.rows == kRowsAll ? even_rows_ + odd_rows_ : (piece.rows == 1 ? even_rows_ : odd_rows_);
            if (rows == 0) {
                continue;
            }
            const uint32_t shard = tile_row_ * k + piece.piece;
            const uint32_t block = shard / num_cores;
            const uint32_t coord = get_common_arg_val<uint32_t>(coords_arg_ + (shard - block * num_cores));
            Semaphore<>(sem_arrived_ + block)
                .up(noc_, coord_x(coord), coord_y(coord), rows * (piece.bytes / kTileRowBytes));
        }
        even_rows_ = 0;
        odd_rows_ = 0;
    }

private:
    const Noc& noc_;
    PushPiece pieces_[k]{};
    uint32_t num_pieces_ = 0;
    uint32_t coords_arg_;
    uint32_t sem_arrived_;
    uint32_t tile_row_ = 0;
    uint32_t even_rows_ = 0;
    uint32_t odd_rows_ = 0;
};

// Signals the arrivals of everything push_rows_block pushed for the same RISC (same units, risc_counts[e] slots of
// expert e from row base[e]); call after the pushes are acked (write barrier).
template <uint32_t k, uint32_t num_cores>
inline void signal_arrivals(
    const Noc& noc,
    uint32_t pieces_arg,
    uint32_t coords_arg,
    uint32_t sem_arrived,
    const volatile tt_l1_ptr uint32_t* header,
    const uint32_t (&base)[kNumExperts],
    const uint32_t (&risc_counts)[kNumExperts],
    uint32_t rows,
    uint32_t first_unit,
    uint32_t owned_units,
    uint32_t num_units,
    uint32_t risc_tokens) {
    ArrivalCounter<k, num_cores> counter(noc, pieces_arg, coords_arg, sem_arrived);
    if (!counter.pushes()) {
        return;
    }
    for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
        counter.add_rows(base[expert], base[expert] + risc_counts[expert]);
    }
    for (uint32_t unit = first_unit; unit < first_unit + owned_units; ++unit) {
        push_pad_rows<0>(counter, header, rows, unit, num_units, 0, risc_tokens);
    }
    counter.flush();
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

// write_combine_weights_block, one tile (column 0 = the selection's weights) per call: tile `write` of the core's
// k-major tiles, staged rps tiles a choice at `tiles_base`. A member writes tiles first, first + step, ...: the
// leader all, or (WEIGHTS_SPREAD) every G-th, alternated over the two DM RISCs when both hold tokens so the 1 MB of
// DRAM writes travels on both NoCs.
template <uint32_t rps, uint32_t t_pad, typename Accessor>
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
        noc.async_write(
            CoreLocalMem<uint32_t>(tiles_base + (choice * rps + tile) * kFp32TileBytes),
            combine_weights,
            kFp32TileBytes,
            {},
            {.page_id = choice * (t_pad / kTile) + first_token / kTile + tile});
    }
}

}  // namespace route_dispatch
