// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul reader (NCRISC / NoC0) — block operation `load_kblocks` of regime R1.
//
// For every block of this core's region (b-block, m-block, n-block; n fastest) and every K-block,
// reads A[block_m x block_k] and B[block_k x block_n] from interleaved DRAM/L1 through
// TensorAccessor, ONE barrier per K-block covering both operands, one reserve/push per operand.
// Only valid pages are read (rows < tensor_m, cols < tensor_n, K tiles < last_block_k on the last
// K-block) but the NOMINAL page count is always pushed, so the CB rings stay closed and the compute
// kernel runs the nominal block shape (ragged K is excluded in compute by the inner-dim functor).
//
// Epilogue operands (Refinement 2):
//   - swiglu: within each N-block, CB column c <- B column n0 + (c < h ? 2c : 2(c - h) + 1),
//     h = block_n / 2, so all gate tiles of the block precede all up tiles (the compute epilogue
//     strides over the two halves).
//   - bias / gate row (`cb_row`): block_n tiles of the (1, N) row, read ONCE per block, pushed right
//     after the first batch's K-walk (so it never stalls the K-block prefetch; compute waits for it
//     before the first epilogue and pops it after the last batch of the block).
//   - residual (addcmul): row groups of sb_h x block_n tiles after each batch's K-walk, in the
//     epilogue's consumption order; depth 2 row groups.
//
// Operand residency (Refinement 8, R1 / R1a, one K-block): `weight_resident` -- the core's rows are split
// into M-blocks over its whole column range; B is read (or line-delivered) and pushed on the first M-block
// only and compute keeps it fronted across the rest, so only A streams. `act_resident` (the twin) -- the
// columns are split into N-blocks; each M-block's A is read / delivered and pushed on its first N-block
// only, so only B streams. Every core of an R1a line walks the same nominal sequence, so the skipped
// deliveries match on injector and receivers.
//
// Regime R1a `interleaved_2d_mcast` (Refinement 3, `operand_mcast` = 1): same region, CBs and walk,
// only the operand SOURCE changes. A K-block depends only on the core's row region, B only on its
// column region, so one injector per line reads it from DRAM and multicasts it into the line's
// cb_act_tiles / cb_weight_tiles (ttnn.Mcast + McastArgs, handshake = the receiver's CB slot is
// reserved). `a_sends` / `b_sends` (CT, per kernel group) mark this core as the A-line / B-line
// injector; every other core of the line receives. All cores of a line walk the same nominal block
// sequence (host passes the grid-uniform nb_m / nb_n), so their CB write pointers stay in lockstep
// and the multicast lands at the sender's own slot address. Per K-block every core delivers A, then
// B: no core can block in B(kb) before every core has passed A(kb), so the two channels cannot
// deadlock. The bias / gate row and the residual stay per-core reads (they are tiny or unshared).
// R1a split lines (Refinement 16, GM_R1A_SPLIT, line_stream.hpp): the A row line runs on the writer instead and this
// reader runs only the B column line, with rotating injectors staging their turns ahead.
//
// Regime R1b `interleaved_stream_mcast_a` (Refinement 9, `operand_mode` = 2): a 1-D N split (every core
// owns all rows), so A is reuse-shared by every core. The writer RISC produces cb_act_tiles (one injector
// reads each A K-block and broadcasts it, genify_matmul_writer.cpp); this reader streams only its own B
// column slice (+ the epilogue operands), never waiting on the A handshake.
//
// Regime R4 `dram_sharded` (Refinement 4, `dram_sharded` = 1): one worker per DRAM bank owns the block
// rows x (its bank's B column shard) and walks K one A shard per round (`k_blocks_per_round` K-blocks
// per shard). Per K-block it (1) issues ONE coalesced read of the K-block's rows from its bank shard
// (the rows of a bank shard are contiguous) -- before waiting on A, since B does not depend on it --
// then (2) runs the A round on the `act` channel: the core owning A shard `round` multicasts the
// K-block from its resident L1 shard into every worker's reserved cb_act_tiles slot (rotating senders,
// ttnn.Mcast + McastArgs); a worker that owns that shard sends instead of receiving. Cores of the
// broadcast that are not workers (`ds_worker` = 0) only send their own rounds (no CB traffic: they
// address the workers' slot as base + (K-block % depth) * slot bytes) or, inside the receiver
// rectangle, just land the payload and exit. B columns stay in natural (gate, up) order for swiglu.
//
// Regime R3 `block_sharded` (Refinement 5, `block_sharded` = 1): A is L1 block-sharded (rows over grid
// rows, K over grid columns), so a core owns the block rows of its grid row x the output columns of its
// grid column (B column slice read from interleaved DRAM, R1's per-tile reads) and walks K one A shard
// per K-block: K-block kb is the shard of grid column kb of its row. On its own turn (kb == own_kb) the
// core pushes its RESIDENT shard rows into cb_act_shard (backed on A's shard: zero-copy, no NoC read);
// compute selects that CB for the block (In0SourceFn). Peer K-slices land in cb_act_tiles:
//   bs_mcast = 0 (R3 stepping stone): unicast reads of the peer's shard tiles through A's (sharded)
//                TensorAccessor; nothing is pushed to cb_act_tiles on the own turn.
//   bs_mcast = 1 (R3a): a rotating-sender row multicast on the `act` channel -- in round kb the core of
//                grid column kb sends its shard rows from L1 to every row peer's reserved cb_act_tiles
//                slot. To keep the row's cb_act_tiles write pointers in lockstep (the multicast lands at
//                the sender's own slot address) the sender also reserves / pushes its slot on its own
//                turn (it receives the loopback copy); compute retires that slot unread.
// B's K-block reads are issued before the A round (they do not depend on it); one barrier per K-block.
//
// Regime R2 `height_sharded` (Refinement 7) is the same walk with ONE K shard (`block_sharded` = 1,
// bs_mcast = 0, own_kb = 0): every K-block is the core's own resident shard (pushed into cb_act_shard,
// which compute reads directly -- no cb_act_tiles exists), B's column range from DRAM.
// Batched B (R2 / R3): the host maps each shard onto (b0, row0) so no block straddles a batch; a shard
// holding whole images walks them as the block's batch slices (block_batches), pushing the resident
// shard image by image, i.e. sequentially.
//
// R2a / R3a weight multicast (Refinement 10, `bs_b_mcast` = 1, single B only): B does not vary along the
// shard rows, so every core of a B group -- R3: a grid column (it owns one output column slice); R2: all of
// A's shard cores -- needs the same B K-blocks. One fixed injector per group (`weight` channel, staggered
// so R3's column injectors sit on different grid rows) reads them from DRAM once and multicasts them into
// every group core's reserved cb_weight_tiles slot (lockstep CB pointers: one uniform walk). A B round
// carries `b_round_kblocks` K-blocks (contiguous K rows; the CB holds whole rounds), reserved together and
// pushed one per K-block; the injector issues the round's reads (trid per chunk) before the A round
// (stall-shadow) and every core runs A round -> B round in that order, so the two channels cannot deadlock.
// The walk's first bias / gate row is the same on the whole group (a DRAM hotspot when each core reads it):
// the injector reads it beside B and multicasts it in one row round right after the first B round.

// Per-stage device zones (Perf 1, PERMANENT -- free with the profiler off; never remove, extend instead).
// One zone per question, per K-block (<= 5 executions per K-block; the 125-zone budget covers ~25 K-blocks
// per core, beyond which later zones are dropped SILENTLY -- check the span before trusting a breakdown):
//   rd_reserve   cb_reserve_back on the operand slots   (back-pressure: compute not popping)
//   rd_issue     the NoC read issue loop (address gen + command-buffer writes; RISC-serial)
//   rd_barrier   noc_async_read_barrier                 (waiting on DRAM / NoC)
//   rd_mcast_a   R1a A-line delivery (injector: send incl. receivers' slot acks; receiver: wait for data)
//   rd_mcast_b   R1a B-line delivery
//   rd_ds_kblock / rd_bs_kblock   one zone per R4 / R3 K-block (B issue + A round + barrier)
//   rd_row / rd_residual          epilogue operand reads
//   rd_b_issue     (Refinement 10) R2a / R3a injector: B round + first-row read issue
//   rd_b_round     R2a / R3a B round (injector: trid barrier + send incl. the group's acks; receiver: wait)
//   rd_row_round   R2a / R3a first bias / gate row round
//   rd_tail_write  (Perf 2) the reader's share of the last block's output writes on NoC0, per output batch
//                  (OCCUPANCY: includes the wait for the writer's post, i.e. for compute's epilogue)
// Perf 2 head ramp (genify_matmul_kschedule.hpp): the R1 / R1a K loop walks the schedule's deliveries (one
// more than num_k_blocks on PERF FOCUS 1), so the per-K-block zones above run once per delivery.
// Perf 2 tail write split (tail_split.hpp): after its K-walk this RISC writes TAIL_NC_PCT % of every output row
// of the core's last block on NoC0 (the writer keeps the rest on NoC1 and stays the output CB's only consumer).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_ds_act.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/tail_split.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/out_chunks.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/a_segments.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_kschedule.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/line_stream.hpp"

// R1a injector stall-shadow (Refinement 11, host R1A_INJECTOR_AHEAD): a line injector issues K-block kb+1's
// DRAM reads (own transaction id) before it multicasts kb, so its line's read and multicast overlap instead
// of alternating. Needs operand rings of GM_R1A_DEPTH >= 3 K-blocks (compute holds kb-1 while kb is sent and
// kb+1 lands). The helper walk only (no head ramp, no residency, one batch per block). 1 = read, barrier, send.
#ifndef GM_R1A_AHEAD
#define GM_R1A_AHEAD 1
#endif
// R1b bank-local B stream (Refinement 15, host ACT_STREAM_BANK_LOCAL): a core's CB units (B columns; swiglu:
// (gate, up) column pairs) are GM_BANK_STRIDE B columns apart instead of adjacent -- with the stride equal to the
// number of DRAM banks every unit of a core lives in one bank (a column's bank is page % banks), and consecutive
// units are consecutive pages of that bank. GM_BANK_RUNS (host: B's aligned page == its tile, stride == banks):
// those units are one contiguous bank run, so each K row of the core's slice is ONE read (two under swiglu: the
// gate run and the up run, landing in the CB's gate / up halves). 0 = the adjacent-column mapping.
#ifndef GM_BANK_STRIDE
#define GM_BANK_STRIDE 0
#endif
#ifndef GM_BANK_RUNS
#define GM_BANK_RUNS 0
#endif
#ifndef GM_OUT_STRIDE
#define GM_OUT_STRIDE 1
#endif
#ifndef GM_R1A_DEPTH
#define GM_R1A_DEPTH 2
#endif
// R1a split lines (Refinement 16, host R1A_SPLIT_LINES; line_stream.hpp): the A row lines run on the writer RISC
// (it produces cb_act_tiles), this reader runs only the B column line, with rotating injectors (round r of a column
// is read + multicast by its core r % grid_m). 0 = both lines here, one fixed injector per line.
#ifndef GM_R1A_SPLIT
#define GM_R1A_SPLIT 0
#endif
// The line stream's staging buffer (one K-block of the operand; scratch CB owned by this RISC).
#ifndef GM_LINE_STAGE_CB
#define GM_LINE_STAGE_CB 0
#endif
// ... and the rounds ahead over which an injector spreads its turn's reads (host R1A_LINE_WINDOW).
#ifndef GM_LINE_WINDOW
#define GM_LINE_WINDOW 2
#endif
#ifndef GM_LINE_STARTUP
#define GM_LINE_STARTUP 1
#endif
// R2a / R3a one-shot weight channel (Refinement 18, host SHARDED_B_ONE_SHOT): the walk is ONE B round in ONE chunk,
// so the channel runs handshake-free (every slot it lands in is fresh) and carries exactly one delivery per walk
// (one flag set, one clear: no flag race). With a bias / gate row (GM_BS_ROW_TAIL) that delivery is
// [B round | row]: the injector reads the walk's first row into cb_weight_tiles' tail (host-sized pages past the
// round's slot) beside B, and every core copies it L1 -> L1 into cb_row after the walk -- no separate row round.
#ifndef GM_BS_ROW_TAIL
#define GM_BS_ROW_TAIL 0
#endif
// R3 split roles (Refinement 18, host BS_B_ON_WRITER): this kernel runs the R3 walk's A half only (1, NCRISC) or its
// B half only (2, BRISC / NoC1), so the column's B rounds overlap the row's A rounds instead of queueing behind them on
// one RISC. 0 = both halves here.
#ifndef GM_BS_ROLE
#define GM_BS_ROLE 0
#endif
// R3a one-shot A exchange (Refinement 18, host BS_A_ONE_SHOT): a single-block walk whose cb_act_tiles holds every
// K-block (one slot per grid column, all fresh) exchanges the row's shards at once -- every core multicasts its
// shard into its own column's slot on every row peer up front (handshake-free, counter signal) and waits for the
// count to reach the row's column count -- instead of num_k_blocks handshaked rounds in turn.
#ifndef GM_BS_A_ONE_SHOT
#define GM_BS_A_ONE_SHOT 0
#endif
// R2a B-read helper (Refinement 18, host SHARDED_B_READ_HELPER): on a one-shot walk's injector the B round's DRAM
// read issue (RISC-serial, ~27 ns a tile) is split across both RISCs. The injector's BRISC (GM_BS_ROLE 3: no other
// work there, the output is packed in place) reads the round's last floor(K / 2) K rows into the same slot and posts
// local semaphore GM_BS_B_HELPER_SEM; the reader (NCRISC) reads the first ceil(K / 2) rows (+ the row tail) and
// waits for the post before handing the round over and multicasting it. -1 = no helper.
#ifndef GM_BS_B_HELPER_SEM
#define GM_BS_B_HELPER_SEM -1
#endif

namespace {

// Valid extent of a block starting at `start` inside a region of `valid` tiles (0 when the block lies
// wholly in the padded tail of a ragged edge region: the grid-uniform R1a walk visits such blocks).
FORCE_INLINE uint32_t block_extent(uint32_t valid, uint32_t start, uint32_t nominal) {
    if (valid <= start) {
        return 0;
    }
    return (valid - start) < nominal ? (valid - start) : nominal;
}

// One line pipe per operand: the sender face on the injector, the receiver face elsewhere, nothing
// when the channel is absent or not a line channel (regime R1: every core reads for itself; R4: the
// act channel is the rotating A broadcast, built separately).
template <bool SENDS, bool LINE, typename Channel>
FORCE_INLINE auto make_line_pipe(const Channel& channel, const Noc& noc) {
    if constexpr (!LINE || !Channel::active) {
        return 0;
    } else if constexpr (SENDS) {
        return channel.sender(noc);
    } else {
        return channel.receiver(noc);
    }
}

// Deliver one K-block over the line: the injector multicasts its freshly read slot to the same
// address on every receiver; a receiver acks its reserved slot and waits for the data.
template <bool ACTIVE, bool SENDS, typename Pipe>
FORCE_INLINE void line_deliver(Pipe& pipe, uint32_t l1, uint32_t bytes) {
    if constexpr (ACTIVE) {
        if constexpr (SENDS) {
            pipe.send(l1, l1, bytes);
        } else {
            pipe.receive();
        }
    }
}

// R2a / R3a B channel faces (std::nullopt where this core has no such role); 0 when the channel is off, so
// a channel serving another regime (R1a's B line) never gets a second pipe constructed over it.
template <bool ON, bool SENDER, typename Channel>
FORCE_INLINE auto make_bs_face(const Channel& channel, const Noc& noc) {
    if constexpr (!ON) {
        return 0;
    } else if constexpr (SENDER) {
        return channel.optional_sender(noc);
    } else {
        return channel.optional_receiver(noc);
    }
}

template <bool ON, typename Face>
FORCE_INLINE bool bs_has_face(const Face& face) {
    if constexpr (ON) {
        return face.has_value();
    } else {
        return false;
    }
}

// One B chunk over the group: the injector multicasts its slot range to the same address on every receiver;
// a receiver acks its reserved slot and waits for the data.
template <bool ON, typename Sender, typename Receiver>
FORCE_INLINE void bs_b_deliver(bool injector, Sender& sender, Receiver& receiver, uint32_t l1, uint32_t bytes) {
    if constexpr (ON) {
        if (injector) {
            sender->send(l1, l1, bytes);
        } else {
            receiver->receive();
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_batches = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t b_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t a_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t b_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t epilogue = get_compile_time_arg_val(12);  // 0 none, 1 bias, 2 bias_relu, 3 swiglu, 4 addcmul
    constexpr uint32_t cb_row = get_compile_time_arg_val(13);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t sb_h = get_compile_time_arg_val(15);
    constexpr uint32_t row_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t row_page_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t res_tile_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t res_page_bytes = get_compile_time_arg_val(19);
    constexpr bool weight_resident = get_compile_time_arg_val(20) != 0;    // B read once per core (R1)
    constexpr bool act_resident = get_compile_time_arg_val(21) != 0;       // A read once per M-block (R1)
    // Operand source mode: 0 = R1 (every core reads both operands), 1 = R1a (line injectors multicast
    // A and B), 2 = R1b (A K-blocks arrive on the WRITER's broadcast stream; this reader streams B only).
    constexpr uint32_t operand_mode = get_compile_time_arg_val(22);
    constexpr bool operand_mcast = operand_mode == 1;                      // regime R1a
    constexpr bool r1a_split = operand_mcast && GM_R1A_SPLIT != 0;         // R1a split lines (A on the writer)
    constexpr bool act_on_writer = operand_mode == 2;                      // regime R1b
    constexpr bool a_sends = get_compile_time_arg_val(23) != 0;            // A-line injector (R1a)
    constexpr bool b_sends = get_compile_time_arg_val(24) != 0;            // B-line injector (R1a)
    constexpr bool dram_sharded = get_compile_time_arg_val(25) != 0;       // regime R4
    constexpr bool ds_worker = get_compile_time_arg_val(26) != 0;          // R4 bank worker (else sender / lander)
    constexpr uint32_t k_blocks_per_round = get_compile_time_arg_val(27);  // R4: K-blocks per A shard
    constexpr uint32_t act_cb_blocks = get_compile_time_arg_val(28);       // cb_act_tiles depth in K-blocks
    constexpr bool ds_row_reads = get_compile_time_arg_val(29) != 0;       // R4: column slice, one read per K row
    constexpr uint32_t ds_weight_depth = get_compile_time_arg_val(30);     // R4: cb_weight_tiles depth in K-blocks
    constexpr uint32_t ds_b_ahead = get_compile_time_arg_val(31);          // R4: B K-blocks in flight (trids)
    constexpr bool ds_reader_act = get_compile_time_arg_val(32) != 0;      // R4: A rounds here (else the writer)
    constexpr bool block_sharded = get_compile_time_arg_val(33) != 0;      // regime R3
    constexpr bool bs_mcast = get_compile_time_arg_val(34) != 0;           // R3a: rotating row multicast
    constexpr uint32_t cb_act_shard = get_compile_time_arg_val(35);        // R3: A's resident shard (zero-copy)
    // R2a / R3a weight multicast (Refinement 10): one injector per B-sharing group (R3: a grid column; R2: all
    // shard cores) reads B once and multicasts it, one B round per `b_round_kblocks` K-blocks (their slots are
    // contiguous K rows), each round in `b_chunks` K-row chunks of `b_chunk_k` rows.
    constexpr bool bs_b_mcast = get_compile_time_arg_val(36) != 0;
    constexpr uint32_t b_chunks = get_compile_time_arg_val(37);
    constexpr uint32_t b_chunk_k = get_compile_time_arg_val(38);
    constexpr uint32_t b_round_kblocks = get_compile_time_arg_val(39);
    constexpr auto a_args = TensorAccessorArgs<40>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto row_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto res_args = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();

    constexpr bool is_swiglu = epilogue == 3;
    constexpr bool has_row = epilogue == 1 || epilogue == 2 || epilogue == 4;
    constexpr bool has_residual = epilogue == 4;
    constexpr uint32_t half_n = block_n_tiles / 2;
    // CB unit -> B column: unit u of a core's region is B column col0 + u * unit_stride (+1 for swiglu's up tile).
    constexpr uint32_t unit_cols = is_swiglu ? 2 : 1;
    constexpr uint32_t unit_stride = GM_BANK_STRIDE != 0 ? GM_BANK_STRIDE : unit_cols;
    constexpr bool bank_runs = GM_BANK_RUNS != 0;
    static_assert(!bank_runs || GM_BANK_STRIDE != 0);
    static_assert(GM_BANK_STRIDE == 0 || (act_on_writer && !weight_resident && !act_resident));
    constexpr uint32_t num_row_groups = block_m_tiles / sb_h;
    constexpr uint32_t residual_group_tiles = sb_h * block_n_tiles;

    constexpr uint32_t act_block_tiles = block_m_tiles * block_k_tiles;
    constexpr uint32_t weight_block_tiles = block_k_tiles * block_n_tiles;
    // K-delivery schedule (Perf 2 head ramp, genify_matmul_kschedule.hpp; the identity without GM_K_RAMP).
    using KS = gm_ks::Sched<block_k_tiles, num_k_blocks, last_block_k_tiles>;
    static_assert(
        !gm_ks::ramp || (!dram_sharded && !block_sharded && !act_on_writer && !weight_resident && !act_resident),
        "the head ramp runs on the R1 / R1a K-walk only");
    // R1: every core reads both operands. R1a: only the line injectors read from DRAM. R1b: B only (A is
    // produced into cb_act_tiles by the writer).
    constexpr bool reads_act = !act_on_writer && !r1a_split && (!operand_mcast || a_sends);
    constexpr bool reads_weight = !r1a_split && (!operand_mcast || b_sends);
    constexpr bool r1a_prefetch = GM_R1A_AHEAD > 1;
#ifdef GM_ABLATE_DM
    constexpr bool gm_ablate_dm = true;
#else
    constexpr bool gm_ablate_dm = false;
#endif
    constexpr uint32_t r1a_depth = GM_R1A_DEPTH;
    static_assert(
        !r1a_prefetch || (GM_R1A_AHEAD == 2 && r1a_depth >= 3 && operand_mcast && !gm_ks::ramp && !weight_resident &&
                          !act_resident && block_batches == 1),
        "the R1a injector prefetch runs on the plain R1a helper walk only");

    // A rides the A-line channel, B the B-line channel (absent under R1).
    constexpr auto mc_act = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("act_ct_offset"), get_named_compile_time_arg_val("act_rt_offset")>();
    constexpr auto mc_weight = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("weight_ct_offset"),
        get_named_compile_time_arg_val("weight_rt_offset")>();
    static_assert(
        mc_act.active == ((operand_mcast && !r1a_split) || (dram_sharded && ds_reader_act) || bs_mcast) &&
        mc_weight.active == (operand_mcast || bs_b_mcast));
    static_assert(
        !bs_b_mcast ||
        (block_sharded && b_chunks >= 1 && b_chunks <= 8 && b_round_kblocks >= 1 &&
         num_k_blocks % b_round_kblocks == 0 && b_chunks * b_chunk_k >= b_round_kblocks * block_k_tiles));
    static_assert(bs_b_mcast || b_round_kblocks == 1);
    static_assert(operand_mode <= 2 && int(operand_mode != 0) + int(dram_sharded) + int(block_sharded) <= 1);
    static_assert(!bs_mcast || block_sharded);
    static_assert(ds_b_ahead >= 1 && ds_b_ahead <= ds_weight_depth);
    static_assert(
        !(weight_resident || act_resident) || (num_k_blocks == 1 && block_batches == 1 && !dram_sharded &&
                                               !block_sharded && !(weight_resident && act_resident)));

    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b0 = get_arg_val<uint32_t>(arg++);
    const uint32_t batches_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t cols_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_b = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_m = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_n = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_m_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_k_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t weight_batch_stride = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t row_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t res_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t own_kb = get_arg_val<uint32_t>(arg++);  // R3: this core's K shard (grid column)
    // Refinement 14: A as K segments (interleaved R1 / R1a only; a_segments.hpp). Segment 0 is a_addr; with
    // GM_A_SEGMENTS > 1 the segment widths / addresses follow here.
    const a_segments::Table a_tab = a_segments::read_table(a_addr, tensor_k_tiles, arg);
#if TAIL_SPLIT_MODE != 0
    // tail_writes_both_nocs: the writer's output region (same RT layout as the writer's first 12 args).
    constexpr auto out_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr uint32_t t_out_block_n = TAIL_OUT_BLOCK_N;
    constexpr uint32_t t_write_batch = TAIL_WRITE_BATCH;
    constexpr uint32_t t_o_tile_bytes = TAIL_O_TILE_BYTES;
    constexpr uint32_t t_batch_rows = t_write_batch * sb_h;
    constexpr uint32_t t_groups_per_batch = num_row_groups / t_write_batch;  // writer batches per block batch
    const uint32_t t_out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t_b0 = get_arg_val<uint32_t>(arg++);
    const uint32_t t_batches_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t t_row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t t_rows_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t t_col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t t_cols_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t t_nb_b = get_arg_val<uint32_t>(arg++);
    const uint32_t t_nb_m = get_arg_val<uint32_t>(arg++);
    const uint32_t t_nb_n = get_arg_val<uint32_t>(arg++);
    const uint32_t t_tensor_m_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t t_out_n_tiles = get_arg_val<uint32_t>(arg++);  // per output tensor (chunk)
    const out_chunks::Table t_out_tab = out_chunks::read_table(t_out_addr, t_out_n_tiles, arg);
    const uint32_t tail_mbox = get_semaphore(TAIL_SEM_MBOX);
    const uint32_t tail_post = get_semaphore(TAIL_SEM_POST);
    const uint32_t tail_done = get_semaphore(TAIL_SEM_DONE);
    // The writer's LAST block (the tail): its coordinates exactly as the writer computes them.
    const uint32_t t_blk_b0 = (t_nb_b - 1) * block_batches;
    const uint32_t t_blk_b_valid =
        (t_batches_valid - t_blk_b0) < block_batches ? (t_batches_valid - t_blk_b0) : block_batches;
    const uint32_t t_blk_r0 = (t_nb_m - 1) * block_m_tiles;
    const uint32_t t_blk_rows = block_extent(t_rows_valid, t_blk_r0, block_m_tiles);
    const uint32_t t_blk_c0 = (t_nb_n - 1) * t_out_block_n;
    const uint32_t t_blk_cols = block_extent(t_cols_valid, t_blk_c0, t_out_block_n);
    const out_chunks::Col t_blk_col = out_chunks::locate(t_out_tab, t_col0 + t_blk_c0 * GM_OUT_STRIDE);
    // The tail is the LAST block-batch of that block only (see the writer): its output follows the reader's final
    // K-block push, so serving it after the walk can never hold the writer while this reader still has reads.
    constexpr uint32_t t_bi = block_batches - 1;
    uint32_t tail_served = 0;  // writer batches of the tail this reader has served
    // Serve the reader's share of tail batch s (0-based, writer order: row-group batches of batch t_bi).
    auto serve_one = [&]() {
        const uint32_t s = tail_served;
        constexpr uint32_t bi = t_bi;
        const uint32_t g = s * t_write_batch;
        tail_split::wait_at_least(tail_post, s + 1);
        const uint32_t l1_base = tail_split::read(tail_mbox);
        if (bi < t_blk_b_valid) {
            const uint32_t batch = t_b0 + t_blk_b0 + bi;
            for (uint32_t r = 0; r < t_batch_rows; ++r) {
                const uint32_t blk_row = g * sb_h + r;
                if (blk_row >= t_blk_rows) {
                    break;
                }
                out_chunks::write_row(
                    out_args,
                    t_out_tab,
                    TAIL_O_PAGE_BYTES,
                    t_o_tile_bytes,
                    l1_base + r * t_out_block_n * t_o_tile_bytes,
                    batch * t_tensor_m_tiles + t_row0 + t_blk_r0 + blk_row,
                    t_blk_col,
                    t_blk_cols,
                    [&](uint32_t n) { return tail_split::reader_owns(n, t_blk_cols); },
                    GM_OUT_STRIDE);
            }
            noc_async_writes_flushed();  // the reader's share left L1
        }
        *tail_split::word(tail_done) = s + 1;
        ++tail_served;
    };
    // Serve every tail batch whose last row group (global over the block's batches) is < `rg_end`.
    [[maybe_unused]] auto serve_before = [&](uint32_t rg_end) {
        while (tail_served < t_groups_per_batch) {
            const uint32_t s = tail_served;
            const uint32_t last_rg = t_bi * num_row_groups + s * t_write_batch + t_write_batch - 1;
            if (last_rg >= rg_end) {
                break;
            }
            MaybeDeviceZoneScope("rd_tail_write");
            serve_one();
        }
    };
#endif

    const auto a_acc = TensorAccessor(a_args, a_addr, a_page_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, b_page_bytes);
    [[maybe_unused]] const auto row_acc = TensorAccessor(row_args, row_addr, row_page_bytes);
    [[maybe_unused]] const auto res_acc = TensorAccessor(res_args, res_addr, res_page_bytes);

    Noc noc;
    [[maybe_unused]] auto act_pipe = make_line_pipe < a_sends, operand_mcast && !r1a_split > (mc_act, noc);
    [[maybe_unused]] auto weight_pipe = make_line_pipe < b_sends, operand_mcast && !r1a_split > (mc_weight, noc);

    // A broadcast faces (std::nullopt where this core has no such role): R4's rotating A rounds (when
    // they run on the reader) and R3a's rotating row multicast.
    [[maybe_unused]] auto a_sender = mc_act.optional_sender(noc);
    [[maybe_unused]] auto a_receiver = mc_act.optional_receiver(noc);
    // R2a / R3a: the B group's injector (sender face) or a receiver of its multicast.
    [[maybe_unused]] auto b_sender = make_bs_face<bs_b_mcast, true>(mc_weight, noc);
    [[maybe_unused]] auto b_receiver = make_bs_face<bs_b_mcast, false>(mc_weight, noc);
    [[maybe_unused]] const bool b_injector = bs_has_face<bs_b_mcast>(b_sender);
    // Refinement 18 R3 split roles (GM_BS_ROLE): 0 = this kernel runs the whole R2 / R3 walk; 1 = only its A half
    // (A rounds, residual: NCRISC); 2 = only its B half (B rounds, the bias / gate row: BRISC / NoC1, the in-place
    // output needing no writer).
    constexpr uint32_t bs_role = GM_BS_ROLE;
    constexpr bool bs_does_a = bs_role != 2;
    constexpr bool bs_does_b = bs_role != 1;
    static_assert(bs_role <= 3 && (bs_role == 0 || block_sharded));  // (3: the R2a B-read helper)
    static_assert(bs_does_b || !bs_b_mcast);
    static_assert(bs_does_a || !bs_mcast);
    constexpr bool a_one_shot = GM_BS_A_ONE_SHOT != 0;
    static_assert(!a_one_shot || bs_mcast);
    constexpr bool row_tail = GM_BS_ROW_TAIL != 0;
    static_assert(!row_tail || (bs_b_mcast && has_row && b_chunks == 1));
    [[maybe_unused]] constexpr uint32_t row_bytes = block_n_tiles * row_tile_bytes;
    [[maybe_unused]] uint32_t row_tail_l1 = 0;  // the walk's first row in cb_weight_tiles' tail (GM_BS_ROW_TAIL)
    constexpr uint32_t act_block_bytes = act_block_tiles * a_tile_bytes;
    [[maybe_unused]] const genify_ds::ActKBlocks<block_m_tiles, block_k_tiles, k_blocks_per_round, a_tile_bytes> ds_act{
        a_addr};
    if constexpr (dram_sharded && !ds_worker) {
        // Non-worker cores of the broadcast run the reader only when the A rounds live here.
        genify_ds::send_only(ds_act, mc_act, a_sender, get_write_ptr(cb_act_tiles), act_cb_blocks, nb_m, num_k_blocks);
        return;
    }

    // A K-block: CB slot r*block_k + k  <-  page (a_row_base + r) * Kt + k0 + k (interleaved A under R1;
    // a peer's shard tiles under the R3 unicast stepping stone -- the accessor resolves the owning core).
    // `a_stride`: the slot's A row stride in tiles (block_k; the delivery's K width under GM_COMPACT_A).
    auto read_act =
        [&](uint32_t act_l1, uint32_t a_row_base, uint32_t k0, uint32_t blk_rows, uint32_t k_valid, uint32_t a_stride) {
            if constexpr (a_segments::num > 1) {
                // Refinement 14: the K-block's rows of concat(A_i) -- one run per segment it touches.
                const a_segments::Pos at = a_segments::locate(a_tab, k0);
                for (uint32_t r = 0; r < blk_rows; ++r) {
                    a_segments::read_row(
                        a_args,
                        a_tab,
                        a_page_bytes,
                        a_tile_bytes,
                        act_l1 + r * a_stride * a_tile_bytes,
                        a_row_base + r,
                        at,
                        k_valid);
                }
            } else {
                for (uint32_t r = 0; r < blk_rows; ++r) {
                    uint32_t page = (a_row_base + r) * tensor_k_tiles + k0;
                    uint32_t l1 = act_l1 + r * a_stride * a_tile_bytes;
                    for (uint32_t k = 0; k < k_valid; ++k) {
                        noc_async_read(a_acc.get_noc_addr(page + k), l1, a_tile_bytes);
                        l1 += a_tile_bytes;
                    }
                }
            }
        };
    // B K-block: CB slot k*block_n + c  <-  page b_base + (k0 + k) * Nt + col(c)
    // (col(c) = c, or the swiglu gate/up permutation).
    auto read_weight = [&](uint32_t weight_l1, uint32_t b_base, uint32_t k0, uint32_t k_valid, uint32_t blk_cols) {
#ifdef GM_ABLATE_B
        return;
#endif
        for (uint32_t k = 0; k < k_valid; ++k) {
            uint32_t page = b_base + (k0 + k) * tensor_n_tiles;
            uint32_t l1 = weight_l1 + k * block_n_tiles * b_tile_bytes;
            if constexpr (bank_runs) {
                // Bank-local stream: the row's units are one run of consecutive pages of one bank.
                if (blk_cols != 0) {
                    if constexpr (is_swiglu) {
                        const uint32_t run = (blk_cols / 2) * b_tile_bytes;
                        noc_async_read(b_acc.get_noc_addr(page), l1, run);
                        noc_async_read(b_acc.get_noc_addr(page + 1), l1 + half_n * b_tile_bytes, run);
                    } else {
                        noc_async_read(b_acc.get_noc_addr(page), l1, blk_cols * b_tile_bytes);
                    }
                }
            } else if constexpr (is_swiglu) {
                // blk_cols is even: pair p = (gate, up) at B columns (+p * unit_stride, +1) -> slots p, h + p.
                for (uint32_t p = 0; p < blk_cols / 2; ++p) {
                    noc_async_read(b_acc.get_noc_addr(page + unit_stride * p), l1 + p * b_tile_bytes, b_tile_bytes);
                    noc_async_read(
                        b_acc.get_noc_addr(page + unit_stride * p + 1), l1 + (half_n + p) * b_tile_bytes, b_tile_bytes);
                }
            } else {
                for (uint32_t n = 0; n < blk_cols; ++n) {
                    noc_async_read(b_acc.get_noc_addr(page + unit_stride * n), l1, b_tile_bytes);
                    l1 += b_tile_bytes;
                }
            }
        }
    };

    // R2a B-read helper (GM_BS_ROLE 3, BRISC of a one-shot walk's injector): the second half of the walk's only B
    // round -- its single block's K rows [ceil(K / 2), K) into the same slot -- then the post.
    constexpr bool b_helper = GM_BS_B_HELPER_SEM >= 0;
    [[maybe_unused]] const uint32_t b_split_row = tensor_k_tiles - tensor_k_tiles / 2;
    static_assert(bs_role != 3 || (b_helper && block_sharded && !bs_b_mcast && num_k_blocks == 1));
    if constexpr (bs_role == 3) {
        MaybeDeviceZoneScope("rd_b_helper");
        read_weight(
            get_write_ptr(cb_weight_tiles) + b_split_row * block_n_tiles * b_tile_bytes,
            col0,
            b_split_row,
            tensor_k_tiles - b_split_row,
            cols_valid < block_n_tiles ? cols_valid : block_n_tiles);
        noc_async_read_barrier();
        *tail_split::word(get_semaphore(GM_BS_B_HELPER_SEM)) = 1;
        return;
    }

    // R4 B stream (Refinement 6): a ring of `ds_b_ahead` K-blocks in flight, each tagged with its own
    // transaction id, so retiring K-block s waits only its own reads while s+1 .. s+T-1 stay on the wire
    // (a plain read barrier would drain the bank pipe once per K-block). B does not depend on the block
    // row, so the stream runs over the linear K-block sequence s = mb * num_k_blocks + kb. The CB write
    // pointer sits at K-block s's slot until it is pushed, so K-block j >= s lands (j - s) slots past it:
    // reserve (j - s + 1) blocks, address base + (j % depth) * block bytes (the CB is exactly `depth`
    // blocks). The ring needs no counters: K-block j's id (j % T) + 1 was freed by retiring j - T.
    [[maybe_unused]] uint32_t ds_seq = 0;
    [[maybe_unused]] const uint32_t ds_total = nb_m * num_k_blocks;
    constexpr uint32_t weight_block_bytes = weight_block_tiles * b_tile_bytes;
    [[maybe_unused]] const uint32_t weight_base = get_write_ptr(cb_weight_tiles);
    // R1a prefetch: the operand rings' bases (the CBs are exactly r1a_depth K-blocks) and the K-blocks pushed so
    // far (every R1a K-block pushes both operands, so one sequence addresses both rings).
    [[maybe_unused]] const uint32_t act_base = get_write_ptr(cb_act_tiles);
    [[maybe_unused]] uint32_t r1a_seq = 0;
    [[maybe_unused]] auto ds_issue_weight = [&](uint32_t j) {
        cb_reserve_back(cb_weight_tiles, (j - ds_seq + 1) * weight_block_tiles);
        const uint32_t kbj = j % num_k_blocks;
        const uint32_t k0j = kbj * block_k_tiles;
        const uint32_t kvj = (kbj == num_k_blocks - 1) ? last_block_k_tiles : block_k_tiles;
        const uint32_t l1 = weight_base + (j % ds_weight_depth) * weight_block_bytes;
        dataflow_kernel_lib::set_read_trid(noc, j % ds_b_ahead + 1);
#ifdef GM_ABLATE_B
        return;
#endif
        if constexpr (ds_row_reads) {
            // cores_per_bank split: this worker's slice of each K row is one run (the rows are not
            // adjacent) of its valid columns only (a bank's narrower last slice, the last bank's padding:
            // the slots past them hold stale tiles whose output columns are never written). A slice wholly
            // in the last bank's padding reads nothing (its page ids would alias the next row).
            if (cols_valid != 0) {
                for (uint32_t k = 0; k < kvj; ++k) {
                    noc_async_read(
                        b_acc.get_noc_addr((k0j + k) * tensor_n_tiles + col0),
                        l1 + k * block_n_tiles * b_tile_bytes,
                        cols_valid * b_tile_bytes);
                }
            }
        } else {
            // One worker per bank: rows [k0, k0 + k_valid) of the bank's column shard are one run.
            noc_async_read(b_acc.get_noc_addr(k0j * tensor_n_tiles + col0), l1, kvj * block_n_tiles * b_tile_bytes);
        }
    };

    // R1a split lines: this core's B column line (rotating injectors). Round r = (block, K-block) of the walk.
    static_assert(
        !r1a_split ||
        (block_batches == 1 && !weight_resident && !act_resident && !r1a_prefetch && GM_BANK_STRIDE == 0));
    [[maybe_unused]] auto b_line_sender = make_bs_face<r1a_split, true>(mc_weight, noc);
    [[maybe_unused]] auto b_line_receiver = make_bs_face<r1a_split, false>(mc_weight, noc);
    constexpr uint32_t b_line_span = r1a_split ? mc_weight.num_senders : 1;
    // Round r = delivery r % KS::n of block r / KS::n (the head ramp's deliveries under GM_K_RAMP).
    [[maybe_unused]] auto b_line_read = [&](uint32_t r, uint32_t l1, uint32_t k_lo, uint32_t k_hi) {
        const uint32_t d = r % KS::n;
        const uint32_t kv = KS::kv(d);
        const uint32_t c0 = ((r / KS::n) % nb_n) * block_n_tiles;
        if (k_lo < kv) {
            read_weight(
                l1 + k_lo * block_n_tiles * b_tile_bytes,
                col0 + (c0 / unit_cols) * unit_stride,
                KS::k0(d) + k_lo,
                (k_hi < kv ? k_hi : kv) - k_lo,
                block_extent(cols_valid, c0, block_n_tiles));
        }
    };
    using BLine = gm_line::Stream<
        cb_weight_tiles,
        weight_block_tiles,
        weight_block_bytes,
        GM_R1A_DEPTH,
        b_line_span,
        1,
        block_k_tiles,
        GM_LINE_WINDOW,
        GM_LINE_STARTUP>;
    [[maybe_unused]] BLine b_line(
        r1a_split ? nb_m * nb_n * KS::n : 0,
        r1a_split ? gm_line::first_turn<b_line_span>(mc_weight) : b_line_span,
        r1a_split ? get_write_ptr(GM_LINE_STAGE_CB) : 0);

    for (uint32_t bb = 0; bb < nb_b; ++bb) {
        const uint32_t blk_b0 = bb * block_batches;
        const uint32_t blk_b_valid =
            (batches_valid - blk_b0) < block_batches ? (batches_valid - blk_b0) : block_batches;
        for (uint32_t mb = 0; mb < nb_m; ++mb) {
            const uint32_t blk_r0 = mb * block_m_tiles;
            const uint32_t blk_rows = block_extent(rows_valid, blk_r0, block_m_tiles);
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * block_n_tiles;
                const uint32_t blk_cols = block_extent(cols_valid, blk_c0, block_n_tiles);
                // The helper's batch loop consumes block_batches whole K-walks in batch order.
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    const uint32_t a_row_base = batch * tensor_m_tiles + row0 + blk_r0;
                    const uint32_t b_base = batch * weight_batch_stride + col0 + (blk_c0 / unit_cols) * unit_stride;
                    // R2a / R3a (Refinement 10): the walk's first bias / gate row is the same on every core of the
                    // B group too (a DRAM hotspot when each core reads it: ~2.5 us bank-serialized on PDL's 86-103
                    // core HS sets), so the injector reads it beside B (own transaction id) and multicasts it in
                    // one row round right after the first B round, while compute runs the matmul. It is pushed
                    // after the walk as usual. Later blocks (cb_row holds one row) keep the per-core read.
                    const bool row_mcast = bs_b_mcast && has_row && bb == 0 && mb == 0 && nbk == 0 && bi == 0;
                    // Refinement 18: the split roles' B half reading its own B also issues the block's row with
                    // its first K-block (it is off the A chain, so the row no longer trails the walk).
                    const bool row_early = bs_role == 2 && !bs_b_mcast && has_row;
                    constexpr uint32_t row_trid = 15;
                    static_assert(!bs_b_mcast || b_chunks < row_trid);
                    auto read_row = [&](uint32_t l1) {
                        const uint32_t page = col0 + blk_c0 * unit_stride;  // has_row: unit = one column
                        for (uint32_t n = 0; n < blk_cols; ++n) {
                            noc_async_read(row_acc.get_noc_addr(page + n * unit_stride), l1, row_tile_bytes);
                            l1 += row_tile_bytes;
                        }
                    };
                    auto issue_row = [&]() {
                        cb_reserve_back(cb_row, block_n_tiles);
                        read_row(get_write_ptr(cb_row));
                    };
                    for (uint32_t kb = 0; kb < KS::n; ++kb) {
                        const uint32_t k_valid = KS::kv(kb);
                        const uint32_t k0 = KS::k0(kb);

                        if constexpr (dram_sharded) {
                            MaybeDeviceZoneScope("rd_ds_kblock");
                            // B (stall-shadow: issued before the A round it does not depend on), kept
                            // ds_b_ahead K-blocks ahead (the ring above; nb_b = nb_n = block_batches = 1).
                            if (ds_seq == 0) {
                                for (uint32_t j = 0; j + 1 < ds_b_ahead && j < ds_total; ++j) {
                                    ds_issue_weight(j);
                                }
                            }
                            if (ds_seq + ds_b_ahead - 1 < ds_total) {
                                ds_issue_weight(ds_seq + ds_b_ahead - 1);
                            }
                            // A round (here, or on the writer: DS_ACT_ON_WRITER): the owner of A shard
                            // `round` multicasts this K-block to every worker's reserved slot.
                            if constexpr (ds_reader_act) {
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                genify_ds::worker_round(
                                    ds_act, mc_act, a_sender, a_receiver, mb, kb, get_write_ptr(cb_act_tiles));
                            }
                            dataflow_kernel_lib::async_read_barrier_with_trid(noc, ds_seq % ds_b_ahead + 1);
                            if constexpr (ds_reader_act) {
                                cb_push_back(cb_act_tiles, act_block_tiles);
                            }
                            cb_push_back(cb_weight_tiles, weight_block_tiles);
                            ++ds_seq;
                            if (ds_seq == ds_total) {
                                dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged reads
                            }
                        } else if constexpr (block_sharded) {
                            MaybeDeviceZoneScope("rd_bs_kblock");
                            // A B round starts here (always without the multicast: one K-block per round). Its
                            // b_round_kblocks K-block slots are reserved together; they are contiguous K rows
                            // (the CB holds a whole number of rounds and rounds start on a multiple of it).
                            // Refinement 18 (GM_BS_ROLE): the B half (cb_weight_tiles, cb_row) and the A half
                            // (cb_act_shard / cb_act_tiles, cb_residual_tiles) may run on two RISCs.
                            const bool b_round = bs_does_b && (kb % b_round_kblocks) == 0;
                            if (b_round) {
                                cb_reserve_back(cb_weight_tiles, b_round_kblocks * weight_block_tiles);
                            }
                            const uint32_t weight_l1 = get_write_ptr(cb_weight_tiles);
                            constexpr uint32_t chunk_row_bytes = block_n_tiles * b_tile_bytes;
                            constexpr uint32_t round_rows = b_round_kblocks * block_k_tiles;
                            if (row_tail && row_mcast && kb == 0) {
                                row_tail_l1 = weight_l1 + round_rows * chunk_row_bytes;
                            }
                            if constexpr (bs_b_mcast) {
                                // R2a / R3a: only the injector reads B, chunk c tagged with trid c + 1 so each
                                // chunk goes on the wire as soon as its own reads land (stall-shadow: issued
                                // before the A round, which does not depend on B).
                                if (b_injector && b_round) {
                                    MaybeDeviceZoneScope("rd_b_issue");
                                    const uint32_t round_valid =
                                        (tensor_k_tiles - k0) < round_rows ? (tensor_k_tiles - k0) : round_rows;
                                    for (uint32_t c = 0; c < b_chunks; ++c) {
                                        const uint32_t r0 = c * b_chunk_k;
                                        if (r0 >= round_valid) {
                                            break;
                                        }
                                        uint32_t rows = (round_valid - r0) < b_chunk_k ? (round_valid - r0) : b_chunk_k;
                                        if constexpr (b_helper) {
                                            rows = b_split_row;  // (one chunk, one round: the helper reads the rest)
                                        }
                                        dataflow_kernel_lib::set_read_trid(noc, c + 1);
                                        read_weight(weight_l1 + r0 * chunk_row_bytes, b_base, k0 + r0, rows, blk_cols);
                                    }
                                    if (row_mcast && kb == 0) {
                                        if constexpr (row_tail) {
                                            // Refinement 18: into the tail past the round's slot, under the last
                                            // chunk's trid (the one delivery carries both).
                                            read_row(weight_l1 + round_rows * chunk_row_bytes);
                                        } else {
                                            dataflow_kernel_lib::set_read_trid(noc, row_trid);
                                            issue_row();  // after B: the row round follows the B round
                                        }
                                    }
                                    dataflow_kernel_lib::set_read_trid(noc, 0);  // the A round's reads stay untagged
                                }
                            } else if (bs_does_b && batch_valid) {  // padded batch slots (batched B) read no weight
                                read_weight(weight_l1, b_base, k0, k_valid, blk_cols);
                                if (row_early && kb == 0 && bi == 0) {
                                    issue_row();  // retired by this K-block's barrier, pushed after the walk
                                }
                            }
                            const bool own = bs_does_a && kb == own_kb;
                            if (own) {
                                // The block rows of this core's own shard are resident: hand them over.
                                cb_reserve_back(cb_act_shard, act_block_tiles);
                            }
                            if constexpr (bs_mcast && a_one_shot) {
                                if (kb == 0) {
                                    // Refinement 18: the whole exchange up front (the walk's only block). This
                                    // core's shard (the block) lands in slot own_kb of every row peer.
                                    MaybeDeviceZoneScope("rd_a_exchange");
                                    cb_reserve_back(cb_act_tiles, num_k_blocks * act_block_tiles);
                                    a_sender->send(
                                        get_write_ptr(cb_act_shard),
                                        get_write_ptr(cb_act_tiles) + own_kb * act_block_bytes,
                                        act_block_bytes);
                                    a_receiver->receive(num_k_blocks - 1);  // count >= num_k_blocks: all landed
                                }
                            } else if constexpr (bs_mcast) {
                                // Every core reserves round kb's slot (the sender too: lockstep pointers).
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                if (own) {
                                    a_sender->send(
                                        get_write_ptr(cb_act_shard), get_write_ptr(cb_act_tiles), act_block_bytes);
                                } else {
                                    a_receiver->receive(kb);
                                }
                            } else if (bs_does_a && !own) {
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                read_act(get_write_ptr(cb_act_tiles), a_row_base, k0, blk_rows, k_valid, block_k_tiles);
                            }
                            const bool a_pending = bs_does_a && !bs_mcast && !own;  // unicast A reads in flight
                            auto push_act = [&]() {
                                if (own) {
                                    cb_push_back(cb_act_shard, act_block_tiles);
                                }
                                if (bs_mcast || (bs_does_a && !own)) {
                                    cb_push_back(cb_act_tiles, act_block_tiles);
                                }
                            };
                            // Refinement 18: the injector hands its K-block to compute as soon as it has landed
                            // (one-chunk round; its A is resident, already exchanged, or another RISC's) and
                            // multicasts after -- the send only reads the slot, so its own matmul overlaps the
                            // group's delivery instead of trailing it (the injector was the op's last core).
                            const bool early_push = bs_b_mcast && b_injector && b_round && b_chunks == 1 && !a_pending;
                            if (bs_b_mcast && b_round) {
                                MaybeDeviceZoneScope("rd_b_round");
                                // B round after the A round on every core (fixed order: no channel deadlock).
                                for (uint32_t c = 0; c < b_chunks; ++c) {
                                    const uint32_t r0 = c * b_chunk_k;
                                    const uint32_t rows = (round_rows - r0) < b_chunk_k ? (round_rows - r0) : b_chunk_k;
                                    if (b_injector) {
                                        dataflow_kernel_lib::async_read_barrier_with_trid(noc, c + 1);
                                        if constexpr (b_helper) {
                                            tail_split::wait_at_least(get_semaphore(GM_BS_B_HELPER_SEM), 1);
                                        }
                                    }
                                    if (early_push) {
                                        push_act();
                                        cb_push_back(cb_weight_tiles, weight_block_tiles);
                                    }
                                    bs_b_deliver<bs_b_mcast>(
                                        b_injector,
                                        b_sender,
                                        b_receiver,
                                        weight_l1 + r0 * chunk_row_bytes,
                                        rows * chunk_row_bytes + (row_tail && row_mcast ? row_bytes : 0));
                                }
                            }
                            // One barrier per K-block, both operands. Under the B multicast B's reads are retired
                            // by their transaction ids, so only the unicast A stepping stone still needs it.
                            if ((bs_does_b && !bs_b_mcast) || a_pending) {
                                noc_async_read_barrier();
                            }
                            if (!early_push) {
                                push_act();
                                if constexpr (bs_does_b) {
                                    cb_push_back(cb_weight_tiles, weight_block_tiles);
                                }
                            }
                            if (!row_tail && row_mcast && kb == 0) {
                                // Row round (same channel, same order on every core): lands at the injector's
                                // cb_row slot address on every receiver (lockstep pointers).
                                MaybeDeviceZoneScope("rd_row_round");
                                if (b_injector) {
                                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, row_trid);
                                } else {
                                    cb_reserve_back(cb_row, block_n_tiles);
                                }
                                bs_b_deliver<bs_b_mcast>(
                                    b_injector,
                                    b_sender,
                                    b_receiver,
                                    get_write_ptr(cb_row),
                                    block_n_tiles * row_tile_bytes);
                            }
                        } else if constexpr (r1a_split) {
                            // R1a split lines: B round of this K-block (A arrives on the writer's row line).
                            // B's valid rows of a short (head ramp) delivery are a prefix of its slot.
                            gm_line::round_if<r1a_split>(
                                b_line,
                                noc,
                                b_line_sender,
                                b_line_receiver,
                                b_line_read,
                                (gm_ks::ramp ? k_valid * block_n_tiles : weight_block_tiles) * b_tile_bytes);
                        } else {
                            // B residency: the core's one B block (one K-block, its whole column range)
                            // is read on the first M-block only; compute keeps it fronted for the rest.
                            const bool load_weight = !weight_resident || mb == 0;
                            // A residency (the twin): the M-block's A is read on its first N-block only.
                            // R1b: cb_act_tiles has its own producer (the writer's A stream).
                            const bool load_act = !act_on_writer && (!act_resident || nbk == 0);
                            if constexpr (r1a_prefetch) {
                                // Injector stall-shadow: K-block j of this walk lands in ring slot r1a_seq + j - kb;
                                // its reads carry trid (seq % 2) + 1. kb's reads were issued one K-block ago (at
                                // kb = 0: now), kb+1's go out now, before kb is retired and multicast.
                                auto issue = [&](uint32_t j) {
                                    const uint32_t seq = r1a_seq + j - kb;
                                    dataflow_kernel_lib::set_read_trid(noc, seq % 2 + 1);
                                    if (batch_valid && a_sends) {
                                        read_act(
                                            act_base + (seq % r1a_depth) * act_block_bytes,
                                            a_row_base,
                                            KS::k0(j),
                                            blk_rows,
                                            KS::kv(j),
                                            block_k_tiles);
                                    }
                                    if (batch_valid && b_sends) {
                                        read_weight(
                                            weight_base + (seq % r1a_depth) * weight_block_bytes,
                                            b_base,
                                            KS::k0(j),
                                            KS::kv(j),
                                            blk_cols);
                                    }
                                    dataflow_kernel_lib::set_read_trid(noc, 0);
                                };
                                const bool ahead = kb + 1 < KS::n;
                                MaybeDeviceZoneScope("rd_reserve");
                                cb_reserve_back(cb_act_tiles, (a_sends && ahead ? 2 : 1) * act_block_tiles);
                                cb_reserve_back(cb_weight_tiles, (b_sends && ahead ? 2 : 1) * weight_block_tiles);
                                if constexpr (a_sends || b_sends) {
                                    if (kb == 0) {
                                        issue(0);
                                    }
                                    if (ahead) {
                                        issue(kb + 1);
                                    }
                                }
                            } else {
                                MaybeDeviceZoneScope("rd_reserve");
                                if (load_act) {
                                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                                }
                                if (load_weight) {
                                    cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                                }
                            }
                            const uint32_t act_l1 = get_write_ptr(cb_act_tiles);
                            const uint32_t weight_l1 = get_write_ptr(cb_weight_tiles);
                            if constexpr (r1a_prefetch) {
                                if constexpr (a_sends || b_sends) {
                                    MaybeDeviceZoneScope("rd_barrier");
                                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, r1a_seq % 2 + 1);
                                }
                            }
#ifndef GM_ABLATE_DM
                            else if constexpr (reads_act || reads_weight) {
                                {
                                    MaybeDeviceZoneScope("rd_issue");
                                    if (batch_valid && reads_act && load_act) {
                                        read_act(
                                            act_l1,
                                            a_row_base,
                                            k0,
                                            blk_rows,
                                            k_valid,
                                            gm_ks::compact_a ? k_valid : block_k_tiles);
                                    }
                                    if (batch_valid && reads_weight && load_weight) {
                                        read_weight(weight_l1, b_base, k0, k_valid, blk_cols);
                                    }
                                }
                                MaybeDeviceZoneScope("rd_barrier");
                                noc_async_read_barrier();  // one barrier per K-block, both operands
                            }
#endif
                            // R1a: A over its line, then B over its line (fixed order on every core).
                            // GM_ABLATE_DM (perf tooling): the operand reads and line deliveries are stubbed
                            // (reserve / push only), so the walk times compute + the output stream alone.
                            if constexpr (operand_mcast && !gm_ablate_dm) {
                                if (load_act) {
                                    MaybeDeviceZoneScope("rd_mcast_a");
                                    // Head ramp: only the delivery's valid tiles go on the wire (A compact
                                    // under GM_COMPACT_A; B's valid rows are always a prefix of its slot).
                                    line_deliver<operand_mcast, a_sends>(
                                        act_pipe,
                                        act_l1,
                                        (gm_ks::compact_a ? block_m_tiles * k_valid : act_block_tiles) * a_tile_bytes);
                                }
                                if (load_weight) {
                                    MaybeDeviceZoneScope("rd_mcast_b");
                                    line_deliver<operand_mcast, b_sends>(
                                        weight_pipe,
                                        weight_l1,
                                        (gm_ks::ramp ? k_valid * block_n_tiles : weight_block_tiles) * b_tile_bytes);
                                }
                            }
                            if (load_act) {
                                cb_push_back(cb_act_tiles, act_block_tiles);
                            }
                            if (load_weight) {
                                cb_push_back(cb_weight_tiles, weight_block_tiles);
                            }
                            ++r1a_seq;
                        }
                    }

                    if constexpr (has_row && bs_does_b) {
                        // bias / gate row of this N-block: once per block, after the first K-walk.
                        if (bi == 0) {
                            MaybeDeviceZoneScope("rd_row");
                            if (row_early) {
                                // (issued with the walk's first K-block)
                            } else if (!row_mcast) {  // (the multicast row already landed in the row round)
                                issue_row();
                                noc_async_read_barrier();
                            } else if constexpr (row_tail) {
                                // Refinement 18: the row landed in cb_weight_tiles' tail with B (one delivery);
                                // copy it into cb_row (local L1 -> L1). The tail is past the walk's only round,
                                // so nothing reuses it.
                                cb_reserve_back(cb_row, block_n_tiles);
                                noc_async_read(
                                    get_noc_addr(row_tail_l1), get_write_ptr(cb_row), block_n_tiles * row_tile_bytes);
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_row, block_n_tiles);
                        }
                    }
                    if constexpr (has_residual && bs_does_a) {
                        // residual rows of this batch, in row groups of sb_h (epilogue order).
                        for (uint32_t g = 0; g < num_row_groups; ++g) {
                            MaybeDeviceZoneScope("rd_residual");
                            cb_reserve_back(cb_residual_tiles, residual_group_tiles);
                            if (batch_valid) {
                                const uint32_t res_l1 = get_write_ptr(cb_residual_tiles);
                                for (uint32_t r = 0; r < sb_h; ++r) {
                                    const uint32_t blk_row = g * sb_h + r;
                                    if (blk_row >= blk_rows) {
                                        break;
                                    }
                                    const uint32_t page =
                                        (a_row_base + blk_row) * tensor_n_tiles + col0 + blk_c0 * unit_stride;
                                    uint32_t l1 = res_l1 + r * block_n_tiles * res_tile_bytes;
                                    for (uint32_t n = 0; n < blk_cols; ++n) {
                                        noc_async_read(
                                            res_acc.get_noc_addr(page + n * unit_stride), l1, res_tile_bytes);
                                        l1 += res_tile_bytes;
                                    }
                                }
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_residual_tiles, residual_group_tiles);
#if TAIL_SPLIT_MODE != 0
                            // Tail block: serve the output batches compute can already finish (rows < g) so the
                            // residual ring (refilled only as compute advances) never deadlocks on the writer.
                            if (bb + 1 == nb_b && mb + 1 == nb_m && nbk + 1 == nb_n) {
                                serve_before(bi * num_row_groups + g);
                            }
#endif
                        }
                    }
                }
            }
        }
    }
#if TAIL_SPLIT_MODE != 0
    {
        MaybeDeviceZoneScope("rd_tail_write");
        serve_before(0xFFFFFFFFu);  // the rest of the tail
        noc_async_write_barrier();
    }
#endif
}
