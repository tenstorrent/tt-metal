// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, second data-movement RISC of one core of one expert group (group_cores cores). Per job (one expert, at
// most 32 * row_tiles of its routed rows, from the routing table; job = its index in this group's list):
// - x: the job's x moves in chunks of chunk_tiles hidden tiles (one chunk when chunk_tiles covers them all). Every
//   group core owns an equal piece of each chunk (so all of them send at once): it reads that slice of each of the
//   job's token rows from the row-major input, rearranges it into tile faces ([hidden tile][row tile]: the job's row
//   `row` is row row % 32 of row tile row / 32) and writes it into slot (chunk % x_slots) of every group core's cb_x,
//   then increments that slot's semaphore on every group core.
// - intermediate exchange: this core's activated tiles (its real intermediate columns of every row tile) to slot
//   (job % intermediate_slots) of every group core's cb_intermediate, one semaphore per slot.
// - output rows: per W2 output group and row tile, the compute leaves 4 bf16 tiles in cb_rows; the job's real rows are
//   copied out of the tile faces and written as posted writes to output row (first row of the job + row), columns of
//   the group's real output tiles.
// Slot reuse. With one x chunk per job and intermediate_slots >= 3 (no credits) the exchange orders it: once a core
// holds every group core's activated tiles of job j, every core finished the gate/up projection of jobs j (x slot of
// job j free) and j - 1, hence the down projection of job j - 3. With several chunks per job a slot is reused only
// after every group core handed it back (a credit after its compute popped the slot's previous chunk), and the
// intermediate slot the same way; a slot's first use waits for nothing.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "expert_rows_common.h"

namespace {

constexpr uint32_t group_cores = get_named_compile_time_arg_val("group_cores");
constexpr uint32_t intermediate_tiles = get_named_compile_time_arg_val("intermediate_tiles");
constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
constexpr uint32_t intermediate_slots = get_named_compile_time_arg_val("intermediate_slots");
constexpr uint32_t x_slots = get_named_compile_time_arg_val("x_slots");
constexpr uint32_t activated_tiles = get_named_compile_time_arg_val("activated_tiles");  // per row tile
constexpr uint32_t routing_ctl = get_named_compile_time_arg_val("routing_ctl");
constexpr uint32_t routing_row_tokens = get_named_compile_time_arg_val("routing_row_tokens");
constexpr uint32_t routing_jobs = get_named_compile_time_arg_val("routing_jobs");
constexpr uint32_t grid_w = get_named_compile_time_arg_val("grid_w");
constexpr uint32_t grid_h = get_named_compile_time_arg_val("grid_h");
constexpr uint32_t mask_words = get_named_compile_time_arg_val("mask_words");
constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");  // 2 hidden_size
constexpr uint32_t expert_groups = get_named_compile_time_arg_val("expert_groups");
constexpr uint32_t sem_intermediate = get_named_compile_time_arg_val("sem_intermediate");  // intermediate_slots
constexpr uint32_t sem_x = get_named_compile_time_arg_val("sem_x");                        // x_slots
constexpr uint32_t sem_intermediate_free = get_named_compile_time_arg_val("sem_intermediate_free");  // with credits
constexpr uint32_t sem_x_free = get_named_compile_time_arg_val("sem_x_free");                        // with credits
constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
constexpr uint32_t chunk_piece_max = get_named_compile_time_arg_val("chunk_piece_max");
constexpr bool slot_credits = get_named_compile_time_arg_val("slot_credits") == 1;
static_assert(x_slots == 1 || x_slots == 2, "one or two x slots");
static_assert(slot_credits || intermediate_slots >= 3, "intermediate slots without credits");
static_assert(slot_credits || chunk_tiles == hidden_tiles, "x slots without credits: one chunk per job");
// bf16 tile geometry (expert_rows_common.h): a face row, a tile row (two faces side by side), a face, a tile
constexpr uint32_t tile_rows = expert_rows::tile_rows;
constexpr uint32_t face_row_bytes = expert_rows::face_columns * expert_rows::bf16_bytes;
constexpr uint32_t row_faces = tile_rows / expert_rows::face_columns;
constexpr uint32_t tile_row_bytes = row_faces * face_row_bytes;
constexpr uint32_t face_bytes = expert_rows::face_rows * face_row_bytes;
constexpr uint32_t tile_bytes = tile_rows * tile_row_bytes;
constexpr uint32_t group_tiles = expert_rows::group_tiles;            // tiles of an output group
constexpr uint32_t output_row_stride = group_tiles * tile_row_bytes;  // staging bytes of one row of one output group
constexpr uint32_t piece_row_stride = chunk_piece_max * tile_row_bytes;
// a job of several row tiles takes x in chunks of chunk_tiles hidden tiles, a job of one row tile in chunks as large as
// an x slot of row_tiles row tiles holds (rows_compute.cpp)
constexpr uint32_t single_chunk_tiles = expert_rows::single_chunk_tiles(chunk_tiles, row_tiles, hidden_tiles);
constexpr uint32_t virtual_x_arg = 0, virtual_y_arg = grid_w;  // common args: virtual x / y of the grid

// Byte offset of row `row` (0..31) of a bf16 tile in its left face; the row's right half is face_bytes further.
FORCE_INLINE uint32_t tile_row_offset(uint32_t row) {
    return ((row / expert_rows::face_rows) * row_faces) * face_bytes + (row % expert_rows::face_rows) * face_row_bytes;
}

// The cores of this core's expert group: the bits of the group mask (runtime args from `mask_arg`; bit i is core
// (i / grid_h, i % grid_h) of the grid), with their virtual coordinates from the common args.
struct GroupCoreIterator {
    uint32_t mask_arg;
    uint32_t word = 0, bits = 0;
    uint32_t x = 0, y = 0;
    explicit GroupCoreIterator(uint32_t mask_arg) : mask_arg(mask_arg) {}
    FORCE_INLINE bool next() {
        while (bits == 0) {
            if (word == mask_words) {
                return false;
            }
            bits = get_arg_val<uint32_t>(mask_arg + word++);
        }
        const uint32_t core = (word - 1) * 32 + __builtin_ctz(bits);
        bits &= bits - 1;
        x = get_common_arg_val<uint32_t>(virtual_x_arg + core / grid_h);
        y = get_common_arg_val<uint32_t>(virtual_y_arg + core % grid_h);
        return true;
    }
};

FORCE_INLINE void increment_group(const Noc& noc, uint32_t semaphore_id, uint32_t group_mask_arg) {
    Semaphore<> semaphore(semaphore_id);
    for (GroupCoreIterator core(group_mask_arg); core.next();) {
        semaphore.up(noc, core.x, core.y, 1);
    }
}

// This group's jobs in the routing table (rows_reader.cpp): job j is table job group + expert_groups j with one row
// tile per job, else the group's j-th as the root dealt them (from the group's first job in the ctl block).
using GroupJobs = expert_rows::GroupJobs<(row_tiles > 1), expert_groups, volatile tt_l1_ptr expert_rows::JobRecord*>;

FORCE_INLINE uint32_t x_chunk_tiles(const GroupJobs& jobs, uint32_t job) {
    return jobs.rows(job) > tile_rows ? chunk_tiles : single_chunk_tiles;
}

FORCE_INLINE uint32_t x_chunks(const GroupJobs& jobs, uint32_t job) {
    const uint32_t tiles = x_chunk_tiles(jobs, job);
    return expert_rows::div_up(hidden_tiles, tiles);
}

// This core's piece of x chunk `chunk` of a job: hidden tiles [first, first + tiles) of that chunk
struct XPiece {
    uint32_t first, tiles;
};

FORCE_INLINE XPiece x_piece(const GroupJobs& jobs, uint32_t job, uint32_t chunk, uint32_t member) {
    const uint32_t job_chunk_tiles = x_chunk_tiles(jobs, job);
    const uint32_t tiles = chunk + 1 < x_chunks(jobs, job) ? job_chunk_tiles : hidden_tiles - chunk * job_chunk_tiles;
    const uint32_t first = member * tiles / group_cores;
    return {first, (member + 1) * tiles / group_cores - first};
}

// The schedule calls the steps below from several places; they stay out of line (kernel size) and build their own
// Noc: a Noc passed to an out-of-line function carries its id as a run-time value, which costs instructions in every
// NoC command.

// The x stream of this core: its piece of each x chunk, built from the job's token rows, to every group core. The piece
// is tilized on this RISC, not by the compute: each core builds only its piece, once, for every group core, while the
// computes run on the previous chunk.
template <typename XAccessor>
struct XStream {
    CircularBuffer cb_x;
    XAccessor x_tensor;
    volatile tt_l1_ptr uint16_t* row_tokens;  // token of each routed row
    uint32_t member;                          // index of this core in its group
    uint32_t staging;                         // tile_rows row_tiles rows of piece_row_stride
    uint32_t piece_tiles_addr;                // [piece tile][row tile] bf16 tiles
    uint32_t x_base;                          // cb_x, same L1 address on every group core
    uint32_t group_mask_arg;

    // The piece of chunk `chunk` for every row of a job, as tiles [piece tile][row tile] at piece_tiles_addr.
    __attribute__((noinline)) void build_piece(const GroupJobs& jobs, uint32_t job, uint32_t chunk) const {
        const Noc noc;
        const uint32_t first_row = jobs.first_row(job), rows = jobs.rows(job);
        const uint32_t job_row_tiles = expert_rows::row_tiles_of(rows);
        const XPiece piece = x_piece(jobs, job, chunk, member);
        if (piece.tiles == 0) {
            return;
        }
        // staging row stride = this piece (a one-row-tile job's piece is up to row_tiles times a held job's)
        const uint32_t first_tile = chunk * x_chunk_tiles(jobs, job) + piece.first;
        const uint32_t stride = piece.tiles * tile_row_bytes;
        for (uint32_t row = 0; row < rows; ++row) {
            noc.async_read(
                x_tensor,
                CoreLocalMem<uint32_t>(staging + row * stride),
                piece.tiles * tile_row_bytes,
                {.page_id = row_tokens[first_row + row], .offset_bytes = first_tile * tile_row_bytes},
                {});
        }
        noc.async_read_barrier();
        // this core's own L1 as the read source: no Device 2.0 endpoint takes it (CoreLocalMem is a local source only)
        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            PrecomposedUnicastEndpoint{}, face_row_bytes, {.noc_addr = get_noc_addr(staging)});
        for (uint32_t piece_tile = 0; piece_tile < piece.tiles; ++piece_tile) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t src = staging + row * stride + piece_tile * tile_row_bytes;
                const uint32_t dst = piece_tiles_addr + (piece_tile * job_row_tiles + row / tile_rows) * tile_bytes +
                                     tile_row_offset(row % tile_rows);
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{}, CoreLocalMem<uint32_t>(dst), face_row_bytes, {.addr = src}, {});
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(dst + face_bytes),
                    face_row_bytes,
                    {.addr = src + face_row_bytes},
                    {});
            }
        }
        noc.async_read_barrier();
    }

    // The built piece of chunk `chunk` of a job, chunk number seq of this core's stream, to slot seq % x_slots of every
    // group core's cb_x; then that slot's semaphore on every group core.
    __attribute__((noinline)) void send_piece(const GroupJobs& jobs, uint32_t job, uint32_t chunk, uint32_t seq) {
        const Noc noc;
        const uint32_t rows = jobs.rows(job);
        const uint32_t job_row_tiles = expert_rows::row_tiles_of(rows);
        const uint32_t slot = seq % x_slots, use = seq / x_slots;
        if constexpr (slot_credits) {
            cb_x.reserve_back(chunk_tiles * row_tiles);  // this core's compute popped the slot's previous chunk
            if (use > 0) {  // a slot starts free on every group core: only reuses wait for the hand-back
                increment_group(noc, sem_x_free + slot, group_mask_arg);
                Semaphore<>(sem_x_free + slot).wait_min(group_cores * use);
            }
        }
        const XPiece piece = x_piece(jobs, job, chunk, member);
        if (piece.tiles) {
            const uint32_t dst = x_base + (slot * chunk_tiles * row_tiles + piece.first * job_row_tiles) * tile_bytes;
            if (job_row_tiles == 1 && rows <= expert_rows::face_rows) {  // the rows are in the top two faces of a tile
                for (GroupCoreIterator core(group_mask_arg); core.next();) {
                    for (uint32_t piece_tile = 0; piece_tile < piece.tiles; ++piece_tile) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(piece_tiles_addr + piece_tile * tile_bytes),
                            UnicastEndpoint{},
                            rows * face_row_bytes,
                            {},
                            {.noc_x = core.x, .noc_y = core.y, .addr = dst + piece_tile * tile_bytes});
                        noc.async_write(
                            CoreLocalMem<uint32_t>(piece_tiles_addr + piece_tile * tile_bytes + face_bytes),
                            UnicastEndpoint{},
                            rows * face_row_bytes,
                            {},
                            {.noc_x = core.x, .noc_y = core.y, .addr = dst + piece_tile * tile_bytes + face_bytes});
                    }
                }
            } else {
                for (GroupCoreIterator core(group_mask_arg); core.next();) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(piece_tiles_addr),
                        UnicastEndpoint{},
                        piece.tiles * job_row_tiles * tile_bytes,
                        {},
                        {.noc_x = core.x, .noc_y = core.y, .addr = dst});
                }
            }
            noc.async_write_barrier();
        }
        increment_group(noc, sem_x + slot, group_mask_arg);
    }

    // Chunk seq is in slot seq % x_slots of this core once every group core sent its piece: hand it to the compute.
    FORCE_INLINE void push_chunk(uint32_t seq) {
        const uint32_t slot = seq % x_slots, use = seq / x_slots;
        Semaphore<>(sem_x + slot).wait_min(group_cores * (use + 1));
        if constexpr (!slot_credits) {
            cb_x.reserve_back(chunk_tiles * row_tiles);
        }
        cb_x.push_back(chunk_tiles * row_tiles);
    }
};

// The intermediate exchange of a job: this core's activated tiles to every group core's cb_intermediate, then the
// job's whole intermediate to this core's compute.
struct IntermediateExchange {
    CircularBuffer cb_activated;
    CircularBuffer cb_intermediate;
    uint32_t activated_columns;  // real intermediate columns of this core
    uint32_t first_column;
    uint32_t intermediate_base;  // cb_intermediate, same L1 address on every group core
    uint32_t group_mask_arg;

    __attribute__((noinline)) void exchange(const GroupJobs& jobs, uint32_t job) {
        const Noc noc;
        const uint32_t job_row_tiles = expert_rows::row_tiles_of(jobs.rows(job));
        const uint32_t slot = job % intermediate_slots, use = job / intermediate_slots;
        cb_activated.wait_front(row_tiles * activated_tiles);
        const uint32_t activated_addr = cb_activated.get_read_ptr();
        if constexpr (slot_credits) {
            cb_intermediate.reserve_back(row_tiles * intermediate_tiles);  // this core's compute popped the slot
            if (use > 0) {
                increment_group(noc, sem_intermediate_free + slot, group_mask_arg);
                Semaphore<>(sem_intermediate_free + slot).wait_min(group_cores * use);
            }
        }
        for (GroupCoreIterator core(group_mask_arg); core.next();) {
            for (uint32_t row_tile = 0; row_tile < job_row_tiles; ++row_tile) {
                const uint32_t tile =
                    slot * row_tiles * intermediate_tiles + row_tile * intermediate_tiles + first_column;
                noc.async_write(
                    CoreLocalMem<uint32_t>(activated_addr + row_tile * activated_tiles * tile_bytes),
                    UnicastEndpoint{},
                    activated_columns * tile_bytes,
                    {},
                    {.noc_x = core.x, .noc_y = core.y, .addr = intermediate_base + tile * tile_bytes});
            }
        }
        noc.async_write_barrier();
        increment_group(noc, sem_intermediate + slot, group_mask_arg);
        cb_activated.pop_front(row_tiles * activated_tiles);
        Semaphore<>(sem_intermediate + slot).wait_min(group_cores * (use + 1));
        if constexpr (!slot_credits) {
            cb_intermediate.reserve_back(row_tiles * intermediate_tiles);
        }
        cb_intermediate.push_back(row_tiles * intermediate_tiles);
    }
};

// The output rows of a job: per output group and row tile, the real rows out of the compute's 4 bf16 tiles into
// row-major staging, then to the row buffer. The rows are untilized on this RISC, not by the packer: a per-job
// pack_untilize init / uninit between rows_compute.cpp's normal packs hangs the program's second launch on Blackhole.
template <typename RowsAccessor>
struct OutputRows {
    CircularBuffer cb_rows;
    RowsAccessor row_buffer;
    uint32_t output_groups;
    uint32_t first_output_tile;
    uint32_t output_tiles;  // real output tiles of this core
    uint32_t staging;       // tile_rows rows of output_row_stride

    __attribute__((noinline)) void write(const GroupJobs& jobs, uint32_t job) {
        const Noc noc;
        const uint32_t first_row = jobs.first_row(job), rows = jobs.rows(job);
        const uint32_t job_row_tiles = expert_rows::row_tiles_of(rows);
        for (uint32_t output_group = 0; output_group < output_groups; ++output_group) {
            for (uint32_t row_tile = 0; row_tile < job_row_tiles; ++row_tile) {
                cb_rows.wait_front(group_tiles);
                const uint32_t tiles_addr = cb_rows.get_read_ptr();
                const uint32_t real_tiles = output_tiles - group_tiles * output_group < group_tiles
                                                ? output_tiles - group_tiles * output_group
                                                : group_tiles;
                const uint32_t tile_real_rows =
                    rows - tile_rows * row_tile < tile_rows ? rows - tile_rows * row_tile : tile_rows;
                // this core's own L1 as the read source, as in XStream::build_piece
                noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    PrecomposedUnicastEndpoint{}, face_row_bytes, {.noc_addr = get_noc_addr(tiles_addr)});
                for (uint32_t group_tile = 0; group_tile < real_tiles; ++group_tile) {
                    for (uint32_t row = 0; row < tile_real_rows; ++row) {
                        const uint32_t src = tiles_addr + group_tile * tile_bytes + tile_row_offset(row);
                        const uint32_t dst = staging + row * output_row_stride + group_tile * tile_row_bytes;
                        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                            UnicastEndpoint{}, CoreLocalMem<uint32_t>(dst), face_row_bytes, {.addr = src}, {});
                        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                            UnicastEndpoint{},
                            CoreLocalMem<uint32_t>(dst + face_row_bytes),
                            face_row_bytes,
                            {.addr = src + face_bytes},
                            {});
                    }
                }
                noc.async_read_barrier();
                cb_rows.pop_front(group_tiles);
                // posted writes: non-posted row writes, many in flight under the weight stream, go unacknowledged and
                // hang the device
                const uint32_t column_offset = (first_output_tile + group_tiles * output_group) * tile_row_bytes;
                for (uint32_t row = 0; row < tile_real_rows; ++row) {
                    noc.async_write<NocOptions::POSTED>(
                        CoreLocalMem<uint32_t>(staging + row * output_row_stride),
                        row_buffer,
                        real_tiles * tile_row_bytes,
                        {},
                        {.page_id = first_row + tile_rows * row_tile + row, .offset_bytes = column_offset});
                }
                noc.async_writes_flushed<NocOptions::POSTED>();
            }
        }
    }
};

}  // namespace

void kernel_main() {
    CircularBuffer cb_activated(get_named_compile_time_arg_val("cb_activated"));
    CircularBuffer cb_intermediate(get_named_compile_time_arg_val("cb_intermediate"));
    CircularBuffer cb_routing(get_named_compile_time_arg_val("cb_routing"));
    CircularBuffer cb_x(get_named_compile_time_arg_val("cb_x"));
    CircularBuffer cb_scratch(get_named_compile_time_arg_val("cb_scratch"));
    CircularBuffer cb_rows(get_named_compile_time_arg_val("cb_rows"));
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto row_buffer_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t row_buffer_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t activated_columns = get_arg_val<uint32_t>(arg++);  // real intermediate columns of this core
    const uint32_t first_column = get_arg_val<uint32_t>(arg++);
    const uint32_t output_groups = get_arg_val<uint32_t>(arg++);  // W2 output groups
    const uint32_t first_output_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t output_tiles = get_arg_val<uint32_t>(arg++);  // real output tiles
    const uint32_t group = get_arg_val<uint32_t>(arg++);
    const uint32_t member = get_arg_val<uint32_t>(arg++);  // index of this core in its group
    const uint32_t group_mask_arg = arg;

    // cb_scratch: [x staging: tile_rows row_tiles rows of piece_row_stride][piece tiles: chunk_piece_max x row_tiles]
    // [output row staging: tile_rows rows of output_row_stride]
    const uint32_t x_staging = cb_scratch.get_write_ptr();
    const uint32_t piece_tiles_addr = x_staging + tile_rows * row_tiles * piece_row_stride;
    const uint32_t row_staging = piece_tiles_addr + chunk_piece_max * row_tiles * tile_bytes;
    const auto x_tensor = TensorAccessor(x_args, x_addr, row_bytes);
    const auto row_buffer = TensorAccessor(row_buffer_args, row_buffer_addr, row_bytes);
    const uint32_t x_base = cb_x.get_write_ptr();
    const uint32_t intermediate_base = cb_intermediate.get_write_ptr();
    const Noc noc;

    cb_routing.wait_front(1);
    invalidate_l1_cache();  // the table arrives by NoC read from the root core (rows_reader.cpp)
    const uint32_t routing = cb_routing.get_read_ptr();
    volatile tt_l1_ptr uint16_t* row_tokens =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(routing + routing_row_tokens);
    volatile tt_l1_ptr expert_rows::JobRecord* job_table =
        reinterpret_cast<volatile tt_l1_ptr expert_rows::JobRecord*>(routing + routing_jobs);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routing + routing_ctl);
    volatile tt_l1_ptr uint16_t* group_first_job =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(routing + routing_ctl + expert_rows::ctl_header_bytes);
    const uint32_t table_jobs = ctl[0];
    const GroupJobs jobs = expert_rows::group_jobs<(row_tiles > 1), expert_groups>(job_table, group_first_job, group);
    // the group's jobs: from the ctl block when dealt, else its share of the table's jobs
    const uint32_t job_count =
        row_tiles > 1 ? group_first_job[group + 1] - jobs.first
                      : (table_jobs > group ? (table_jobs - group + expert_groups - 1) / expert_groups : 0);
    if (job_count == 0) {
        return;
    }

    XStream<decltype(x_tensor)> x_stream{
        .cb_x = cb_x,
        .x_tensor = x_tensor,
        .row_tokens = row_tokens,
        .member = member,
        .staging = x_staging,
        .piece_tiles_addr = piece_tiles_addr,
        .x_base = x_base,
        .group_mask_arg = group_mask_arg};
    IntermediateExchange intermediate{
        .cb_activated = cb_activated,
        .cb_intermediate = cb_intermediate,
        .activated_columns = activated_columns,
        .first_column = first_column,
        .intermediate_base = intermediate_base,
        .group_mask_arg = group_mask_arg};
    OutputRows<decltype(row_buffer)> output_rows{
        .cb_rows = cb_rows,
        .row_buffer = row_buffer,
        .output_groups = output_groups,
        .first_output_tile = first_output_tile,
        .output_tiles = output_tiles,
        .staging = row_staging};

    if constexpr (!slot_credits) {
        // One chunk per job: job j's x goes to slot j % x_slots after the exchange of job j - x_slots (every core
        // finished its gate/up projection); job j - 1's rows are drained after the exchange of job j: the compute packs
        // them in the down projection of job j - 1, right after the gate/up projection of job j.
        for (uint32_t job = 0; job < x_slots && job < job_count; ++job) {
            x_stream.build_piece(jobs, job, 0);
            x_stream.send_piece(jobs, job, 0, job);
        }
        for (uint32_t job = 0; job < x_slots && job < job_count; ++job) {
            x_stream.push_chunk(job);
        }
        for (uint32_t job = 0; job < job_count; ++job) {
            intermediate.exchange(jobs, job);
            if (job + x_slots < job_count) {
                x_stream.build_piece(jobs, job + x_slots, 0);
                x_stream.send_piece(jobs, job + x_slots, 0, job + x_slots);
                x_stream.push_chunk(job + x_slots);
            }
            if (job > 0) {
                output_rows.write(jobs, job - 1);
            }
        }
        output_rows.write(jobs, job_count - 1);
    } else {
        // Several chunks per job: one stream of x chunks over all of this group's jobs; each chunk's piece is built
        // while the other group cores send theirs. The compute runs gate/up(0), gate/up(1), down(0), gate/up(2),
        // down(1), ...: once the first x_slots chunks of job j + 1 are in (their slots were freed by gate/up(j)), drain
        // the rows of down(j - 1) and exchange job j's activated tiles, which with one intermediate slot needs
        // down(j - 1) done on every group core. The next job's first chunks are then already in place when down(j)
        // ends.
        uint32_t seq = 0;
        x_stream.build_piece(jobs, 0, 0);
        for (uint32_t job = 0; job < job_count; ++job) {
            const uint32_t chunks = x_chunks(jobs, job);
            const uint32_t trigger = (x_slots < chunks ? x_slots : chunks) - 1;
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                x_stream.send_piece(jobs, job, chunk, seq);
                if (chunk + 1 < chunks) {
                    x_stream.build_piece(jobs, job, chunk + 1);
                } else if (job + 1 < job_count) {
                    x_stream.build_piece(jobs, job + 1, 0);
                }
                x_stream.push_chunk(seq++);
                if (chunk == trigger && job >= 1) {
                    if (job >= 2) {
                        output_rows.write(jobs, job - 2);
                    }
                    intermediate.exchange(jobs, job - 1);
                }
            }
        }
        if (job_count >= 2) {
            output_rows.write(jobs, job_count - 2);
        }
        intermediate.exchange(jobs, job_count - 1);
        output_rows.write(jobs, job_count - 1);
    }
    noc.async_full_barrier();
}
