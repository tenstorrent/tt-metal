// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Most reader threads a core runs (Quasar's DM cores); the factory binds a set of multicast semaphores for each.
constexpr uint32_t QUASAR_NUM_READER_THREADS = 4;

// Derived from a MatmulUnifiedProgramConfig + operand shapes by one function used by validation, output-spec
// derivation, the placement chooser and the factory, so they cannot disagree (all config constraints TT_FATAL there).
struct MatmulUnifiedPlan {
    uint32_t M_tiles = 0;
    uint32_t K_tiles = 0;
    uint32_t N_tiles = 0;
    // Batches of A (and of C), each with its own B; batches that use one B are folded into M_tiles.
    uint32_t batch_size = 0;
    // Tiles from one batch of B to the next: K_tiles * N_tiles when B has a [K x N] per batch, 0 when B is a
    // single [K x N] (or [1 x K x N]) that every batch of A multiplies. A's stride is always M_tiles * K_tiles.
    uint32_t B_batch_stride_tiles = 0;

    // Blocking, after the config's auto fields are resolved.
    // The C slices of one batch are walked across N then down M; C_slices_across_N is the walk's row
    // length, which the factory's per-core start positions and the sharded output layout both follow.
    uint32_t C_slices_across_N = 0;
    // Valid element columns in A's last K tile; 0 when K is a tile multiple (reader zeroes the rest).
    uint32_t A_last_K_tile_valid_columns = 0;
    // DST capacity (and the unpack modes the factory programs) assumed this accumulation mode.
    bool fp32_dest_acc_en = false;
    uint32_t C_slice_M_tiles = 0;
    uint32_t C_slice_N_tiles = 0;
    uint32_t K_chunk_tiles = 0;
    uint32_t K_chunks_per_C_slice = 0;  // K_tiles / K_chunk_tiles; every compute thread multiplies all of them
    // Rounded up to num_reader_threads so every reader thread pushes as many per C slice; the extra K chunks carry
    // credits only.
    uint32_t K_chunks_per_C_slice_padded = 0;
    uint32_t subblock_M_tiles = 0;
    uint32_t subblock_N_tiles = 0;
    // C slice dims rounded up to subblock multiples (equal when the subblock divides the slice).
    // Buffers and kernel loops use these; overshoot rows/columns are clipped on write.
    uint32_t C_slice_M_padded_tiles = 0;
    uint32_t C_slice_N_padded_tiles = 0;

    // Compute threads per core (NEOs). The C slice's subblocks, numbered across N then down M, are assigned
    // round-robin to the threads, which all see the whole A and B slices. A thread's subblocks of the C slice
    // (ceil(subblocks / threads) of them, back to back) take C_entries_per_thread one-tile entries of C_slice and
    // C_partials, the credits it moves per K chunk. With several threads each owns every num_compute_threads-th
    // entry, which the packer steps through (a whole stride per tile, #56195) and the writer reads by stride.
    uint32_t num_compute_threads = 1;
    uint32_t C_entries_per_thread = 0;

    // DM threads per core (Quasar DM cores; one each elsewhere). Reader thread t reads the A and B slices of K
    // chunks t, t + num_reader_threads, ... of every C slice into its own part of the A and B DFBs, which the
    // compute threads' waits take in turn. Writer thread t writes the
    // entries of compute threads t, t + num_writer_threads, ... (num_writer_threads divides num_compute_threads).
    uint32_t num_reader_threads = 0;
    uint32_t num_writer_threads = 0;
    // A and B slices each operand DFB holds, and C slices in flight (C_slice holds C_buffer_depth C slices of every
    // compute thread's entries).
    uint32_t operand_buffer_depth = 0;
    uint32_t C_buffer_depth = 0;
    // Multicast, when one batch group (all the cores) produces one C slice per core and batch on a rectangle laid out
    // as the C slices tile C: the first core of each row reads the row's A slices and multicasts them along the row,
    // and the first core of each column does the same with B down the column. When C is one row of C slices (one
    // column) every core of the rectangle needs the same A (B), which its first core multicasts to all of them. Cores
    // each sender multicasts to; 0 = that operand is not.
    uint32_t A_mcast_num_dests = 0;
    uint32_t B_mcast_num_dests = 0;
    bool A_mcast_to_all_cores = false;
    bool B_mcast_to_all_cores = false;

    // C slice assignment: the active cores split into num_batch_groups groups of consecutive cores, and group g takes
    // a contiguous run of the batches (the first batch_size % num_batch_groups groups one longer). Within a group one
    // batch's C slices, walked across N then down M, are split into contiguous runs per core, which the core produces
    // for each of its group's batches (the factory derives the per-core RTAs). One group unless one batch has too few
    // C slices for the cores (batches with their own B).
    uint32_t C_slices_per_batch = 0;
    uint32_t num_batch_groups = 1;
    tt::tt_metal::ShardOrientation orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t max_C_slices_per_core = 0;  // per batch; sizes the DFBs and gates partials aliasing

    // Borrowable: the operand's L1 shard on each active core matches the C slice (batch 1, one C slice per core, a
    // shard grid in assignment order; C also needs one compute thread: with several, each thread owns every N-th
    // C_slice entry, not the shard's tile order). Borrowed: also nothing padded, so the shard is bound as the DFB
    // itself, no copy (A: and one K chunk covers K; C: and the subblocks span the C slice width).
    bool A_shard_borrowable = false;
    bool B_shard_borrowable = false;
    bool C_shard_borrowable = false;
    bool borrow_A = false;
    bool borrow_B = false;
    bool borrow_C = false;

    // DFB sizing; entry sizes are in bytes. Every entry holds one tile; C_slice / C_partials hold
    // C_entries_per_thread entries per thread (see above).
    bool packer_l1_acc_en = false;
    // With packer L1 accumulation in C's own format and no activation on the finished sum, every K chunk adds
    // straight onto the C slice's entries: no partials round trip, and the C slice is reserved once.
    bool accumulate_in_C_slice = false;
    tt::DataFormat A_format{};
    tt::DataFormat B_format{};
    tt::DataFormat C_format{};
    tt::DataFormat C_partials_format{};
    uint32_t A_entry_bytes = 0;
    uint32_t B_entry_bytes = 0;
    uint32_t C_entry_bytes = 0;
    uint32_t C_partials_entry_bytes = 0;
    uint32_t A_slice_entries = 0;
    uint32_t B_slice_entries = 0;
    uint32_t C_slice_entries = 0;
    uint32_t C_partials_entries = 0;
    // C_partials reuses C_slice's L1; only safe when partials are never live while C_slice holds unread data.
    bool alias_C_partials_onto_C_slice = false;
    bool has_fused_activation = false;  // it needs the finished sum in DST, so no accumulate_in_C_slice
    // Fused bias: one tile row of the C slice's columns per C slice, loaded into DST before K chunk 0 so the
    // sum carries it once. Reader thread t pushes the bias of every num_reader_threads-th C slice into its own
    // part of the DFB, one set of C_slice_N_padded_tiles entries each.
    bool has_bias = false;
    uint32_t bias_entry_bytes = 0;
    uint32_t bias_entries = 0;
    uint64_t l1_bytes = 0;  // total DFB footprint per core

    // C is sharded and each active core's shard is its one C slice (C's own shard spec, or for a C the op allocates,
    // C_slice_shard_spec's), so every core writes its own shard. Otherwise the writer reaches C's shards through the
    // tensor accessor wherever they are.
    bool C_slices_are_C_shards = false;

    // Estimated cycles (the lowest of every C slice, subblock and K chunk tried), and the busiest core's pipeline
    // alone, which breaks ties between plans that DRAM bandwidth bounds equally. See ArchCostModel.
    std::pair<double, double> estimated_cycles;
};

// `output` is the C tensor when the caller supplied one (or the op already created it); it decides whether
// C can be packed in place. Without it the op allocates C, a sharded C without a shard spec with C_slice_shard_spec's.
// Of the auto choices, those whose C slices are a sharded C's shards come first, then the lowest estimate.
// must_fit FATALs with the breakdown when nothing fits L1; without it that returns nullopt (choose_unified_placement's
// candidate search). Other config errors always FATAL.
std::optional<MatmulUnifiedPlan> plan_unified_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    uint32_t bias_tile_bytes,  // 0 without a bias
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config,
    const MatmulParams& attributes,
    const std::optional<ttnn::Tensor>& output,
    bool must_fit);

// The shard spec of C whose shards are the plan's C slices, each on the core that produces it, in C_memory's layout;
// nullopt when they do not form one: C not L1-sharded, several C slices per core (over all its batches), batches whose
// rows the C slices do not tile exactly, or for block sharding cores that are not one rectangle laid out as the C
// slices tile C.
std::optional<tt::tt_metal::ShardSpec> C_slice_shard_spec(
    const MatmulUnifiedPlan& plan, const tt::tt_metal::MemoryConfig& C_memory);

// The machine the plan's cycle estimate assumes, per arch, fitted to measured kernel times: Wormhole on an n150,
// Quasar on craq-sim (which models no DRAM bandwidth, so Quasar's DRAM figure is unmeasured). Blackhole is
// Wormhole with Blackhole's NoC and DRAM bandwidth. Costs are per compute thread or per core; a tile is 2 KB (a
// Float16_b tile), and other formats scale by their tile size.
struct ArchCostModel {
    // Compute, on one compute thread.
    double tile_matmul_cycles;    // one 32x32x32 tile product per math fidelity phase
    double unpack_A_tile_cycles;  // per K tile, per A tile (subblock row) unpacked; overlaps the math
    double unpack_B_tile_cycles;  // per K tile, per B tile (subblock column) unpacked
    double subblock_cycles;       // per subblock per K chunk: DST handshake and inits
    double tile_pack_cycles;      // per tile packed out of DST
    double tile_reload_cycles;    // per partial sum tile reloaded into DST
    double K_chunk_cycles;        // per K chunk: DFB waits and pack init
    // Data movement, on one core.
    double tile_read_cycles;       // per tile one reader thread reads
    double core_tile_read_cycles;  // per tile all of a core's reader threads read (its NoC bandwidth)
    double tile_mcast_cycles;      // per tile multicast to the other cores of a row, column or rectangle
    double tile_write_cycles;      // per C tile written
    double K_chunk_read_cycles;    // per K chunk a reader thread reads: barrier and DFB credits
    double K_chunk_mcast_cycles;   // more per K chunk when an operand is multicast: the semaphore round trip
    double read_latency_cycles;    // before the first K chunk's data arrives
    double C_slice_cycles;         // per C slice
    double A_pad_tile_cycles;      // per A row of a C slice whose last tile (ragged K) the reader zero-pads
    // A pipeline step costs (read^p + compute^p)^(1/p) for p = overlap_exponent: large p overlaps the two fully,
    // p = 1 not at all.
    double overlap_exponent;
    // The device.
    double dram_tile_cycles;  // per tile of DRAM traffic, all cores together
};

constexpr ArchCostModel WORMHOLE_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .unpack_A_tile_cycles = 47.1,
    .unpack_B_tile_cycles = 37.5,
    .subblock_cycles = 0.0,
    .tile_pack_cycles = 26.2,
    .tile_reload_cycles = 223.6,
    .K_chunk_cycles = 538.9,
    .tile_read_cycles = 64.5,
    .core_tile_read_cycles = 90.0,
    .tile_mcast_cycles = 110.6,
    .tile_write_cycles = 62.3,
    .K_chunk_read_cycles = 0.0,
    .K_chunk_mcast_cycles = 373.3,
    .read_latency_cycles = 1611.0,
    .C_slice_cycles = 269.6,
    .A_pad_tile_cycles = 8099.9,
    .overlap_exponent = 10.0,
    .dram_tile_cycles = 10.7,
};

// Fitted on p100a (11x10 grid, 7 DRAM channels); p150 (more cores and DRAM channels) uses it untuned.
constexpr ArchCostModel BLACKHOLE_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .unpack_A_tile_cycles = 17.4,
    .unpack_B_tile_cycles = 17.3,
    .subblock_cycles = 0.0,
    .tile_pack_cycles = 8.5,
    .tile_reload_cycles = 20.0,
    .K_chunk_cycles = 1450.2,
    .tile_read_cycles = 92.9,
    .core_tile_read_cycles = 0.0,
    .tile_mcast_cycles = 48.2,
    .tile_write_cycles = 65.5,
    .K_chunk_read_cycles = 0.7,
    .K_chunk_mcast_cycles = 364.6,
    .read_latency_cycles = 0.0,
    .C_slice_cycles = 1.1,
    .A_pad_tile_cycles = 544.8,
    .overlap_exponent = 4.4,
    .dram_tile_cycles = 8.9,
};

constexpr ArchCostModel QUASAR_COST_MODEL{
    .tile_matmul_cycles = 16.0,
    .unpack_A_tile_cycles = 13.5,
    .unpack_B_tile_cycles = 9.7,
    .subblock_cycles = 0.0,
    .tile_pack_cycles = 0.0,
    .tile_reload_cycles = 8.5,
    .K_chunk_cycles = 164.3,
    .tile_read_cycles = 37.9,
    .core_tile_read_cycles = 5.8,
    .tile_mcast_cycles = 0.7,
    .tile_write_cycles = 19.9,
    .K_chunk_read_cycles = 0.0,
    .K_chunk_mcast_cycles = 163.7,
    .read_latency_cycles = 378.9,
    .C_slice_cycles = 0.0,
    .A_pad_tile_cycles = 1000.0,
    .overlap_exponent = 6.9,
    // Not on the simulator (no DRAM model): QSR1.A1's 1.0 TB/s at 1.4 GHz is 2.9 cycles per 2 KB tile at peak; 75%
    // of it, as p100a streams at 71-76% of its peak.
    .dram_tile_cycles = 3.8,
};

// A config is what the caller asks for, its placement is the cores, C slice and orientation, and the plan
// (MatmulUnifiedPlan) is everything derived from a config with a placement.
// Returns `requested` with the placement whose plan has the lowest estimated cycles, and that plan's C slice, subblock,
// K chunk and batch groups. The candidates: on the grid, a 2D rectangle and 1D across N and across M (every C slice
// length), and the whole grid for batches with their own B; a sharded C's shard grid; and the cores of a
// height-sharded A or a width-sharded B. A sharded C prefers placements whose C slices are its shards (written in
// place), but takes any. nullopt when the unified matmul cannot run the case or no placement fits.
std::optional<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig> choose_unified_placement(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    uint32_t bias_tile_bytes,  // 0 without a bias
    const MatmulParams& attributes,
    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& requested);

}  // namespace ttnn::prim::qsr
