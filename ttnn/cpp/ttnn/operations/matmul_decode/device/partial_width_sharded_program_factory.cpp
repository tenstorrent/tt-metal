// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <map>
#include <optional>
#include <vector>

namespace ttnn::operations::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

// Partial width-sharded matmul: C = A @ B, where B is sharded along BOTH K and N.
//
// The caller reshapes/permutes a [K, N] weight into a width-sharded tensor whose
// shard shape is [Kc, Nc] across K_blocks * N_blocks cores (Kc = K / K_blocks,
// Nc = N / N_blocks). Cores are laid out k-major in row-major order, so the core at
// row-major index `c` owns B block (k_idx = c / N_blocks, n_idx = c % N_blocks).
//
// Pipeline (per core):
//   1. Reader (reused reader_full_width_sharded): gather the *entire* A matrix onto
//      every core via multicast, and publish this core's resident B block.
//   2. Compute (phase 1): matmul this core's K-slice of A with its B block to produce
//      a *partial* [M, Nc] product.
//   3. Writer: NoC-write the partial into slot `k_idx` of the reduce CB on the base
//      core (the k_idx == 0 core for this n_idx, which owns the output N-slice), then
//      bump the base core's reduce semaphore.
//   4. Compute (phase 2, base cores only): once all K_blocks partials have arrived,
//      sum them into the output shard.
//
// Base cores (k_idx == 0) coincide with the width(N)-sharded output cores.
ProgramDescriptor MatmulDecodeDeviceOperation::PartialWidthSharded::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    const auto& inputA_tile = input_tensor_a.tensor_spec().tile();
    const auto& inputB_tile = input_tensor_b.tensor_spec().tile();
    const auto& output_tile = output_tensor.tensor_spec().tile();
    const uint32_t in0_tile_size = inputA_tile.get_tile_size(in0_data_format);
    const uint32_t in1_tile_size = inputB_tile.get_tile_size(in1_data_format);
    const uint32_t out_tile_size = output_tile.get_tile_size(out_data_format);

    // With tiny tiles (e.g. tile height 16) the in0/in1/out tiles no longer share a
    // common geometry, so each circular buffer must carry its own tile descriptor in
    // addition to its own page (tile) size. full_in0 (gathered A) reuses the in0 tile;
    // the partial and reduce CBs hold [M, Nc] products and reuse the out tile.
    const TileDescriptor in0_tile_desc{inputA_tile};
    const TileDescriptor in1_tile_desc{inputB_tile};
    const TileDescriptor out_tile_desc{output_tile};

    log_info(
        tt::LogOp,
        "MatmulDecode(partial): in0_tile_size: {}, in1_tile_size: {}, out_tile_size: {}",
        in0_tile_size,
        in1_tile_size,
        out_tile_size);

    const uint32_t inputA_tile_height = inputA_tile.get_height();
    const uint32_t inputA_tile_width = inputA_tile.get_width();
    const uint32_t inputB_tile_height = inputB_tile.get_height();
    const uint32_t inputB_tile_width = inputB_tile.get_width();
    const uint32_t output_tile_height = output_tile.get_height();
    const uint32_t output_tile_width = output_tile.get_width();

    TT_FATAL(
        inputA_tile_height == output_tile_height,
        "Input tensor A {} and output tile height {} must be equal",
        inputA_tile_height,
        output_tile_height);
    TT_FATAL(
        inputB_tile_height == tt::constants::TILE_HEIGHT,
        "Input tensor B {} tile height must be 32",
        inputB_tile_height);
    TT_FATAL(
        inputA_tile_width == tt::constants::TILE_WIDTH,
        "Input tensor A tile width {} must be equal to the tile width 32",
        inputA_tile_width);
    TT_FATAL(
        inputB_tile_width == tt::constants::TILE_WIDTH,
        "Input tensor B tile width {} must be equal to the tile width 32",
        inputB_tile_width);
    TT_FATAL(
        output_tile_width == tt::constants::TILE_WIDTH,
        "Output tensor tile width {} must be equal to the tile width 32",
        output_tile_width);

    log_info(tt::LogOp, "MatmulDecode(partial): inputA_tile: {}", inputA_tile);

    IDevice* device = input_tensor_a.device();

    // ---- Recover the 2D (K x N) block-sharding geometry ----
    // Operation attributes M,N,K are the real matmul dimensions. Weights tensor has been reshaped, so its logical shape
    // is no longer [K, N].
    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    // The compute kernel processes the entire M dimension in a single DST block
    // (out_block_h = M_tiles), so M_tiles must fit in DST (<= 8 tiles in half-sync mode).
    TT_FATAL(
        M_tiles <= 8,
        "partial_width_sharded matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got "
        "M_tiles={} (M={}, inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    // ---- in0 (A) geometry ----
    // Default (buffer-backed in0): A is ALREADY width-sharded, so its geometry comes from its
    // shard spec. reshard_input: A is passed un-resharded (interleaved OR block/width-sharded) and
    // the reader reshards it internally via a TensorAccessor over A's buffer, so we SYNTHESIZE the
    // staging geometry the gather wants (A staged [M, K/reshard_cores] across the first
    // reshard_cores cores, row-major) independent of A's actual source layout.
    const bool reshard_input = operation_attributes.reshard_input;
    std::array<uint32_t, 2> inputA_shard_shape;
    CoreRangeSet inputA_core_range_set;
    if (reshard_input) {
        const uint32_t reshard_cores = operation_attributes.reshard_cores;
        inputA_shard_shape = {M_tiles * inputA_tile_height, operation_attributes.K / reshard_cores};
        const std::vector<CoreCoord> a_sender_cores = corerange_to_cores(
            tt::tt_metal::num_cores_to_corerangeset(reshard_cores, device->compute_with_storage_grid_size(), true),
            std::nullopt,
            true);
        std::vector<CoreRange> a_sender_ranges;
        a_sender_ranges.reserve(a_sender_cores.size());
        for (const auto& core : a_sender_cores) {
            a_sender_ranges.emplace_back(core, core);
        }
        inputA_core_range_set = CoreRangeSet(a_sender_ranges);
    } else {
        inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
        inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    }
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height),
        "Input tensor A shard height {} must equal M_tiles {} * tile height {}",
        inputA_shard_shape[0],
        M_tiles,
        inputA_tile_height);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A shard width must be divisible by the tile width");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
    // Shard shape correctly maps to the per core K and N dimensions.`
    const uint32_t Kc = inputB_shard_shape[0];
    const uint32_t Nc = inputB_shard_shape[1];
    const uint32_t Kc_tiles = Kc / tt::constants::TILE_WIDTH;
    const uint32_t Nc_tiles = Nc / tt::constants::TILE_WIDTH;

    const auto inputB_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    const uint32_t num_B_cores_along_N = N_tiles / Nc_tiles;
    TT_FATAL(
        num_B_cores % num_B_cores_along_N == 0,
        "num_B_cores {} must be divisible by num_B_cores_along_N {}",
        num_B_cores,
        num_B_cores_along_N);
    const uint32_t num_B_cores_along_K = num_B_cores / num_B_cores_along_N;
    const uint32_t K_blocks = K_tiles / Kc_tiles;
    TT_FATAL(
        num_B_cores_along_K == K_blocks,
        "num_B_cores_along_K {} must equal K_blocks {}",
        num_B_cores_along_K,
        K_blocks);
    const uint32_t N_blocks = num_B_cores / K_blocks;

    // Base (output-owning) cores are the k_idx==0 cores = the first N_blocks of the
    // k-major B cores. When the output is width-sharded they coincide with the output
    // shard grid; for the interleaved-output fold there is no output shard spec, so we
    // derive the base cores directly from B's core ordering.
    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::vector<CoreRange> base_core_ranges;
    base_core_ranges.reserve(N_blocks);
    for (uint32_t i = 0; i < N_blocks; ++i) {
        base_core_ranges.emplace_back(b_cores[i], b_cores[i]);
    }
    const CoreRangeSet base_core_range_set(base_core_ranges);
    const auto output_core_range_set = operation_attributes.interleaved_output
                                           ? base_core_range_set
                                           : output_tensor.memory_config().shard_spec().value().grid;
    if (!operation_attributes.interleaved_output) {
        TT_FATAL(
            output_core_range_set.num_cores() == N_blocks,
            "Output must be sharded across N_blocks {} cores, but got {}",
            N_blocks,
            output_core_range_set.num_cores());
    }

    // A is multicast onto every B core; senders are the A-holding cores.
    log_debug(
        tt::LogOp,
        "num_B_cores: {}, num_B_cores_along_N: {}, num_B_cores_along_K: {}, K_blocks: {}, N_blocks: {}",
        num_B_cores,
        num_B_cores_along_N,
        num_B_cores_along_K,
        K_blocks,
        N_blocks);
    log_debug(
        tt::LogOp,
        "inputA_num_cores: {}, inputB_num_cores: {}, output_num_cores: {}",
        inputA_core_range_set.num_cores(),
        inputB_core_range_set.num_cores(),
        output_core_range_set.num_cores());
    const auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set).merge(output_core_range_set);
    const auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());

    log_debug(
        tt::LogOp,
        "MatmulDecode(partial): M_tiles={}, K_tiles={}, Kc_tiles={}, Nc_tiles={}, K_blocks={}, N_blocks={}",
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        N_blocks);

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    constexpr uint32_t in0_cb_index = CBIndex::c_0;       // this core's A slice (gather source)
    constexpr uint32_t in1_cb_index = CBIndex::c_1;       // this core's B block (resident)
    constexpr uint32_t out_cb_index = CBIndex::c_2;       // final output shard (base cores)
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;  // gathered full A
    constexpr uint32_t partial_cb_index = CBIndex::c_4;   // this core's partial product
    constexpr uint32_t reduce_cb_index = CBIndex::c_5;    // gathered K_blocks partials (base cores)

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;  // tiles in one (partial / output) shard

    // in0: this core's resident A slice (gather source). Default: buffer-backed by the
    // already-width-sharded A. reshard_input: a plain CB the reader NoC-reads A's K-slice into
    // (from the interleaved A buffer) before the multicast gather.
    CBDescriptor in0_cb_desc{
        .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    };
    if (!reshard_input) {
        in0_cb_desc.buffer = input_tensor_a.buffer();
    }
    desc.cbs.push_back(in0_cb_desc);
    // in1: this core's resident B block (buffer-backed).
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * Nc_tiles * in1_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
    });
    // out: final output shard (base cores only). For a width-sharded output the CB is
    // buffer-backed (it IS the output shard). For the interleaved-output fold it is a
    // plain staging CB the writer NoC-scatters into the interleaved output buffer.
    CBDescriptor out_cb_desc{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    };
    if (!operation_attributes.interleaved_output) {
        out_cb_desc.buffer = output_tensor.buffer();
    }
    desc.cbs.push_back(out_cb_desc);
    // full_in0: gathered full A (multicast destination).
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * K_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });
    // partial: this core's matmul partial product (compute -> writer).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    // reduce: gathered K_blocks partials on the base core (writer -> compute). Allocated
    // identically on every core so each sender can use its local write pointer as the
    // (matching) destination L1 address on the base core.
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_blocks * block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = reduce_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });

    // ---- Semaphores ----
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t gather_sem_id = 0;  // senders -> coordinator (A gather)
    constexpr uint32_t done_sem_id = 1;    // coordinator -> all (A gathered)
    constexpr uint32_t reduce_sem_id = 2;  // partial-producers -> base core (reduction)
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = gather_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = done_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = reduce_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});

    // ---- Reader kernel (A gather) : reuse the full-width-sharded reader ----
    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    const CoreCoord coordinator_logical = mcast_bbox.start_coord;
    const CoreCoord mcast_start_phys = device->worker_core_from_logical_core(coordinator_logical);
    const CoreCoord mcast_end_phys = device->worker_core_from_logical_core(mcast_bbox.end_coord);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        in0_cb_index,
        full_in0_cb_index,
        shard_num_tiles,
        in0_tile_size,
        num_senders,
        num_receivers,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        static_cast<uint32_t>(mcast_end_phys.x),
        static_cast<uint32_t>(mcast_end_phys.y),
        gather_sem_id,
        done_sem_id,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        in1_cb_index,
        Kc_tiles * Nc_tiles,
        static_cast<uint32_t>(reshard_input),  // 16: read interleaved A K-slice before multicast
        M_tiles,                               // 17: A height in tiles
        inA_K_tiles_per_core,                  // 18: this sender's K-slice width in tiles
        K_tiles,                               // 19: global A width in tiles (row-major page stride)
    };
    // 20+: TensorAccessor args for the interleaved in0 buffer (senders NoC-read their K-slice from
    // it when reshard_input). Harmless when unused (read_interleaved == 0).
    tt::tt_metal::TensorAccessorArgs(*input_tensor_a.buffer()).append_to(reader_compile_time_args);

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);

    auto build_reader_kernel = [&](const std::vector<CoreCoord>& cores, NOC noc) {
        std::vector<CoreRange> ranges;
        ranges.reserve(cores.size());
        for (const auto& core : cores) {
            ranges.emplace_back(core, core);
        }
        KernelDescriptor reader_kernel_desc;
        reader_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/reader_partial_width_sharded.cpp";
        reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_kernel_desc.core_ranges = CoreRangeSet(ranges);
        reader_kernel_desc.compile_time_args = reader_compile_time_args;
        reader_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = noc,
        };
        reader_kernel_desc.runtime_args.reserve(cores.size());
        for (const auto& core : cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            const uint32_t sender_id = is_sender ? it->second : 0;
            const bool is_coordinator = (core == coordinator_logical);
            reader_kernel_desc.emplace_runtime_args(
                core,
                {static_cast<uint32_t>(is_sender),
                 sender_id,
                 static_cast<uint32_t>(is_coordinator),
                 input_tensor_a.buffer()});  // 3: interleaved A base addr (patched; used iff read_interleaved)
        }
        return reader_kernel_desc;
    };

    // Single reader instance over all cores on NOC_0; the writer below runs on NOC_1 so the reader
    // (RISCV_1) and writer (RISCV_0) on each core use opposite NOCs.
    desc.kernels.push_back(build_reader_kernel(all_reader_cores, NOC::NOC_0));

    // ---- Writer kernel (cross-core K-reduction) ----
    //
    // Runs on every B core. Each core ships its partial to slot `k_idx` of the base
    // core's reduce CB and bumps that core's reduce semaphore. Base cores additionally
    // wait for all K_blocks partials and publish the reduce CB to the compute kernel.
    log_debug(tt::LogOp, "b_cores: {}", b_cores);
    log_debug(tt::LogOp, "output_core_range_set: {}", output_core_range_set);
    log_debug(tt::LogOp, "inputB_core_range_set: {}", inputB_core_range_set);
    log_debug(tt::LogOp, "inputA_core_range_set: {}", inputA_core_range_set);
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    auto build_writer_kernel = [&](const std::vector<uint32_t>& core_indices, NOC noc) {
        std::vector<CoreRange> ranges;
        ranges.reserve(core_indices.size());
        for (const uint32_t idx : core_indices) {
            ranges.emplace_back(b_cores[idx], b_cores[idx]);
        }
        KernelDescriptor writer_kernel_desc;
        writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/writer_partial_width_sharded.cpp";
        writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_kernel_desc.core_ranges = CoreRangeSet(ranges);
        writer_kernel_desc.compile_time_args = {
            partial_cb_index,
            reduce_cb_index,
            block_num_tiles,
            out_tile_size,
            K_blocks,
            reduce_sem_id,
            static_cast<uint32_t>(operation_attributes.interleaved_output),  // 6: fold S->I reshard
            out_cb_index,                                                    // 7
            M_tiles,                                                         // 8
            Nc_tiles,                                                        // 9
            N_tiles,                                                         // 10
        };
        // 11+: TensorAccessor args for the interleaved output buffer (base cores scatter
        // their N-slice into it when interleaved_output). Harmless when unused.
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_kernel_desc.compile_time_args);
        writer_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = noc,
        };
        writer_kernel_desc.runtime_args.reserve(core_indices.size());
        for (const uint32_t idx : core_indices) {
            const uint32_t k_idx = idx / N_blocks;
            const uint32_t n_idx = idx % N_blocks;
            const CoreCoord base_logical = b_cores[n_idx];  // k_idx == 0 core for this n_idx
            const CoreCoord base_phys = device->worker_core_from_logical_core(base_logical);
            const bool is_base = (k_idx == 0);
            log_trace(
                tt::LogOp,
                "Writer core {}, idx {}, k_idx: {}, n_idx: {}, base_logical: {}, base_phys: {}, is_base: {}",
                b_cores[idx],
                idx,
                k_idx,
                n_idx,
                base_logical,
                base_phys,
                is_base);
            // n_idx (this base core's output N-slice) + output buffer base address
            // (Buffer* -> patched by the framework) for the interleaved-output scatter.
            writer_kernel_desc.emplace_runtime_args(
                b_cores[idx],
                {k_idx,
                 static_cast<uint32_t>(base_phys.x),
                 static_cast<uint32_t>(base_phys.y),
                 static_cast<uint32_t>(is_base),
                 n_idx,
                 output_tensor.buffer()});
        }
        return writer_kernel_desc;
    };

    // Single writer instance over all b_cores on NOC_1 (opposite NOC to the reader).
    std::vector<uint32_t> all_writer_indices(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        all_writer_indices[idx] = idx;
    }
    desc.kernels.push_back(build_writer_kernel(all_writer_indices, NOC::NOC_1));

    // ---- Compute kernel (partial matmul + base-core reduction) ----
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/compute/compute_partial_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        inA_K_tiles_per_core,  // needed to translate global K-tile -> sender-major full_in0 slot (M_tiles>1)
        static_cast<uint32_t>(operation_attributes.fused_gelu),         // fuse (erf) gelu into phase-2 output pack
        static_cast<uint32_t>(operation_attributes.fused_gelu_approx),  // 0=erf, 1=tanh-approx
    };
    log_debug(
        tt::LogOp,
        "M_tiles: {}, K_tiles: {}, Kc_tiles: {}, Nc_tiles: {}, K_blocks: {}",
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks);
    const auto ckc = ttnn::init_device_compute_kernel_config(
        input_tensor_a.device()->arch(),
        operation_attributes.compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = ckc.math_fidelity,
        .fp32_dest_acc_en = ckc.fp32_dest_acc_en,
        .dst_full_sync_en = ckc.dst_full_sync_en,
        .math_approx_mode = ckc.math_approx_mode,
    };
    compute_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const bool is_base = (k_idx == 0);
        if (is_base) {
            TT_FATAL(
                output_core_range_set.contains(b_cores[idx]),
                "Base core {} is not in output core range set",
                b_cores[idx]);
        }
        log_trace(
            tt::LogOp,
            "core {}, idx {}, k_idx: {}, is_base: {}",
            b_cores[idx],
            idx,
            k_idx,
            static_cast<uint32_t>(is_base));
        compute_kernel_desc.runtime_args.emplace_back(
            b_cores[idx], KernelDescriptor::CoreRuntimeArgs{k_idx, static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::matmul_decode
