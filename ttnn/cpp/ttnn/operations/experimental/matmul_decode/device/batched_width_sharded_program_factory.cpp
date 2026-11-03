// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <map>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::ComputeHardwareConfig;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::PrefetcherPipeArgument;
using tt::tt_metal::experimental::PrefetcherPipeParameter;
using tt::tt_metal::experimental::PrefetcherPipeParamName;
using tt::tt_metal::experimental::ProducerOf;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {
// Batched width-sharded: weights folded along batch and N; block-diagonal, no cross-core reduction.
ttnn::device_operation::ProgramArtifacts create_batched_artifacts(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    const auto& inputB_tile = input_tensor_b.tensor_spec().tile();
    const tt::tt_metal::Tile in0_tile = in0_tile_for_compute(input_tensor_a);
    const tt::tt_metal::Tile output_tile = out_tile_for_compute(input_tensor_a, output_tensor);
    const uint32_t in0_tile_size = in0_tile.get_tile_size(in0_data_format);
    const uint32_t in1_tile_size = inputB_tile.get_tile_size(in1_data_format);
    const uint32_t out_tile_size = output_tile.get_tile_size(out_data_format);

    const uint32_t inputA_tile_height = in0_tile.get_height();
    const uint32_t inputA_tile_width = in0_tile.get_width();
    const uint32_t inputB_tile_height = inputB_tile.get_height();
    const uint32_t inputB_tile_width = inputB_tile.get_width();
    const uint32_t output_tile_height = output_tile.get_height();
    const uint32_t output_tile_width = output_tile.get_width();

    TT_FATAL(
        inputA_tile_height == output_tile_height,
        "Input tensor A tile height {} and output tile height {} must be equal",
        inputA_tile_height,
        output_tile_height);
    TT_FATAL(
        inputB_tile_height == tt::constants::TILE_HEIGHT,
        "Input tensor B tile height {} must be 32",
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

    const uint32_t batch = operation_attributes.batch;
    const uint32_t b_blocks = operation_attributes.b_blocks;
    const uint32_t n_blocks = operation_attributes.n_blocks;
    const uint32_t Bc = batch / b_blocks;                   // batches per core
    const uint32_t Nc = operation_attributes.N / n_blocks;  // N per core

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t Nc_tiles = div_up(Nc, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "batched matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got M_tiles={} (M={}, "
        "inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A shard width {} must be divisible by the tile width {}",
        inputA_shard_shape[1],
        tt::constants::TILE_WIDTH);
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    const auto& prefetcher_pipes = operation_attributes.prefetcher_pipes;
    const bool use_pipes = !prefetcher_pipes.empty();
    // The reader loads this core's slab of a DRAM ND-sharded weight itself.
    const bool in1_from_dram = operation_attributes.dram_weight_grid.has_value();
    // A packed weight is a region of a fused height-sharded tensor: its [Bc*K, Nc] slab shape
    // and its grid come from the spec, since the fused tensor's shard spec describes the whole
    // pack. A DRAM ND-sharded weight has no legacy shard spec, so both come from the ND shard
    // spec and the pipes' receivers or dram_weight_grid.
    const auto& packed = operation_attributes.packed_weight;
    const uint32_t b_shard_height = use_pipes || in1_from_dram
                                        ? static_cast<uint32_t>(input_tensor_b.nd_shard_spec()->shard_shape[-2])
                                    : packed.has_value() ? Bc * static_cast<uint32_t>(operation_attributes.K)
                                                         : input_tensor_b.memory_config().shard_spec().value().shape[0];
    const uint32_t b_shard_K_tiles = b_shard_height / tt::constants::TILE_HEIGHT;  // = Bc * K_tiles

    const auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto inputB_core_range_set =
        use_pipes
            ? tt::tt_metal::experimental::GetPrefetcherPipeReceiverCores(ttnn::prefetcher_pipe_refs(prefetcher_pipes))
        : in1_from_dram      ? *operation_attributes.dram_weight_grid
        : packed.has_value() ? packed->cores
                             : input_tensor_b.memory_config().shard_spec().value().grid;
    const bool in0_rm_hs = operation_attributes.in0_row_major_height_sharded;
    if (in0_rm_hs) {
        TT_FATAL(
            inputA_core_range_set.contains(inputB_core_range_set),
            "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A requires A's core grid {} to contain B's core grid {}",
            inputA_core_range_set.str(),
            inputB_core_range_set.str());
        TT_FATAL(
            inputA_shard_shape[1] == static_cast<uint32_t>(operation_attributes.K),
            "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A requires shard width {} to equal K {}",
            inputA_shard_shape[1],
            operation_attributes.K);
        TT_FATAL(
            inputA_shard_shape[0] == batch * M_tiles * inputA_tile_height,
            "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A requires shard height {} to equal batch * M = {} * {}",
            inputA_shard_shape[0],
            batch,
            operation_attributes.M);
    }

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    TT_FATAL(
        num_B_cores == b_blocks * n_blocks,
        "batched matmul_decode expects B sharded across b_blocks * n_blocks = {} * {} = {} cores, but got {}",
        b_blocks,
        n_blocks,
        b_blocks * n_blocks,
        num_B_cores);

    const bool mcast_out = operation_attributes.output_core_grid.has_value();
    const bool mcast_two_hub = mcast_out && operation_attributes.output_mcast_two_hub;
    const auto output_core_range_set = mcast_out ? *operation_attributes.output_core_grid : inputB_core_range_set;
    // A row-major output without mcast is the out buffer itself, so nothing writes it back out.
    const bool out_is_output = output_tensor.layout() == Layout::ROW_MAJOR && !mcast_out;

    if (mcast_out) {
        TT_FATAL(Bc == 1, "batched matmul_decode output_core_grid requires Bc = 1, but got Bc={}", Bc);
    }

    log_debug(
        tt::LogOp,
        "MatmulDecode(batched): batch={}, b_blocks={}, n_blocks={}, Bc={}, Nc={}, M_tiles={}, K_tiles={}, "
        "Nc_tiles={}, num_B_cores={}",
        batch,
        b_blocks,
        n_blocks,
        Bc,
        Nc,
        M_tiles,
        K_tiles,
        Nc_tiles,
        num_B_cores);

    MeshDevice* device = input_tensor_a.device();

    const uint32_t out_block_num_tiles = Bc * M_tiles * Nc_tiles;
    const uint32_t full_in0_num_tiles = Bc * M_tiles * K_tiles;
    const uint32_t block_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core;
    const uint32_t in1_num_tiles = b_shard_K_tiles * Nc_tiles;

    // A pipe entry is `num_k_blocks`-th of a receiver's [Bc*K, Nc] weight slab: a whole number of
    // rows, contiguous in the slab. num_k_blocks == 1 makes the entry the whole slab (the ring must
    // hold a slab); higher values stream the slab in and let the ring be smaller than it. in1 stays
    // tile-paged so the compute kernel indexes tiles within the entry it is holding.
    //
    // The slab stacks Bc batches of K rows, so a page boundary may fall inside a batch or between
    // batches -- the compute kernel walks whatever segments a page happens to contain, and only
    // the row count has to divide evenly.
    const uint32_t num_k_blocks = operation_attributes.prefetch_k_blocks;
    TT_FATAL(
        b_shard_K_tiles % num_k_blocks == 0,
        "batched matmul_decode with prefetch_k_blocks={} requires the weight slab's row count in tiles, Bc*K = {}, "
        "to be divisible by it, because a pipe entry is a whole number of rows of the slab",
        num_k_blocks,
        b_shard_K_tiles);
    // custom_mm reduces a whole batch in one call and needs an even kt_dim, which a batch split
    // across a page boundary could not offer. Pages that divide the slab on batch boundaries keep
    // every batch inside a single page, so the reduction stays whole.
    const bool custom_mm_pages_hold_whole_batches = Bc % num_k_blocks == 0;
    // Each in0 tile row is its own custom_mm call; the reader lays full_in0 out as [Bc][M_tiles][K].
    const bool use_custom_mm = device->arch() == tt::ARCH::BLACKHOLE &&
                               is_custom_mm_in0_tile_height(inputA_tile_height) && is_custom_mm_kt_dim(K_tiles) &&
                               is_custom_mm_ct_dim(Nc_tiles) && custom_mm_pages_hold_whole_batches;
    if (!use_custom_mm) {
        std::string_view reason;
        if (device->arch() != tt::ARCH::BLACKHOLE) {
            reason = "custom_mm is Blackhole-only";
        } else if (!is_custom_mm_in0_tile_height(inputA_tile_height)) {
            reason = "in0 tile height is not in {1, 2, 4, 8}";
        } else if (!is_custom_mm_kt_dim(K_tiles)) {
            reason = "the K dimension must contain an even number of tiles in [2, 256]";
        } else if (!is_custom_mm_ct_dim(Nc_tiles)) {
            reason = "the per-core output width exceeds 16 tiles";
        } else {
            reason =
                "custom_mm reduces a batch in a single call, so prefetch_k_blocks must divide the per-core batch "
                "count for a page to hold whole batches";
        }
        log_warning(tt::LogOp, "matmul_decode is falling back to the general block matmul LLKs: {}", reason);
    }
    // Streaming can complete an output tile across several pages, and on the fallback path the
    // running sum then lives in the out buffer because the packer accumulates into it. A block-float
    // output cannot be read back and added to, so it has to be rejected rather than quietly
    // dropping partial sums. custom_mm is exempt: its pages hold whole batches, so a batch is
    // reduced entirely in DST and the out buffer is written once.
    TT_FATAL(
        num_k_blocks == 1 || use_custom_mm || out_data_format == tt::DataFormat::Float32 ||
            out_data_format == tt::DataFormat::Float16_b || out_data_format == tt::DataFormat::Float16,
        "batched matmul_decode with prefetch_k_blocks={} accumulates partial sums in the out buffer, so the output "
        "dtype must be float32/bfloat16/float16, but it is {}",
        num_k_blocks,
        out_data_format);

    const uint32_t in1_page_num_tiles = in1_num_tiles / num_k_blocks;
    const uint32_t in1_page_bytes = in1_page_num_tiles * in1_tile_size;
    const uint32_t in1_tile_offset = packed.has_value() ? packed->tile_offset : 0;

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    const DFBSpecName FULL_IN0_DFB{"full_in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName OUT_FULL_DFB{"out_full"};
    const DFBSpecName OUT_STAGE_DFB{"out_stage"};

    const SemaphoreSpecName OUT_STAGE_SEM{"out_stage"};
    const SemaphoreSpecName OUT_DONE_SEM{"out_done"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    Group<DataflowBufferSpec> dataflow_buffers;

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FULL_IN0_DFB,
        .entry_size = in0_tile_size,
        .num_entries = full_in0_num_tiles,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    });

    std::vector<PrefetcherPipeParamName> pipe_names;
    Group<PrefetcherPipeParameter> pipe_parameters;
    if (use_pipes) {
        const uint32_t ring_size = prefetcher_pipes.front()->ring_size();
        // Streaming keeps one entry un-acked while the next is published, so the ring has to hold
        // two. With one entry it would deadlock: the reader waits for an entry the prefetcher cannot
        // write until the reader returns the credit it is still holding.
        const uint32_t min_entries = num_k_blocks > 1 ? 2 : 1;
        TT_FATAL(
            ring_size >= min_entries * in1_page_bytes,
            "batched matmul_decode with prefetch_k_blocks={} needs a pipe ring of at least {} entr{} ({} B), but "
            "the ring holds {} B",
            num_k_blocks,
            min_entries,
            min_entries == 1 ? "y" : "ies",
            min_entries * in1_page_bytes,
            ring_size);
        for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
            const PrefetcherPipeParamName name{fmt::format("in1_prefetcher_pipe_{}", p)};
            pipe_names.push_back(name);
            pipe_parameters.push_back(PrefetcherPipeParameter{
                .unique_id = name,
                .receivers = prefetcher_pipes[p]->receiver_cores(),
                .ring_size = ring_size,
                .entry_size = in1_page_bytes,
            });
        }
        // A relay over the rings that pages each entry as its tiles. It covers the ring's whole
        // entries, and an entry's tiles never wrap: the pipe skips any trailing gap.
        const uint32_t relay_size = ring_size - ring_size % in1_page_bytes;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_tile_size,
            .num_entries = relay_size / in1_tile_size,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = inputB_tile,
            .advanced_options = {.prefetcher_pipe_relays = pipe_names},
        });
    } else if (in1_from_dram) {
        // The reader fills this with the core's whole DRAM slab before publishing full_in0.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_tile_size,
            .num_entries = in1_num_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = inputB_tile,
        });
    } else {
        // Borrows the resident weight. For a packed weight the tensor is the fused one, and the
        // region's tile offset into every core's shard is skipped by compute before it reads in1.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_tile_size,
            .num_entries = in1_tile_offset + in1_num_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = inputB_tile,
            .borrowed_from = IN1,
        });
    }

    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = out_tile_size,
        .num_entries = out_block_num_tiles,
        .data_format_metadata = out_data_format,
        .tile_format_metadata = output_tile,
    };
    if (out_is_output) {
        out_dfb.borrowed_from = OUTPUT;
    }
    dataflow_buffers.push_back(std::move(out_dfb));

    const uint32_t packed_N_tiles = b_blocks * div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
    if (mcast_out) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_FULL_DFB,
            .entry_size = out_tile_size,
            .num_entries = M_tiles * packed_N_tiles,
            .data_format_metadata = out_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = OUTPUT,
        });
        // Producers stage their slice here for the hub to multicast. Every output writer binds it, so a
        // producer can derive the hub's staging address from its own write pointer.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_STAGE_DFB,
            .entry_size = out_tile_size,
            .num_entries = M_tiles * packed_N_tiles,
            .data_format_metadata = out_data_format,
            .tile_format_metadata = output_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Geometry
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_senders = in0_rm_hs ? 1u : static_cast<uint32_t>(inputA_core_range_set.num_cores());
    std::vector<uint32_t> sender_phys_coords;
    if (!in0_rm_hs) {
        const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
        sender_phys_coords.reserve(2 * num_senders);
        for (const auto& sender : sender_cores) {
            const CoreCoord phys = device->worker_core_from_logical_core(sender);
            sender_phys_coords.push_back(static_cast<uint32_t>(phys.x));
            sender_phys_coords.push_back(static_cast<uint32_t>(phys.y));
        }
    }

    // Weight slab idx belongs to the idx-th B core in this order (b_idx = idx / n_blocks, n_idx = idx % n_blocks).
    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> b_index_by_core;
    for (uint32_t idx = 0; idx < b_cores.size(); ++idx) {
        b_index_by_core[b_cores[idx]] = idx;
    }

    // Pipe delivery pins the reader to NOC 0: pop_front acks the entry with a non-posted atomic
    // increment into the DRAM sender, and that ack only comes back on NOC 0 -- on NOC 1 the following
    // atomic barrier never drains and the core hangs after the matmul has otherwise finished.
    const NOC reader_noc = use_pipes ? NOC::NOC_0 : NOC::NOC_1;
    const NOC writer_noc = reader_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
    auto dm_config = [](DataMovementProcessor processor, NOC noc) {
        return DataMovementHardwareConfig{
            .config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc},
        };
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    Group<KernelSpec> kernels;
    Group<SemaphoreSpec> semaphores;
    ProgramRunArgs run_args;
    // The kernels each core runs, from which the work units are derived.
    std::map<CoreCoord, Group<KernelSpecName>> kernels_by_core;

    // ---- Reader (RISCV_1) on the weight cores ----
    {
        const KernelSpecName name{"reader"};
        KernelSpec reader{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                "reader_batched_width_sharded.cpp",
            .dfb_bindings = {ProducerOf(FULL_IN0_DFB, "full_in0"), ProducerOf(IN1_DFB, "in1")},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"}},
            .compile_time_args =
                {
                    {"block_slice_tiles", block_slice_tiles},
                    {"tile_size_bytes", in0_tile_size},
                    {"num_senders", num_senders},
                    {"in1_page_tiles", in1_page_num_tiles},
                    {"Bc", Bc},
                    {"inA_K_tiles_per_core", inA_K_tiles_per_core},
                    {"M_tiles", M_tiles},
                    {"batch", batch},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"b_idx"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_1, reader_noc),
        };
        reader.advanced_options.compile_time_varargs = sender_phys_coords;
        auto& defines = reader.compiler_options.defines;
        if (use_pipes) {
            defines.emplace("IN1_PIPE", "1");
            reader.compile_time_args.emplace("num_k_blocks", num_k_blocks);
            reader.advanced_options.prefetcher_pipe_bindings = {
                {.pipe_parameter_names = pipe_names, .accessor_name = "in1"}};
        } else {
            reader.compile_time_args.emplace("in1_tile_offset", in1_tile_offset);
        }
        if (in1_from_dram) {
            defines.emplace("IN1_FROM_DRAM", "1");
            reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"});
            reader.runtime_arg_schema.runtime_arg_names.push_back("in1_shard_id");
        }
        if (use_custom_mm) {
            defines.emplace("USE_CUSTOM_MM", "1");
        }
        if (in0_rm_hs) {
            defines.emplace("IN0_REPLICATED", "1");
        }
        if (out_is_output) {
            defines.emplace("OUT_CONSUMER", "1");
            reader.dfb_bindings.push_back(ConsumerOf(OUT_DFB, "out"));
            reader.compile_time_args.emplace("out_num_tiles", out_block_num_tiles);
        }
        KernelRunArgs reader_run_args{.kernel = name};
        for (const auto& core : b_cores) {
            const uint32_t idx = b_index_by_core.at(core);
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"b_idx", idx / n_blocks}});
            if (in1_from_dram) {
                AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"in1_shard_id", idx}});
            }
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(reader));
        run_args.kernel_run_args.push_back(std::move(reader_run_args));
    }

    // ---- Compute on the weight cores ----
    {
        const KernelSpecName name{"compute"};
        KernelSpec compute{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
                "compute_batched_width_sharded.cpp",
            .dfb_bindings =
                {ConsumerOf(FULL_IN0_DFB, "full_in0"), ConsumerOf(IN1_DFB, "in1"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {
                    {"M_tiles", M_tiles},
                    {"K_tiles", K_tiles},
                    {"Nc_tiles", Nc_tiles},
                    {"Bc", Bc},
                    {"inA_K_tiles_per_core", inA_K_tiles_per_core},
                    {"num_k_blocks", num_k_blocks},
                },
            .hw_config =
                ComputeHardwareConfig{
                    .fpu_math_fidelity = use_custom_mm ? MathFidelity::LoFi : MathFidelity::HiFi4,
                },
        };
        // The legacy builder's ComputeConfig defaults to O3; Metal 2.0's CompilerOptions defaults to O2.
        compute.compiler_options.opt_level = KernelBuildOptLevel::O3;
        auto& defines = compute.compiler_options.defines;
        if (use_custom_mm) {
            defines.emplace("USE_CUSTOM_MM", "1");
        }
        if (use_pipes) {
            defines.emplace("IN1_PIPE", "1");
        } else {
            compute.compile_time_args.emplace("in1_tile_offset", in1_tile_offset);
        }
        for (const auto& core : b_cores) {
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(compute));
    }

    // ---- Writer (RISCV_0) ----
    if (mcast_out) {
        // One multicast sender per NOC: hub0 covers producers [0, split_P), hub1 the rest. With a single
        // hub, split_P spans every producer so hub0 owns the whole N range.
        const uint32_t num_producers = static_cast<uint32_t>(b_cores.size());
        const uint32_t num_out_hubs = mcast_two_hub ? 2u : 1u;
        const uint32_t split_P = mcast_two_hub ? num_producers / 2 : num_producers;
        const CoreRange dest_bbox = output_core_range_set.bounding_box();
        const CoreCoord hub0_logical = dest_bbox.start_coord;
        const CoreCoord hub1_logical = dest_bbox.end_coord;
        TT_FATAL(
            !mcast_two_hub || hub0_logical != hub1_logical,
            "batched matmul_decode output_mcast_two_hub needs an output_core_grid of at least two cores, but got {}",
            output_core_range_set.str());
        const CoreCoord hub0_phys = device->worker_core_from_logical_core(hub0_logical);
        const CoreCoord hub1_phys = device->worker_core_from_logical_core(hub1_logical);
        auto out_role_of = [&](const CoreCoord& core) -> HubRole {
            if (core == hub0_logical) {
                return HubRole::Hub0;
            }
            if (mcast_two_hub && core == hub1_logical) {
                return HubRole::Hub1;
            }
            return HubRole::Plain;
        };

        const CoreRangeSet writer_core_range_set = inputB_core_range_set.merge(output_core_range_set);
        semaphores.push_back(SemaphoreSpec{.unique_id = OUT_STAGE_SEM, .target_nodes = writer_core_range_set});
        semaphores.push_back(SemaphoreSpec{.unique_id = OUT_DONE_SEM, .target_nodes = writer_core_range_set});

        std::map<std::pair<bool, bool>, std::vector<CoreCoord>> writer_groups;
        for (const auto& core : corerange_to_cores(writer_core_range_set, std::nullopt, true)) {
            writer_groups[{inputB_core_range_set.contains(core), output_core_range_set.contains(core)}].push_back(core);
        }
        for (const auto& [key, cores] : writer_groups) {
            const auto& [is_producer, is_dest] = key;
            const KernelSpecName name{
                fmt::format("out_writer{}{}", is_producer ? "_producer" : "", is_dest ? "_dest" : "")};
            KernelSpec writer{
                .unique_id = name,
                .source =
                    "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                    "writer_full_width_output_mcast.cpp",
                .dfb_bindings = {ProducerOf(OUT_STAGE_DFB, "out_stage"), ConsumerOf(OUT_STAGE_DFB, "out_stage")},
                .semaphore_bindings =
                    {SemaphoreBinding{.semaphore_spec_name = OUT_STAGE_SEM, .accessor_name = "out_stage"},
                     SemaphoreBinding{.semaphore_spec_name = OUT_DONE_SEM, .accessor_name = "out_done"}},
                .compile_time_args =
                    {
                        {"M_tiles", M_tiles},
                        {"Nc_tiles", Nc_tiles},
                        {"N_tiles", packed_N_tiles},
                        {"tile_size_bytes", out_tile_size},
                        {"num_dest_cores", static_cast<uint32_t>(output_core_range_set.num_cores())},
                        {"hub0_noc_x", static_cast<uint32_t>(hub0_phys.x)},
                        {"hub0_noc_y", static_cast<uint32_t>(hub0_phys.y)},
                        {"hub1_noc_x", static_cast<uint32_t>(hub1_phys.x)},
                        {"hub1_noc_y", static_cast<uint32_t>(hub1_phys.y)},
                        {"split_P", split_P},
                        {"num_producers", num_producers},
                        {"num_hubs", num_out_hubs},
                    },
                .runtime_arg_schema = {.runtime_arg_names = {"n_idx", "role"}},
                .hw_config = dm_config(DataMovementProcessor::RISCV_0, writer_noc),
            };
            if (is_producer) {
                writer.compiler_options.defines.emplace("IS_PRODUCER", "1");
                writer.dfb_bindings.push_back(ConsumerOf(OUT_DFB, "out"));
            }
            if (is_dest) {
                // Filled by the hubs' multicast; nothing local reads it.
                writer.compiler_options.defines.emplace("IS_DEST", "1");
                writer.dfb_bindings.push_back(ProducerOf(OUT_FULL_DFB, "out_full"));
                writer.dfb_bindings.push_back(ConsumerOf(OUT_FULL_DFB, "out_full"));
            }
            KernelRunArgs writer_run_args{.kernel = name};
            for (const auto& core : cores) {
                const auto it = b_index_by_core.find(core);
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"n_idx", it != b_index_by_core.end() ? it->second : 0u},
                     {"role", static_cast<uint32_t>(out_role_of(core))}});
                kernels_by_core[core].push_back(name);
            }
            kernels.push_back(std::move(writer));
            run_args.kernel_run_args.push_back(std::move(writer_run_args));
        }
    } else if (!out_is_output) {
        const KernelSpecName name{"writer"};
        kernels.push_back(KernelSpec{
            .unique_id = name,
            .source = "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                      "writer_batched_width_sharded.cpp",
            .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args =
                {
                    {"Bc", Bc},
                    {"M_tiles", M_tiles},
                    {"Nc_tiles", Nc_tiles},
                    {"N_tiles", div_up(static_cast<uint32_t>(operation_attributes.N), tt::constants::TILE_WIDTH)},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"b_idx", "n_idx"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_0, writer_noc),
        });
        KernelRunArgs writer_run_args{.kernel = name};
        for (const auto& core : b_cores) {
            const uint32_t idx = b_index_by_core.at(core);
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values, core, {{"b_idx", idx / n_blocks}, {"n_idx", idx % n_blocks}});
            kernels_by_core[core].push_back(name);
        }
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    // Work units may not overlap, so each one is the set of cores that run exactly the same kernels.
    std::map<std::vector<std::string>, std::pair<Group<KernelSpecName>, std::vector<CoreRange>>> work_unit_by_kernels;
    for (const auto& [core, core_kernels] : kernels_by_core) {
        std::vector<std::string> key;
        for (const auto& name : core_kernels) {
            key.push_back(name.get());
        }
        auto& [unit_kernels, unit_cores] = work_unit_by_kernels[key];
        unit_kernels = core_kernels;
        unit_cores.emplace_back(core, core);
    }
    Group<WorkUnitSpec> work_units;
    for (auto& [key, unit] : work_unit_by_kernels) {
        std::string unit_name;
        for (const auto& kernel : key) {
            unit_name += (unit_name.empty() ? "" : "+") + std::string(kernel);
        }
        work_units.push_back(WorkUnitSpec{
            .name = std::move(unit_name),
            .kernels = std::move(unit.first),
            .target_nodes = CoreRangeSet(unit.second),
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensors
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0, .spec = input_tensor_a.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    };
    run_args.tensor_args = {
        {IN0, input_tensor_a.mesh_tensor()},
        {OUTPUT, output_tensor.mesh_tensor()},
    };
    if (!use_pipes) {
        tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = input_tensor_b.tensor_spec()});
        run_args.tensor_args.emplace(IN1, input_tensor_b.mesh_tensor());
    }
    for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
        run_args.advanced_options.prefetcher_pipe_args.emplace(
            pipe_names[p], PrefetcherPipeArgument{*prefetcher_pipes[p]});
    }

    ProgramSpec spec{
        .name = "matmul_decode_batched_width_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
        .advanced_options = {.prefetcher_pipe_parameters = std::move(pipe_parameters)},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts
MatmulDecodeDeviceOperation::BatchedWidthSharded::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto artifacts = create_batched_artifacts(operation_attributes, tensor_args, tensor_return_value);
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : dispatch_ranges(operation_attributes, tensor_coords)) {
        workload.programs.push_back({.range = range, .spec = artifacts.spec, .run_params = artifacts.run_params});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::matmul_decode
