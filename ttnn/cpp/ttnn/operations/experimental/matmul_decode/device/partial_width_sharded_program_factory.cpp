// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "ring_walk.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <map>
#include <memory>
#include <optional>
#include <algorithm>
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
// Ring-gather implementation of PartialWidthSharded. Mirrors the full-width ring gather:
// replaces the two-hub in0 gather with a pipelined closed-ring all-gather over S ∪ C_B.
// The ring reader lands each shard directly at cb_full_in0[sender_id * shard_size], matching the
// sender-major layout compute_partial_width_ring_gather.cpp assumes. L1-resident weights only.
ProgramDescriptor create_descriptor_ring_gather_partial(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value);

// Partial width-sharded: B is sharded along K and N; K-partials reduce onto base cores. The two-hub
// gather of A is the full-width reader's, which this factory shares.
ttnn::device_operation::ProgramArtifacts create_two_hub_partial_artifacts(
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

    log_debug(
        tt::LogOp,
        "MatmulDecode(partial): in0_tile_size: {}, in1_tile_size: {}, out_tile_size: {}",
        in0_tile_size,
        in1_tile_size,
        out_tile_size);

    const uint32_t inputA_tile_height = in0_tile.get_height();
    const uint32_t inputA_tile_width = in0_tile.get_width();
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

    log_debug(tt::LogOp, "MatmulDecode(partial): inputA_tile: {}", in0_tile);

    MeshDevice* device = input_tensor_a.device();

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "partial_width_sharded matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got "
        "M_tiles={} (M={}, inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
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

    const auto& prefetcher_pipes = operation_attributes.prefetcher_pipes;
    const bool use_pipes = !prefetcher_pipes.empty();
    // The reader loads this core's slab of a DRAM ND-sharded weight itself.
    const bool in1_from_dram = operation_attributes.dram_weight_grid.has_value();
    const bool b_nd_sharded = use_pipes || in1_from_dram;
    // A packed weight is a region of a fused height-sharded tensor: its [Kc, Nc] slab shape and
    // its grid come from the spec, since the fused tensor's shard spec describes the whole pack.
    const auto& packed = operation_attributes.packed_weight;
    // A DRAM ND-sharded weight has no legacy shard spec, so the [Kc, Nc] slab shape comes from
    // the ND shard spec and the weight grid from the pipes' receivers or dram_weight_grid.
    const uint32_t Kc = b_nd_sharded         ? static_cast<uint32_t>(input_tensor_b.nd_shard_spec()->shard_shape[-2])
                        : packed.has_value() ? packed->K / packed->k_blocks
                                             : input_tensor_b.memory_config().shard_spec().value().shape[0];
    const uint32_t Nc = b_nd_sharded         ? static_cast<uint32_t>(input_tensor_b.nd_shard_spec()->shard_shape[-1])
                        : packed.has_value() ? packed->N / packed->n_blocks()
                                             : input_tensor_b.memory_config().shard_spec().value().shape[1];
    const uint32_t Kc_tiles = Kc / tt::constants::TILE_HEIGHT;
    const uint32_t Nc_tiles = Nc / tt::constants::TILE_WIDTH;

    const auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto inputB_core_range_set =
        use_pipes
            ? tt::tt_metal::experimental::GetPrefetcherPipeReceiverCores(ttnn::prefetcher_pipe_refs(prefetcher_pipes))
        : in1_from_dram      ? *operation_attributes.dram_weight_grid
        : packed.has_value() ? packed->cores
                             : input_tensor_b.memory_config().shard_spec().value().grid;
    const auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;

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
    TT_FATAL(
        output_core_range_set.num_cores() == N_blocks,
        "Output must be sharded across N_blocks {} cores, but got {}",
        N_blocks,
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

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;  // tiles in one (partial / output) shard

    // A pipe entry is `num_k_blocks`-th of a receiver's [Kc, Nc] weight slab: a whole number of
    // K-rows, contiguous in the slab. num_k_blocks == 1 makes the entry the whole slab (the ring must
    // hold a slab); higher values stream the slab in and let the ring be smaller than it. in1 stays
    // tile-paged so the compute kernel indexes tiles within the entry it is holding.
    const uint32_t num_k_blocks = operation_attributes.prefetch_k_blocks;
    TT_FATAL(
        Kc_tiles % num_k_blocks == 0,
        "partial_width_sharded matmul_decode with prefetch_k_blocks={} requires the per-core Kc in tiles ({}) to be "
        "divisible by it, because a pipe entry is a whole number of K-rows of the weight slab",
        num_k_blocks,
        Kc_tiles);
    const uint32_t in1_k_block_tiles = Kc_tiles / num_k_blocks;
    // Streaming issues one custom_mm_block per pipe entry into the same DST tiles, so an entry's share
    // of Kc -- not just Kc itself -- has to be a legal kt_dim.
    const bool use_custom_mm = device->arch() == tt::ARCH::BLACKHOLE && M_tiles == 1 &&
                               is_custom_mm_in0_tile_height(inputA_tile_height) &&
                               is_custom_mm_kt_dim(in1_k_block_tiles) && is_custom_mm_ct_dim(Nc_tiles);
    if (!use_custom_mm) {
        std::string_view reason;
        if (device->arch() != tt::ARCH::BLACKHOLE) {
            reason = "custom_mm is Blackhole-only";
        } else if (!is_custom_mm_in0_tile_height(inputA_tile_height)) {
            reason = "in0 tile height is not in {1, 2, 4, 8}";
        } else if (M_tiles != 1) {
            reason = "more than one in0 tile row is not contiguous for custom_mm";
        } else if (!is_custom_mm_kt_dim(in1_k_block_tiles)) {
            reason = num_k_blocks == 1
                         ? "the per-core Kc block must contain an even number of tiles in [2, 256]"
                         : "each streamed prefetch entry must contain an even number of K tiles in [2, 256]";
        } else {
            reason = "the per-core output width exceeds 16 tiles";
        }
        log_warning(tt::LogOp, "matmul_decode is falling back to the general block matmul LLKs: {}", reason);
    }
    // Streaming completes each partial tile across several entries, and on the fallback path the
    // running sum lives in the partial buffer because the packer accumulates into it. A block-float
    // format cannot be read back and added to, so it has to be rejected rather than quietly
    // dropping partial sums. custom_mm is exempt: it chains the entries into DST and writes the
    // partial buffer once.
    TT_FATAL(
        num_k_blocks == 1 || use_custom_mm || out_data_format == tt::DataFormat::Float32 ||
            out_data_format == tt::DataFormat::Float16_b || out_data_format == tt::DataFormat::Float16,
        "partial_width_sharded matmul_decode with prefetch_k_blocks={} accumulates partial sums in the partial "
        "buffer, so the output dtype must be float32/bfloat16/float16, but it is {}",
        num_k_blocks,
        out_data_format);

    const uint32_t in1_slab_num_tiles = Kc_tiles * Nc_tiles;
    const uint32_t in1_page_num_tiles = in1_k_block_tiles * Nc_tiles;
    const uint32_t in1_page_bytes = in1_page_num_tiles * in1_tile_size;
    const uint32_t in1_tile_offset = packed.has_value() ? packed->tile_offset : 0;

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    const DFBSpecName FULL_IN0_DFB{"full_in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName PARTIAL_DFB{"partial"};
    const DFBSpecName REDUCE_DFB{"reduce"};
    const DFBSpecName OUT_DFB{"out"};

    const SemaphoreSpecName STAGE_SEM{"stage"};
    const SemaphoreSpecName DONE_SEM{"done"};
    const SemaphoreSpecName REDUCE_SEM{"reduce"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    Group<DataflowBufferSpec> dataflow_buffers;

    // The gathered A. The hubs multicast it over the whole bounding box at their own offset of it, so
    // it is one buffer on every reader core, and its consumer is a compute kernel on all of them.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FULL_IN0_DFB,
        .entry_size = in0_tile_size,
        .num_entries = M_tiles * K_tiles,
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
            "partial_width_sharded matmul_decode with prefetch_k_blocks={} needs a pipe ring of at least {} entr{} "
            "({} B), but the ring holds {} B",
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
            .num_entries = in1_slab_num_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = inputB_tile,
        });
    } else {
        // Borrows the resident weight. For a packed weight the tensor is the fused one, and the
        // region's tile offset into every core's shard is skipped by compute before it reads in1.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_tile_size,
            .num_entries = in1_tile_offset + in1_slab_num_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = inputB_tile,
            .borrowed_from = IN1,
        });
    }

    // This core's [M, Nc] partial product, which the writer sends to its base core.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = PARTIAL_DFB,
        .entry_size = out_tile_size,
        .num_entries = block_num_tiles,
        .data_format_metadata = out_data_format,
        .tile_format_metadata = output_tile,
    });
    // The K_blocks partials of an N block, gathered on its base core. It is one buffer on every weight
    // core, so a writer addresses the base core's copy with its own write pointer.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = REDUCE_DFB,
        .entry_size = out_tile_size,
        .num_entries = K_blocks * block_num_tiles,
        .data_format_metadata = out_data_format,
        .tile_format_metadata = output_tile,
    });
    // The base core's output shard.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = out_tile_size,
        .num_entries = block_num_tiles,
        .data_format_metadata = out_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = OUTPUT,
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      Gather topology
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    const CoreCoord hub0_logical = mcast_bbox.start_coord;
    const CoreCoord hub1_logical = mcast_bbox.end_coord;
    const CoreCoord hub0_phys = device->worker_core_from_logical_core(hub0_logical);
    const CoreCoord hub1_phys = device->worker_core_from_logical_core(hub1_logical);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t split_H = num_senders / 2;

    TT_FATAL(
        num_receivers >= 2 && hub0_logical != hub1_logical,
        "partial_width_sharded matmul_decode two-hub broadcast requires a compute rectangle of at least 2 cores");

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    // Weight slab p belongs to the p-th B core in this order (k_idx = p / N_blocks, n_idx = p % N_blocks).
    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> b_index_by_core;
    for (uint32_t idx = 0; idx < b_cores.size(); ++idx) {
        b_index_by_core[b_cores[idx]] = idx;
        if (idx < N_blocks) {
            TT_FATAL(
                output_core_range_set.contains(b_cores[idx]),
                "Base core {} is not in output core range set",
                b_cores[idx]);
        }
    }

    auto role_of = [&](const CoreCoord& core) -> HubRole {
        if (core == hub0_logical) {
            return HubRole::Hub0;
        }
        if (core == hub1_logical) {
            return HubRole::Hub1;
        }
        return HubRole::Plain;
    };

    // Pick NOC by traffic direction: hub0 mcasts on NOC0, hub1 on NOC1; senders use the NOC toward
    // their hub. Pipe delivery pins every reader to NOC 0: pop_front acks the entry with a non-posted
    // atomic increment into the DRAM sender, and that ack only comes back on NOC 0 -- on NOC 1 the
    // following atomic barrier never drains and the core hangs after the matmul has otherwise
    // finished. The writer must not land on the same NOC as the reader on any core.
    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);
    std::map<CoreCoord, NOC> reader_noc_by_core;
    for (const auto& core : all_reader_cores) {
        const HubRole role = role_of(core);
        const auto it = sender_id_by_core.find(core);
        NOC noc;
        if (use_pipes || role == HubRole::Hub0) {
            noc = NOC::NOC_0;
        } else if (role == HubRole::Hub1 || it == sender_id_by_core.end() || it->second < split_H) {
            noc = NOC::NOC_1;
        } else {
            noc = NOC::NOC_0;
        }
        reader_noc_by_core[core] = noc;
    }
    const auto opposite = [](NOC noc) { return noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0; };

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphores
    ////////////////////////////////////////////////////////////////////////////
    // stage/done: two-hub A gather (see reader). reduce: partial count on the base core.
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = STAGE_SEM, .target_nodes = all_compute_cores_with_bbox},
        SemaphoreSpec{.unique_id = DONE_SEM, .target_nodes = all_compute_cores_with_bbox},
        SemaphoreSpec{.unique_id = REDUCE_SEM, .target_nodes = inputB_core_range_set},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    Group<KernelSpec> kernels;
    ProgramRunArgs run_args;
    // The kernels each core runs, from which the work units are derived.
    std::map<CoreCoord, Group<KernelSpecName>> kernels_by_core;

    // DM kernels come in one KernelSpec per group of cores that share a NOC and a buffer-binding
    // role, because a KernelSpec fixes both.
    auto group_cores = [](const std::vector<CoreCoord>& cores, auto&& key_of) {
        std::map<decltype(key_of(cores.front())), std::vector<CoreCoord>> groups;
        for (const auto& core : cores) {
            groups[key_of(core)].push_back(core);
        }
        return groups;
    };
    auto dm_config = [](DataMovementProcessor processor, NOC noc) {
        return DataMovementHardwareConfig{
            .config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc},
        };
    };

    // ---- Reader (RISCV_1) on the whole bounding box ----
    const KernelSpec::CompileTimeArgs reader_cta = {
        {"shard_num_tiles", shard_num_tiles},
        {"tile_size_bytes", in0_tile_size},
        {"num_senders", num_senders},
        {"num_receivers", num_receivers},
        {"hub0_noc_x", static_cast<uint32_t>(hub0_phys.x)},
        {"hub0_noc_y", static_cast<uint32_t>(hub0_phys.y)},
        {"hub1_noc_x", static_cast<uint32_t>(hub1_phys.x)},
        {"hub1_noc_y", static_cast<uint32_t>(hub1_phys.y)},
        {"split_H", split_H},
        {"in0_M_tiles", M_tiles},
        {"in0_K_tiles", K_tiles},
    };
    const auto reader_groups = group_cores(all_reader_cores, [&](const CoreCoord& core) {
        return std::make_pair(reader_noc_by_core.at(core), inputB_core_range_set.contains(core));
    });
    for (const auto& [key, cores] : reader_groups) {
        const auto& [noc, is_in1_receiver] = key;
        const KernelSpecName name{
            fmt::format("reader_noc{}{}", static_cast<int>(noc), is_in1_receiver ? "_weight" : "_relay")};
        KernelSpec reader{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                "reader_full_width_sharded.cpp",
            .dfb_bindings = {ProducerOf(FULL_IN0_DFB, "full_in0")},
            .semaphore_bindings =
                {SemaphoreBinding{.semaphore_spec_name = STAGE_SEM, .accessor_name = "stage"},
                 SemaphoreBinding{.semaphore_spec_name = DONE_SEM, .accessor_name = "done"}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"}},
            .compile_time_args = reader_cta,
            .runtime_arg_schema = {.runtime_arg_names = {"is_sender", "sender_id", "role"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_1, noc),
        };
        auto& defines = reader.compiler_options.defines;
        if (is_in1_receiver) {
            defines.emplace("IN1_RECEIVER", "1");
            reader.dfb_bindings.push_back(ProducerOf(IN1_DFB, "in1"));
            reader.compile_time_args.emplace("in1_page_tiles", in1_page_num_tiles);
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
        }

        KernelRunArgs reader_run_args{.kernel = name};
        for (const auto& core : cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"is_sender", static_cast<uint32_t>(is_sender)},
                 {"sender_id", is_sender ? it->second : 0u},
                 {"role", static_cast<uint32_t>(role_of(core))}});
            if (is_in1_receiver && in1_from_dram) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values, core, {{"in1_shard_id", b_index_by_core.at(core)}});
            }
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(reader));
        run_args.kernel_run_args.push_back(std::move(reader_run_args));
    }

    // ---- Compute on the weight cores; the base cores also reduce the partials into the output ----
    const auto compute_groups =
        group_cores(b_cores, [&](const CoreCoord& core) { return b_index_by_core.at(core) < N_blocks; });
    for (const auto& [is_base, cores] : compute_groups) {
        const KernelSpecName name{is_base ? "compute_base" : "compute"};
        KernelSpec compute{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
                "compute_partial_width_sharded.cpp",
            .dfb_bindings =
                {ConsumerOf(FULL_IN0_DFB, "full_in0"),
                 ConsumerOf(IN1_DFB, "in1"),
                 ProducerOf(PARTIAL_DFB, "partial"),
                 ConsumerOf(REDUCE_DFB, "reduce")},
            .compile_time_args =
                {
                    {"M_tiles", M_tiles},
                    {"K_tiles", K_tiles},
                    {"Kc_tiles", Kc_tiles},
                    {"Nc_tiles", Nc_tiles},
                    {"K_blocks", K_blocks},
                    {"inA_K_tiles_per_core", inA_K_tiles_per_core},
                    {"num_k_blocks", num_k_blocks},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"k_idx"}},
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
        if (is_base) {
            defines.emplace("IS_BASE", "1");
            compute.dfb_bindings.push_back(ProducerOf(OUT_DFB, "out"));
        }
        KernelRunArgs compute_run_args{.kernel = name};
        for (const auto& core : cores) {
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"k_idx", b_index_by_core.at(core) / N_blocks}});
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(compute));
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }

    // ---- Drain of the gathered A on the bounding-box cores that hold no weight ----
    std::vector<CoreCoord> drain_cores;
    for (const auto& core : all_reader_cores) {
        if (!inputB_core_range_set.contains(core)) {
            drain_cores.push_back(core);
        }
    }
    if (!drain_cores.empty()) {
        const KernelSpecName name{"in0_drain"};
        kernels.push_back(KernelSpec{
            .unique_id = name,
            .source = "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
                      "compute_full_width_in0_drain.cpp",
            .dfb_bindings = {ConsumerOf(FULL_IN0_DFB, "full_in0")},
            .compile_time_args = {{"full_num_tiles", num_senders * shard_num_tiles}},
            .hw_config = ComputeHardwareConfig{},
        });
        for (const auto& core : drain_cores) {
            kernels_by_core[core].push_back(name);
        }
    }

    // ---- Writers (RISCV_0), always on the opposite NOC from this core's reader ----
    const auto writer_groups = group_cores(b_cores, [&](const CoreCoord& core) {
        return std::make_pair(opposite(reader_noc_by_core.at(core)), b_index_by_core.at(core) < N_blocks);
    });
    for (const auto& [key, cores] : writer_groups) {
        const auto& [noc, is_base] = key;
        const KernelSpecName name{fmt::format("writer_noc{}{}", static_cast<int>(noc), is_base ? "_base" : "")};
        KernelSpec writer{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                "writer_partial_width_sharded.cpp",
            .dfb_bindings = {ConsumerOf(PARTIAL_DFB, "partial"), ProducerOf(REDUCE_DFB, "reduce")},
            .semaphore_bindings = {SemaphoreBinding{.semaphore_spec_name = REDUCE_SEM, .accessor_name = "reduce"}},
            .compile_time_args =
                {
                    {"block_num_tiles", block_num_tiles},
                    {"tile_size_bytes", out_tile_size},
                    {"K_blocks", K_blocks},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"k_idx", "base_noc_x", "base_noc_y"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_0, noc),
        };
        if (is_base) {
            writer.compiler_options.defines.emplace("IS_BASE", "1");
            writer.dfb_bindings.push_back(ConsumerOf(OUT_DFB, "out"));
        }
        KernelRunArgs writer_run_args{.kernel = name};
        for (const auto& core : cores) {
            const uint32_t idx = b_index_by_core.at(core);
            const CoreCoord base_phys = device->worker_core_from_logical_core(b_cores[idx % N_blocks]);
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"k_idx", idx / N_blocks},
                 {"base_noc_x", static_cast<uint32_t>(base_phys.x)},
                 {"base_noc_y", static_cast<uint32_t>(base_phys.y)}});
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(writer));
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
        .name = "matmul_decode_partial_width_sharded",
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
MatmulDecodeDeviceOperation::PartialWidthSharded::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto artifacts = create_two_hub_partial_artifacts(operation_attributes, tensor_args, tensor_return_value);
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : dispatch_ranges(operation_attributes, tensor_coords)) {
        workload.programs.push_back({.range = range, .spec = artifacts.spec, .run_params = artifacts.run_params});
    }
    return workload;
}

ProgramDescriptor MatmulDecodeDeviceOperation::PartialWidthRingGather::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.mesh_coords.has_value() &&
        (!mesh_dispatch_coordinate.has_value() ||
         std::find(
             operation_attributes.mesh_coords->begin(),
             operation_attributes.mesh_coords->end(),
             *mesh_dispatch_coordinate) == operation_attributes.mesh_coords->end())) {
        return {};
    }
    const auto& a_grid = tensor_args.input_tensor_a.memory_config().shard_spec().value().grid;
    const auto& b_grid = operation_attributes.packed_weight.has_value()
                             ? operation_attributes.packed_weight->cores
                             : tensor_args.input_tensor_b.memory_config().shard_spec().value().grid;
    log_info(
        tt::LogOp,
        "matmul_decode PartialWidthSharded: in0 gather = ring ({}, S={} cores, C={} cores)",
        a_grid.intersects(b_grid) ? "overlapping S/C" : "disjoint S/C",
        a_grid.num_cores(),
        b_grid.num_cores());
    return create_descriptor_ring_gather_partial(operation_attributes, tensor_args, tensor_return_value);
}

namespace {

using ring_walk::arriving_sender_ids_at;
using ring_walk::build_ring_walk;
using ring_walk::RG_ROLE_IDLE;
using ring_walk::RingWalk;

ProgramDescriptor create_descriptor_ring_gather_partial(
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
    const TileDescriptor in0_tile_desc{in0_tile};
    const TileDescriptor in1_tile_desc{inputB_tile};
    const TileDescriptor out_tile_desc{output_tile};

    const uint32_t inputA_tile_height = in0_tile.get_height();
    TT_FATAL(
        in0_tile.get_width() == tt::constants::TILE_WIDTH && inputB_tile.get_height() == tt::constants::TILE_HEIGHT &&
            inputB_tile.get_width() == tt::constants::TILE_WIDTH &&
            output_tile.get_width() == tt::constants::TILE_WIDTH && inputA_tile_height == output_tile.get_height(),
        "matmul_decode partial ring gather: unexpected tile geometry");

    IDevice* device = input_tensor_a.device();

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "matmul_decode partial ring gather: M_tiles must be <= 8 so the output block fits in DST, but got {}",
        M_tiles);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height) &&
            inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "matmul_decode partial ring gather: input A shard shape must be [M_tiles*tile_h, k * tile_w]");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    const auto& packed = operation_attributes.packed_weight;
    const uint32_t Kc = packed.has_value() ? packed->K / packed->k_blocks
                                           : input_tensor_b.memory_config().shard_spec().value().shape[0];
    const uint32_t Nc = packed.has_value() ? packed->N / packed->n_blocks()
                                           : input_tensor_b.memory_config().shard_spec().value().shape[1];
    const uint32_t Kc_tiles = Kc / tt::constants::TILE_HEIGHT;
    const uint32_t Nc_tiles = Nc / tt::constants::TILE_WIDTH;

    const bool use_custom_mm = false;
    log_warning(
        tt::LogOp,
        "matmul_decode is falling back to the general block matmul LLKs: ring-gather and fused all-gather are not "
        "supported by custom_mm");

    const auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto inputB_core_range_set =
        packed.has_value() ? packed->cores : input_tensor_b.memory_config().shard_spec().value().grid;
    const auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    const uint32_t num_B_cores_along_N = N_tiles / Nc_tiles;
    TT_FATAL(
        num_B_cores % num_B_cores_along_N == 0,
        "matmul_decode partial ring gather: num_B_cores {} must be divisible by num_B_cores_along_N {}",
        num_B_cores,
        num_B_cores_along_N);
    const uint32_t num_B_cores_along_K = num_B_cores / num_B_cores_along_N;
    const uint32_t K_blocks = K_tiles / Kc_tiles;
    TT_FATAL(
        num_B_cores_along_K == K_blocks,
        "matmul_decode partial ring gather: num_B_cores_along_K {} must equal K_blocks {}",
        num_B_cores_along_K,
        K_blocks);
    const uint32_t N_blocks = num_B_cores / K_blocks;
    TT_FATAL(
        output_core_range_set.num_cores() == N_blocks,
        "matmul_decode partial ring gather: output must be sharded across N_blocks {} cores, but got {}",
        N_blocks,
        output_core_range_set.num_cores());

    TT_FATAL(
        inputA_core_range_set.num_cores() > 0 && inputB_core_range_set.num_cores() > 0 &&
            output_core_range_set.num_cores() > 0,
        "matmul_decode partial ring gather: input A / input B / output grids must all be non-empty");

    const std::vector<CoreCoord> S_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, /*row_wise=*/true);
    const std::vector<CoreCoord> C_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, /*row_wise=*/true);
    const uint32_t num_senders = static_cast<uint32_t>(S_cores.size());

    TT_FATAL(
        num_senders == K_tiles / inA_K_tiles_per_core,
        "matmul_decode partial ring gather: sender count ({}) must equal K_tiles / inA_K_tiles_per_core ({} / {})",
        num_senders,
        K_tiles,
        inA_K_tiles_per_core);

    // Phase 1: CW ring only. CCW is a straightforward add-on (second RingWalk over the second
    // half of senders, RISCV_0/NOC_1 reader, an additional cb_full_in0-adjacent CB).
    const RingWalk cw = build_ring_walk(device, /*sources=*/S_cores, /*computes=*/C_cores);

    // Kernels and CBs live on the walk's cores (S ∪ C). Cores between them get nothing.
    std::vector<CoreCoord> S_or_C = S_cores;
    for (const auto& c : C_cores) {
        bool dup = false;
        for (const auto& s : S_cores) {
            if (s == c) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            S_or_C.push_back(c);
        }
    }
    const CoreRangeSet all_ring_cores = CoreRangeSet(S_or_C);
    // Reduce-CB must have identical L1 offset on every core that a base core mcasts partials
    // into (see below). The safest core range is the bbox of {A ∪ B ∪ output}, matching the
    // hub-gather path -- CBs allocated on the same set on all consumer cores get the same L1
    // offset. Ring gather doesn't need reduce_cb on cores outside C_B, but keeping the bbox
    // avoids surprising differences with the hub gather when comparing offsets.
    const auto all_compute_cores_with_bbox =
        CoreRangeSet(inputA_core_range_set.merge(inputB_core_range_set).merge(output_core_range_set).bounding_box());

    ProgramDescriptor desc;

    // CB indices match the hub-gather layout so op fusion sees the same "cb_*" named args.
    const uint32_t in0_cb_index = CBIndex::c_0;       // this core's A slice (source only)
    const uint32_t in1_cb_index = CBIndex::c_1;       // this core's B block (B cores)
    const uint32_t out_cb_index = CBIndex::c_2;       // final output shard (base cores)
    const uint32_t full_in0_cb_index = CBIndex::c_3;  // gathered full A (ring target)
    const uint32_t partial_cb_index = CBIndex::c_4;   // per-core partial product
    const uint32_t reduce_cb_index = CBIndex::c_5;    // gathered K_blocks partials (base cores)

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t full_in0_num_tiles = M_tiles * K_tiles;

    // cb_in0: source-only shard, aliased to input A's L1.
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
        .core_ranges = inputA_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
        .buffer = input_tensor_a.buffer(),
    });

    // cb_in1: this core's [Kc, Nc] weight slab, L1-resident. Packed weight rebases via offset.
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * Nc_tiles * in1_tile_size,
        .core_ranges = inputB_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
        .address_offset = packed.has_value() ? packed->tile_offset * in1_tile_size : 0,
    });

    // cb_out: final output shard on base cores, aliased to the output tensor.
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
        .buffer = output_tensor.buffer(),
    });

    // cb_full_in0: the ring target. Placed on S ∪ C so every hop's write lands at the same L1
    // offset on the successor. compute_partial_width_ring_gather.cpp reads this in sender-major
    // layout (in0_tile = sender * sender_slice_tiles + kc_local), which is exactly how the
    // reader lays it down.
    desc.cbs.push_back(CBDescriptor{
        .total_size = full_in0_num_tiles * in0_tile_size,
        .core_ranges = all_ring_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    // Partial CB: one per B core.
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = inputB_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });

    // Reduce CB: sized for K_blocks partials on base cores. Allocated on the bbox so the
    // writer kernel's cross-core writes hit the same L1 offset on every recipient.
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

    // Ring semaphore (CW) for pipeline sync. Reduce semaphore for K-partial arrival count.
    constexpr uint32_t sig_cw_sem_id = 0;
    constexpr uint32_t reduce_sem_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sig_cw_sem_id,
        .core_ranges = all_ring_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_sem_id,
        .core_ranges = all_compute_cores_with_bbox,
        .initial_value = 0,
    });

    // ---- Reader kernel (CW ring gather) ----
    KernelDescriptor reader_cw;
    reader_cw.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
        "reader_partial_width_ring_gather.cpp";
    reader_cw.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_cw.core_ranges = all_ring_cores;
    const uint32_t in1_slab_num_tiles = Kc_tiles * Nc_tiles;
    reader_cw.compile_time_args = {
        shard_num_tiles,
        in0_tile_size,
        sig_cw_sem_id,
        full_in0_num_tiles,
        in1_slab_num_tiles,
    };
    reader_cw.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_full_in0", full_in0_cb_index},
        {"cb_in1", in1_cb_index},
    };
    reader_cw.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    };

    std::map<CoreCoord, uint32_t> cw_pos;
    for (uint32_t p = 0; p < cw.cores.size(); ++p) {
        cw_pos[cw.cores[p]] = p;
    }

    reader_cw.runtime_args.reserve(S_or_C.size());
    for (const auto& core : S_or_C) {
        const auto it = cw_pos.find(core);
        if (it == cw_pos.end()) {
            // Should not happen: S_or_C is the walk's core set.
            reader_cw.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{RG_ROLE_IDLE, 0, 0, 0, 0, 0});
            continue;
        }
        const uint32_t p = it->second;
        const CoreCoord next = cw.next_phys[p];
        const std::vector<uint32_t> ids = arriving_sender_ids_at(p, cw);
        KernelDescriptor::CoreRuntimeArgs rt;
        rt.reserve(6 + ids.size());
        rt.push_back(cw.role[p]);
        rt.push_back(cw.num_recv[p]);
        rt.push_back(cw.num_sends[p]);
        rt.push_back(static_cast<uint32_t>(next.x));
        rt.push_back(static_cast<uint32_t>(next.y));
        rt.push_back(cw.is_source[p] ? cw.own_sender_id[p] : 0u);
        rt.insert(rt.end(), ids.begin(), ids.end());
        reader_cw.runtime_args.emplace_back(core, std::move(rt));
    }
    desc.kernels.push_back(std::move(reader_cw));

    // ---- Writer kernel (K-reduce; identical to hub-gather path) ----
    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    // Writer runs on RISCV_0. Reader is on RISCV_1/NOC_0, so writer uses NOC_1 to keep the
    // two NoCs balanced. (Hub-gather chooses per-core based on the reader's NOC; here every
    // reader is on NOC_0 so writer uniformly goes to NOC_1.)
    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
        "writer_partial_width_ring_gather.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = CoreRangeSet(b_core_ranges);
    writer.compile_time_args = {
        block_num_tiles,
        out_tile_size,
        K_blocks,
        reduce_sem_id,
    };
    writer.named_compile_time_args = {
        {"cb_partial", partial_cb_index},
        {"cb_reduce", reduce_cb_index},
    };
    writer.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
    };
    writer.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const uint32_t n_idx = idx % N_blocks;
        const CoreCoord base_logical = b_cores[n_idx];
        const CoreCoord base_phys = device->worker_core_from_logical_core(base_logical);
        const bool is_base = (k_idx == 0);
        writer.runtime_args.emplace_back(
            b_cores[idx],
            KernelDescriptor::CoreRuntimeArgs{
                k_idx,
                static_cast<uint32_t>(base_phys.x),
                static_cast<uint32_t>(base_phys.y),
                static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(writer));

    // ---- Compute kernel: the whole resident slab is present up front ----
    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
        "compute_partial_width_ring_gather.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = CoreRangeSet(b_core_ranges);
    compute.compile_time_args = {
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        inA_K_tiles_per_core,
    };
    compute.named_compile_time_args = {
        {"cb_full_in0", full_in0_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_out", out_cb_index},
        {"cb_partial", partial_cb_index},
        {"cb_reduce", reduce_cb_index},
    };
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = use_custom_mm ? MathFidelity::LoFi : MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (use_custom_mm) {
        compute.defines.emplace_back("USE_CUSTOM_MM", "1");
    }
    compute.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const bool is_base = (k_idx == 0);
        if (is_base) {
            TT_FATAL(
                output_core_range_set.contains(b_cores[idx]),
                "matmul_decode partial ring gather: base core {} is not in output core range set",
                b_cores[idx]);
        }
        compute.runtime_args.emplace_back(
            b_cores[idx], KernelDescriptor::CoreRuntimeArgs{k_idx, static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(compute));

    log_debug(
        tt::LogOp,
        "matmul_decode partial ring gather: num_senders={}, num_B_cores={}, M_tiles={}, K_tiles={}, "
        "Kc_tiles={}, Nc_tiles={}, K_blocks={}, N_blocks={}",
        num_senders,
        num_B_cores,
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        N_blocks);

    return desc;
}

}  // namespace

}  // namespace ttnn::operations::experimental::matmul_decode
