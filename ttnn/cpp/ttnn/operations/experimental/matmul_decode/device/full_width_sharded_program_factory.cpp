// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "ring_walk.hpp"
#include "all_gather_writer.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tile.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <map>
#include <memory>
#include <optional>
#include <algorithm>
#include <array>
#include <bit>
#include <set>
#include <string_view>
#include <vector>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::ComputeHardwareConfig;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBEndpointType;
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
struct RmsHubContributor {
    uint32_t producer_index;
    uint32_t producer_scale_slot;
};

struct RmsHubGroup {
    uint32_t group_index;
    uint32_t gathered_slot_base;
    std::vector<RmsHubContributor> contributors;
};

struct RmsLocalGroupFragment {
    uint32_t group_index;
    uint32_t local_tile_offset;
    uint32_t tile_count;
    uint32_t hub_producer_index;
    uint32_t hub_gathered_slot;
};

struct RmsProducerTopology {
    std::vector<RmsLocalGroupFragment> local_fragments;
    std::vector<RmsHubGroup> hub_groups;
    uint32_t hub_contributor_slots = 0;
};

// Ring-gather implementation of FullWidthSharded. Replaces the two-hub gather-then-broadcast
// with a pipelined ring all-gather on in0. It shares its kernels with the fused fabric all-gather,
// which Metal 2.0 cannot express, so it stays a ProgramDescriptor; it is used only when
// `ring_gather` is requested on L1-resident weights (plain or packed).
ProgramDescriptor create_descriptor_ring_gather_full(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const GlobalSemaphore* out_ready_semaphore = nullptr,
    const GlobalSemaphore* barrier_semaphore = nullptr);

// Two-hub full width-sharded: B/output are width(N)-sharded; the reader gathers full A onto every core.
ttnn::device_operation::ProgramArtifacts create_two_hub_artifacts(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const bool in0_rm_hs = operation_attributes.in0_row_major_height_sharded;
    TT_FATAL(
        !in0_rm_hs || !operation_attributes.ring_gather,
        "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A is not supported with ring_gather");

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
        "MatmulDecode: in0_tile_size: {}, in1_tile_size: {}, out_tile_size: {}",
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

    log_debug(tt::LogOp, "MatmulDecode: inputA_tile: {}", in0_tile);

    uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);

    MeshDevice* device = input_tensor_a.device();
    auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto& prefetcher_pipes = operation_attributes.prefetcher_pipes;
    const bool use_pipes = !prefetcher_pipes.empty();
    // The reader loads this core's slab of a DRAM ND-sharded weight itself.
    const bool in1_from_dram = operation_attributes.dram_weight_grid.has_value();
    // A packed weight is a region of a fused height-sharded tensor whose shard spec spans the
    // whole chip, so this weight's grid comes from the spec.
    const auto& packed = operation_attributes.packed_weight;
    // A DRAM ND-sharded weight has no legacy shard spec, so the weight grid is the pipes' receiver
    // set or dram_weight_grid. Otherwise it is the weight's own shard grid.
    auto inputB_core_range_set =
        use_pipes
            ? tt::tt_metal::experimental::GetPrefetcherPipeReceiverCores(ttnn::prefetcher_pipe_refs(prefetcher_pipes))
        : in1_from_dram      ? *operation_attributes.dram_weight_grid
        : packed.has_value() ? packed->cores
                             : input_tensor_b.memory_config().shard_spec().value().grid;
    auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;
    const bool mcast_out = operation_attributes.output_core_grid.has_value();
    const bool mcast_two_hub = mcast_out && operation_attributes.output_mcast_two_hub;
    const bool rms_norm = operation_attributes.rms_norm;
    if (!mcast_out) {
        TT_FATAL(
            inputB_core_range_set == output_core_range_set,
            "Input tensor B and output tensor must have the same core range set");
    }
    if (in0_rm_hs) {
        TT_FATAL(
            inputA_core_range_set.contains(inputB_core_range_set),
            "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A requires A's core grid {} to contain B's core grid {}",
            inputA_core_range_set.str(),
            inputB_core_range_set.str());
        // Extra replica cores (e.g. q_a's 32-core broadcast reused by kv's 16-core weight)
        // already hold A but have no B; run the matmul only on B.
        inputA_core_range_set = inputB_core_range_set;
    }

    auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set).merge(output_core_range_set);
    auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());
    const std::vector<CoreCoord> producer_cores =
        corerange_to_cores(inputB_core_range_set, std::nullopt, /*row_wise=*/true);
    TT_FATAL(!producer_cores.empty(), "full_width_sharded matmul_decode requires at least one producer core");
    const uint32_t num_producers = producer_cores.size();

    log_debug(tt::LogOp, "MatmulDecode: all_compute_cores: {}", all_compute_cores_with_bbox.str());

    std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height),
        "Input tensor A shard shape {} [0] must be equal to M_tiles {} * tile height {}",
        inputA_shard_shape[0],
        M_tiles,
        inputA_tile_height);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A must have a width that is divisible by the tile width");
    if (in0_rm_hs) {
        TT_FATAL(
            inputA_shard_shape[1] == static_cast<uint32_t>(operation_attributes.K),
            "matmul_decode ROW_MAJOR HEIGHT_SHARDED input A requires shard width {} to equal K {}",
            inputA_shard_shape[1],
            operation_attributes.K);
    }
    // Replicated RM A already holds full K on every core. Compute still indexes 1-wide 1x32
    // tiles (K-major after the reader restripe), matching the gathered TILE layout.
    uint32_t inA_K_tiles_per_core = in0_rm_hs ? 1u : inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    uint32_t inB_N_tiles_per_core;
    if (use_pipes || in1_from_dram || packed.has_value()) {
        // Neither a DRAM ND-sharded nor a packed weight has a legacy [K, N/n] shard spec to read
        // the per-core width from; it is N split across the weight's cores by definition.
        const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const uint32_t num_weight_cores = inputB_core_range_set.num_cores();
        TT_FATAL(
            N_tiles % num_weight_cores == 0,
            "full_width_sharded matmul_decode with {} requires N in tiles ({}) to be divisible by the weight core "
            "count ({})",
            use_pipes ? "prefetcher_pipes" : (in1_from_dram ? "a DRAM weight" : "packed_weight"),
            N_tiles,
            num_weight_cores);
        inB_N_tiles_per_core = N_tiles / num_weight_cores;
    } else {
        std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
        TT_FATAL(
            inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT),
            "Input tensor B shard shape {} [0] must be equal to K_tiles {} * tile height {}",
            inputB_shard_shape[0],
            K_tiles,
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
            "Input tensor B must have a width that is divisible by the tile width");
        inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    }

    // RMS groups and B shards are both contiguous along N. Intersecting their tile intervals
    // therefore gives every producer at most one contiguous fragment per group. The first
    // producer touched by a group is its reduction hub.
    uint32_t rms_group_tiles = 0;
    uint32_t rms_num_groups = 0;
    uint32_t rms_max_local_groups = 0;
    uint32_t rms_max_hub_groups = 0;
    uint32_t rms_max_hub_contributors = 0;
    std::vector<RmsProducerTopology> rms_topology(num_producers);
    if (rms_norm) {
        const uint32_t N = static_cast<uint32_t>(operation_attributes.N);
        const uint32_t N_tiles = N / tt::constants::TILE_WIDTH;
        const uint32_t resolved_group_size =
            operation_attributes.rms_norm_group_size == 0 ? N : operation_attributes.rms_norm_group_size;
        rms_group_tiles = resolved_group_size / tt::constants::TILE_WIDTH;
        rms_num_groups = N_tiles / rms_group_tiles;
        TT_FATAL(
            num_producers * inB_N_tiles_per_core == N_tiles,
            "full_width_sharded fused RMSNorm requires producer shards to cover N exactly: {} producers x {} tiles "
            "per producer != {} N tiles",
            num_producers,
            inB_N_tiles_per_core,
            N_tiles);

        for (uint32_t group_index = 0; group_index < rms_num_groups; ++group_index) {
            const uint32_t group_first_tile = group_index * rms_group_tiles;
            const uint32_t group_end_tile = group_first_tile + rms_group_tiles;
            const uint32_t first_producer = group_first_tile / inB_N_tiles_per_core;
            const uint32_t last_producer = (group_end_tile - 1) / inB_N_tiles_per_core;
            auto& hub_topology = rms_topology[first_producer];
            RmsHubGroup hub_group{
                .group_index = group_index,
                .gathered_slot_base = hub_topology.hub_contributor_slots,
            };

            for (uint32_t producer = first_producer; producer <= last_producer; ++producer) {
                const uint32_t producer_first_tile = producer * inB_N_tiles_per_core;
                const uint32_t producer_end_tile = producer_first_tile + inB_N_tiles_per_core;
                const uint32_t fragment_first_tile = std::max(group_first_tile, producer_first_tile);
                const uint32_t fragment_end_tile = std::min(group_end_tile, producer_end_tile);
                TT_FATAL(
                    fragment_first_tile < fragment_end_tile,
                    "full_width_sharded fused RMSNorm derived an empty fragment for group {} and producer {}",
                    group_index,
                    producer);

                auto& producer_topology = rms_topology[producer];
                const uint32_t producer_scale_slot = producer_topology.local_fragments.size();
                const uint32_t contributor_slot = hub_topology.hub_contributor_slots++;
                producer_topology.local_fragments.push_back(RmsLocalGroupFragment{
                    .group_index = group_index,
                    .local_tile_offset = fragment_first_tile - producer_first_tile,
                    .tile_count = fragment_end_tile - fragment_first_tile,
                    .hub_producer_index = first_producer,
                    .hub_gathered_slot = contributor_slot,
                });
                hub_group.contributors.push_back(RmsHubContributor{
                    .producer_index = producer,
                    .producer_scale_slot = producer_scale_slot,
                });
            }
            hub_topology.hub_groups.push_back(std::move(hub_group));
        }

        for (uint32_t producer = 0; producer < num_producers; ++producer) {
            const auto& topology = rms_topology[producer];
            TT_FATAL(
                !topology.local_fragments.empty(),
                "full_width_sharded fused RMSNorm producer {} does not contribute to any group",
                producer);
            rms_max_local_groups =
                std::max(rms_max_local_groups, static_cast<uint32_t>(topology.local_fragments.size()));
            rms_max_hub_groups = std::max(rms_max_hub_groups, static_cast<uint32_t>(topology.hub_groups.size()));
            rms_max_hub_contributors = std::max(rms_max_hub_contributors, topology.hub_contributor_slots);
        }

        log_debug(
            tt::LogOp,
            "MatmulDecode fused RMSNorm topology: group_tiles={}, groups={}, max_local_groups={}, "
            "max_hub_groups={}, max_hub_contributors={}",
            rms_group_tiles,
            rms_num_groups,
            rms_max_local_groups,
            rms_max_hub_groups,
            rms_max_hub_contributors);
    }

    const uint32_t num_k_blocks = operation_attributes.prefetch_k_blocks;
    const uint32_t k_block_tiles = K_tiles / num_k_blocks;
    // Streaming issues one custom_mm_block per pipe entry into the same DST tiles and finalizes on
    // the last, so an entry has to be a legal kt_dim on its own and every output of the row must stay
    // resident in DST across the whole traversal -- which is what restricts it to one tile row.
    const bool in0_rm = input_tensor_a.layout() == Layout::ROW_MAJOR;
    const bool custom_mm_streams_ok = num_k_blocks == 1 || M_tiles == 1;
    const bool use_custom_mm = in0_rm && device->arch() == tt::ARCH::BLACKHOLE && !operation_attributes.all_gather &&
                               custom_mm_streams_ok && is_custom_mm_kt_dim(k_block_tiles) &&
                               is_custom_mm_ct_dim(inB_N_tiles_per_core);
    // custom_mm reads each row's K tiles contiguously, but the gather lays A out sender-major
    // ([sender][M][K / senders]), which is that only for one row. Several rows are gathered into a
    // staging buffer and restriped into full_in0 as [M][K] by the reader.
    const bool restripe_in0 = use_custom_mm && !in0_rm_hs && M_tiles > 1;
    TT_FATAL(
        !in0_rm || in0_rm_hs || M_tiles == 1 || use_custom_mm,
        "matmul_decode ROW_MAJOR width-sharded input A with M={} needs custom_mm (Blackhole, no fused all-gather, "
        "one prefetch K block, an even K block of 2..256 tiles and at most 16 output tiles per core)",
        operation_attributes.M);
    // The RMSNorm statistics want FP32, but custom_mm cannot give DST a 32-bit mode (see the compute
    // config below), and a 16-bit DST cannot read an FP32 buffer: the unpacker consumes each
    // FP32 word as two 16-bit datums, which interleaves the tile instead of converting it. The
    // intermediates therefore have to match the DST width.
    const bool rms_fp32_stats = rms_norm && !use_custom_mm;
    if (!use_custom_mm) {
        std::string_view reason;
        if (!in0_rm) {
            reason = "input A is not ROW_MAJOR";
        } else if (device->arch() != tt::ARCH::BLACKHOLE) {
            reason = "custom_mm is Blackhole-only";
        } else if (operation_attributes.all_gather) {
            reason = "fused all-gather is not supported by custom_mm";
        } else if (!custom_mm_streams_ok) {
            reason =
                "streamed prefetch K blocks need all of a row's outputs to fit in DST, so custom_mm handles them "
                "for a single in0 tile row only";
        } else if (!is_custom_mm_kt_dim(k_block_tiles)) {
            reason = "the K block must contain an even number of tiles in [2, 256]";
        } else {
            reason = "the per-core output width exceeds 16 tiles";
        }
        log_warning(tt::LogOp, "matmul_decode is falling back to the general block matmul LLKs: {}", reason);
    }

    // A pipe entry is `num_k_blocks`-th of a receiver's [K, N/num_receivers] weight slab: a whole
    // number of K-rows, contiguous in the slab. num_k_blocks == 1 makes the entry the whole slab
    // (the ring must hold a slab); higher values stream the slab in and let the ring be smaller
    // than it. in1 stays tile-paged so the compute kernel indexes tiles within the entry it holds.
    TT_FATAL(
        K_tiles % num_k_blocks == 0,
        "full_width_sharded matmul_decode with prefetch_k_blocks={} requires K in tiles ({}) to be divisible by it, "
        "because a pipe entry is a whole number of K-rows of the weight slab",
        num_k_blocks,
        K_tiles);
    // Streaming completes each output tile across several entries, and on the fallback path the
    // running sum lives in the output buffer itself because the packer accumulates into it. A
    // block-float output cannot be read back and added to, so it has to be rejected rather than
    // quietly dropping partial sums. custom_mm is exempt: it chains the entries into DST and writes
    // the output once.
    TT_FATAL(
        num_k_blocks == 1 || use_custom_mm || out_data_format == tt::DataFormat::Float32 ||
            out_data_format == tt::DataFormat::Float16_b || out_data_format == tt::DataFormat::Float16,
        "full_width_sharded matmul_decode with prefetch_k_blocks={} accumulates partial sums in the output buffer, "
        "so the output dtype must be float32/bfloat16/float16, but it is {}",
        num_k_blocks,
        out_data_format);
    const uint32_t in1_k_block_tiles = K_tiles / num_k_blocks;
    const uint32_t in1_slab_num_tiles = K_tiles * inB_N_tiles_per_core;
    const uint32_t in1_page_num_tiles = in1_k_block_tiles * inB_N_tiles_per_core;
    const uint32_t in1_page_bytes = in1_page_num_tiles * in1_tile_size;
    const uint32_t in1_tile_offset = packed.has_value() ? packed->tile_offset : 0;

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName GAMMA{"rms_gamma"};

    const DFBSpecName FULL_IN0_DFB{"full_in0"};
    const DFBSpecName GATHER_IN0_DFB{"gather_in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName MM_OUT_DFB{"mm_out"};
    const DFBSpecName RMS_GAMMA_DFB{"rms_gamma"};
    const DFBSpecName RMS_LOCAL_DFB{"rms_local"};
    const DFBSpecName RMS_GATHERED_DFB{"rms_gathered"};
    const DFBSpecName RMS_REDUCE_SCALER_DFB{"rms_reduce_scaler"};
    const DFBSpecName RMS_SCALE_SRC_DFB{"rms_scale_src"};
    const DFBSpecName RMS_SCALE_DFB{"rms_scale"};
    const DFBSpecName OUT_FULL_DFB{"out_full"};
    const DFBSpecName OUT_STAGE_DFB{"out_stage"};

    const SemaphoreSpecName STAGE_SEM{"stage"};
    const SemaphoreSpecName DONE_SEM{"done"};
    const SemaphoreSpecName RMS_ARRIVAL_SEM{"rms_arrival"};
    const SemaphoreSpecName RMS_SCALE_READY_SEM{"rms_scale_ready"};
    const SemaphoreSpecName OUT_STAGE_SEM{"out_stage"};
    const SemaphoreSpecName OUT_DONE_SEM{"out_done"};

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
    if (restripe_in0) {
        // The gather's destination: same offset on every core, so a sender addresses the hub's copy
        // with its own write pointer, as it does full_in0 otherwise.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = GATHER_IN0_DFB,
            .entry_size = in0_tile_size,
            .num_entries = M_tiles * K_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

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
            "full_width_sharded matmul_decode with prefetch_k_blocks={} needs a pipe ring of at least {} entr{} "
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

    // This core's [M, Nc] output slice: the output shard itself unless it is replicated, when it is
    // staged here for the output writer.
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = out_tile_size,
        .num_entries = M_tiles * inB_N_tiles_per_core,
        .data_format_metadata = out_data_format,
        .tile_format_metadata = output_tile,
    };
    if (!mcast_out) {
        out_dfb.borrowed_from = OUTPUT;
    }
    dataflow_buffers.push_back(std::move(out_dfb));

    const tt::DataFormat rms_data_format = rms_fp32_stats ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t rms_tile_size = output_tile.get_tile_size(rms_data_format);
    const tt::tt_metal::Tile rms_reduce_tile({tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
    const uint32_t rms_reduce_tile_size = rms_reduce_tile.get_tile_size(rms_data_format);
    const bool rms_vector_gamma = rms_norm && tensor_args.rms_norm_gamma.has_value();
    if (rms_norm) {
        TT_FATAL(
            rms_reduce_tile_size % rms_tile_size == 0,
            "matmul_decode fused RMSNorm requires the local statistic page size {} to divide the reduction tile "
            "size {}",
            rms_tile_size,
            rms_reduce_tile_size);

        // The matmul result, which the epilogue normalizes into out.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = MM_OUT_DFB,
            .entry_size = out_tile_size,
            .num_entries = M_tiles * inB_N_tiles_per_core,
            .data_format_metadata = out_data_format,
            .tile_format_metadata = output_tile,
        });
        if (rms_vector_gamma) {
            const auto& gamma = tensor_args.rms_norm_gamma.value();
            const tt::DataFormat gamma_data_format = datatype_to_dataformat_converter(gamma.dtype());
            const auto& gamma_tile = gamma.tensor_spec().tile();
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = RMS_GAMMA_DFB,
                .entry_size = gamma_tile.get_tile_size(gamma_data_format),
                .num_entries = inB_N_tiles_per_core,
                .data_format_metadata = gamma_data_format,
                .tile_format_metadata = gamma_tile,
                .borrowed_from = GAMMA,
            });
        }
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RMS_LOCAL_DFB,
            .entry_size = rms_tile_size,
            .num_entries = M_tiles * rms_max_local_groups,
            .data_format_metadata = rms_data_format,
            .tile_format_metadata = output_tile,
        });
        // Producers derive each hub destination from their local write pointer, which holds because a
        // buffer sits at one address on every core it lives on.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RMS_GATHERED_DFB,
            .entry_size = rms_reduce_tile_size,
            .num_entries = M_tiles * rms_max_hub_contributors,
            .data_format_metadata = rms_data_format,
            .tile_format_metadata = rms_reduce_tile,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RMS_REDUCE_SCALER_DFB,
            .entry_size = rms_reduce_tile_size,
            .num_entries = 1,
            .data_format_metadata = rms_data_format,
            .tile_format_metadata = rms_reduce_tile,
        });
        // A group hub's scales, from compute to the writer RISC, which unicasts them out of here into
        // rms_scale, so the hub never has to consume the buffer its own compute is waiting on. Every
        // producer holds one, so that compute and the writer bind the same buffers on every producer.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RMS_SCALE_SRC_DFB,
            .entry_size = rms_tile_size,
            .num_entries = M_tiles * rms_max_hub_groups,
            .data_format_metadata = rms_data_format,
            .tile_format_metadata = output_tile,
        });
        // Every hub unicasts scales into contributor-local slots, deriving each remote destination
        // from its own write pointer.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RMS_SCALE_DFB,
            .entry_size = rms_tile_size,
            .num_entries = M_tiles * rms_max_local_groups,
            .data_format_metadata = rms_data_format,
            .tile_format_metadata = output_tile,
        });
    }
    const uint32_t N_tiles = div_up(static_cast<uint32_t>(operation_attributes.N), tt::constants::TILE_WIDTH);
    if (mcast_out) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_FULL_DFB,
            .entry_size = out_tile_size,
            .num_entries = M_tiles * N_tiles,
            .data_format_metadata = out_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = OUTPUT,
        });
        // Producers stage their slice here for the hub to multicast. Every output writer binds it, so a
        // producer can derive the hub's staging address from its own write pointer.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_STAGE_DFB,
            .entry_size = out_tile_size,
            .num_entries = M_tiles * N_tiles,
            .data_format_metadata = out_data_format,
            .tile_format_metadata = output_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Geometry
    ////////////////////////////////////////////////////////////////////////////
    // stage: senders bump owning hub after slice write; done: each hub bumps all cores after its mcast half.
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    log_debug(tt::LogOp, "MatmulDecode: num_senders: {}", num_senders);

    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    const CoreCoord hub0_logical = mcast_bbox.start_coord;
    const CoreCoord hub1_logical = mcast_bbox.end_coord;
    const CoreCoord hub0_phys = device->worker_core_from_logical_core(hub0_logical);
    const CoreCoord hub1_phys = device->worker_core_from_logical_core(hub1_logical);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t split_H = num_senders / 2;

    // A replicated A skips the gather (see the reader's IN0_REPLICATED path), so one core is enough.
    TT_FATAL(
        in0_rm_hs || (num_receivers >= 2 && hub0_logical != hub1_logical),
        "full_width_sharded matmul_decode two-hub broadcast requires a compute rectangle of at least 2 cores");
    TT_FATAL(
        M_tiles <= 8,
        "full_width_sharded matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got M_tiles={} "
        "(M={}, inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    std::map<CoreCoord, uint32_t> producer_id_by_core;
    for (uint32_t id = 0; id < producer_cores.size(); ++id) {
        producer_id_by_core[producer_cores[id]] = id;
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
    // finished. This costs the two-hub gather its NOC split (both hubs mcast on NOC 0) on that path.
    // The writer must not land on the same NOC as the reader on any core, so remember the choice.
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

    // Runtime RMS topology, as varargs:
    //   [local_group_count, hub_group_count, hub_contributor_count]
    //   local group: [local_tile_offset, tile_count, hub_x, hub_y, hub_gathered_slot, hub_row_stride]
    //   hub group:   [contributor_count, gathered_slot_base,
    //                 contributor_x, contributor_y, contributor_scale_slot, contributor_row_stride, ...]
    // Group and contributor lists are contiguous in global-N order, so indices need not be sent. Every
    // producer's list is padded to the longest, since a kernel takes one vararg count.
    std::vector<std::vector<uint32_t>> rms_topology_varargs(num_producers);
    uint32_t rms_num_varargs = 0;
    if (rms_norm) {
        for (uint32_t producer = 0; producer < num_producers; ++producer) {
            const auto& topology = rms_topology[producer];
            auto& args = rms_topology_varargs[producer];
            args.push_back(static_cast<uint32_t>(topology.local_fragments.size()));
            args.push_back(static_cast<uint32_t>(topology.hub_groups.size()));
            args.push_back(topology.hub_contributor_slots);
            for (const auto& fragment : topology.local_fragments) {
                const CoreCoord hub_phys =
                    device->worker_core_from_logical_core(producer_cores[fragment.hub_producer_index]);
                args.push_back(fragment.local_tile_offset);
                args.push_back(fragment.tile_count);
                args.push_back(hub_phys.x);
                args.push_back(hub_phys.y);
                args.push_back(fragment.hub_gathered_slot);
                args.push_back(rms_topology[fragment.hub_producer_index].hub_contributor_slots);
            }
            for (const auto& hub_group : topology.hub_groups) {
                args.push_back(static_cast<uint32_t>(hub_group.contributors.size()));
                args.push_back(hub_group.gathered_slot_base);
                for (const auto& contributor : hub_group.contributors) {
                    const CoreCoord contributor_phys =
                        device->worker_core_from_logical_core(producer_cores[contributor.producer_index]);
                    args.push_back(contributor_phys.x);
                    args.push_back(contributor_phys.y);
                    args.push_back(contributor.producer_scale_slot);
                    args.push_back(
                        static_cast<uint32_t>(rms_topology[contributor.producer_index].local_fragments.size()));
                }
            }
            rms_num_varargs = std::max(rms_num_varargs, static_cast<uint32_t>(args.size()));
        }
        for (auto& args : rms_topology_varargs) {
            args.resize(rms_num_varargs, 0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphores
    ////////////////////////////////////////////////////////////////////////////
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = STAGE_SEM, .target_nodes = all_compute_cores_with_bbox},
        SemaphoreSpec{.unique_id = DONE_SEM, .target_nodes = all_compute_cores_with_bbox},
    };
    if (rms_norm) {
        semaphores.push_back(SemaphoreSpec{.unique_id = RMS_ARRIVAL_SEM, .target_nodes = inputB_core_range_set});
        semaphores.push_back(SemaphoreSpec{.unique_id = RMS_SCALE_READY_SEM, .target_nodes = inputB_core_range_set});
    }
    // Hubs live in the dest bbox and need not be producers, so the output writer covers both sets.
    const CoreRangeSet writer_core_range_set = inputB_core_range_set.merge(output_core_range_set);
    if (mcast_out) {
        semaphores.push_back(SemaphoreSpec{.unique_id = OUT_STAGE_SEM, .target_nodes = writer_core_range_set});
        semaphores.push_back(SemaphoreSpec{.unique_id = OUT_DONE_SEM, .target_nodes = writer_core_range_set});
    }

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
        if (restripe_in0) {
            // Written by the hubs' multicast and read back by this core alone.
            reader.dfb_bindings.push_back(ProducerOf(GATHER_IN0_DFB, "gather_in0"));
            reader.dfb_bindings.push_back(ConsumerOf(GATHER_IN0_DFB, "gather_in0"));
            defines.emplace("IN0_ROW_RESTRIPE", "1");
        }
        if (in0_rm_hs) {
            defines.emplace("IN0_REPLICATED", "1");
        }
        if (use_custom_mm) {
            defines.emplace("USE_CUSTOM_MM", "1");
        }
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
            if (!mcast_out) {
                defines.emplace("OUT_CONSUMER", "1");
                reader.dfb_bindings.push_back(ConsumerOf(OUT_DFB, "out"));
                reader.compile_time_args.emplace("out_num_tiles", M_tiles * inB_N_tiles_per_core);
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
                    reader_run_args.runtime_arg_values, core, {{"in1_shard_id", producer_id_by_core.at(core)}});
            }
            kernels_by_core[core].push_back(name);
        }
        kernels.push_back(std::move(reader));
        run_args.kernel_run_args.push_back(std::move(reader_run_args));
    }

    log_debug(
        tt::LogOp,
        "MatmulDecode: M_tiles: {}, K_tiles: {}, inB_N_tiles_per_core: {}, inA_K_tiles_per_core: {}",
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core);

    // ---- Compute on the weight cores ----
    {
        const KernelSpecName name{"compute"};
        KernelSpec compute{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
                "compute_full_width_sharded.cpp",
            .dfb_bindings =
                {ConsumerOf(FULL_IN0_DFB, "full_in0"), ConsumerOf(IN1_DFB, "in1"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {
                    {"M_tiles", M_tiles},
                    {"K_tiles", K_tiles},
                    {"N_tiles_per_core", inB_N_tiles_per_core},
                    {"inA_K_tiles_per_core", inA_K_tiles_per_core},
                    {"num_k_blocks", num_k_blocks},
                },
            .hw_config =
                ComputeHardwareConfig{
                    .fpu_math_fidelity = use_custom_mm ? MathFidelity::LoFi : MathFidelity::HiFi4,
                    // The RMSNorm epilogue accumulates a sum of squares in DST, so it prefers a 32-bit
                    // DST. custom_mm cannot offer one: its split_acc finalization folds partials back
                    // through MOVD2B and its dense_packing halves the DST tile stride to 32 rows, and
                    // both encode 16-bit DST geometry. custom_mm therefore wins, and the statistics run
                    // in BF16 (see rms_fp32_stats, which keeps the intermediates in step with the DST
                    // width).
                    .enable_32_bit_dest = rms_fp32_stats,
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
        KernelRunArgs compute_run_args{.kernel = name};
        if (rms_norm) {
            defines.emplace("FUSE_RMS_NORM", "1");
            // TODO(debug): temporary. "local" dumps this core's partial mean of squares to the output,
            // "scale" dumps the multicast scale, so the stages can be read back from host.
            if (const char* dump = std::getenv("TT_RMS_DEBUG_DUMP"); dump != nullptr) {
                if (std::string_view{dump} == "copy") {
                    defines.emplace("RMS_DEBUG_DUMP_COPY", "1");
                } else if (std::string_view{dump} == "local") {
                    defines.emplace("RMS_DEBUG_DUMP_LOCAL", "1");
                } else if (std::string_view{dump} == "scale") {
                    defines.emplace("RMS_DEBUG_DUMP_SCALE", "1");
                }
            }
            compute.dfb_bindings.push_back(ProducerOf(MM_OUT_DFB, "mm_out"));
            compute.dfb_bindings.push_back(ConsumerOf(MM_OUT_DFB, "mm_out"));
            compute.dfb_bindings.push_back(ProducerOf(RMS_LOCAL_DFB, "rms_local"));
            compute.dfb_bindings.push_back(ConsumerOf(RMS_GATHERED_DFB, "rms_gathered"));
            compute.dfb_bindings.push_back(ProducerOf(RMS_SCALE_SRC_DFB, "rms_scale_src"));
            compute.dfb_bindings.push_back(ConsumerOf(RMS_SCALE_DFB, "rms_scale"));
            if (rms_vector_gamma) {
                defines.emplace("FUSE_RMS_VECTOR_GAMMA", "1");
                compute.dfb_bindings.push_back(ConsumerOf(RMS_GAMMA_DFB, "rms_gamma"));
            }
            compute.compile_time_args.emplace("rms_num_groups", rms_num_groups);
            compute.compile_time_args.emplace(
                "rms_inv_group_size_bits",
                std::bit_cast<uint32_t>(1.0F / static_cast<float>(rms_group_tiles * tt::constants::TILE_WIDTH)));
            compute.compile_time_args.emplace(
                "rms_epsilon_bits", std::bit_cast<uint32_t>(operation_attributes.rms_norm_epsilon));
            compute.compile_time_args.emplace(
                "rms_gamma_bits", std::bit_cast<uint32_t>(operation_attributes.rms_norm_gamma.value_or(1.0F)));
            compute.advanced_options.num_runtime_varargs = rms_num_varargs;
            for (uint32_t producer = 0; producer < num_producers; ++producer) {
                compute_run_args.advanced_options.runtime_varargs[producer_cores[producer]] =
                    rms_topology_varargs[producer];
            }
        }
        // Metal 2.0 requires an unpack mode for every Float32 buffer a compute kernel consumes with
        // 32-bit DEST; the legacy default unpacks through SrcA/B.
        auto& compute_hw = std::get<ComputeHardwareConfig>(compute.hw_config);
        if (compute_hw.enable_32_bit_dest) {
            for (const auto& binding : compute.dfb_bindings) {
                const auto dfb = std::find_if(dataflow_buffers.begin(), dataflow_buffers.end(), [&](const auto& d) {
                    return d.unique_id == binding.dfb_spec_name;
                });
                if (binding.endpoint_type == DFBEndpointType::CONSUMER &&
                    dfb->data_format_metadata == tt::DataFormat::Float32) {
                    compute_hw.unpack_modes.emplace(binding.dfb_spec_name, UnpackMode::UnpackToSrc);
                }
            }
        }
        for (const auto& core : producer_cores) {
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
            .compile_time_args = {{"full_num_tiles", in0_rm_hs ? M_tiles * K_tiles : num_senders * shard_num_tiles}},
            .hw_config = ComputeHardwareConfig{},
        });
        for (const auto& core : drain_cores) {
            kernels_by_core[core].push_back(name);
        }
    }

    // ---- Writers (RISCV_0), always on the opposite NOC from this core's reader ----
    // BRISC and NCRISC track their non-posted writes in separate counters but share the NIU's ack
    // register, and the "expected acks" bookkeeping is a non-atomic read-modify-write. Two RISCs
    // issuing on one NOC therefore race and can drop an increment, leaving the expected count below
    // the acks that arrive; since the barrier tests for equality it then spins forever. This showed up
    // as a single core out of 32 stuck in NWBW. Putting both on one NOC for the RMS transport instead
    // tripped the firmware's end-of-kernel "non-posted writes sent" assert. The kernels are
    // NOC-agnostic: they flip the multicast rectangle's corners on NOC 1.
    const uint32_t rms_local_tile_size = rms_tile_size;
    auto add_rms_transport = [&](KernelSpec& writer,
                                 KernelRunArgs& writer_run_args,
                                 const std::vector<CoreCoord>& cores) {
        writer.dfb_bindings.push_back(ConsumerOf(RMS_LOCAL_DFB, "rms_local"));
        writer.dfb_bindings.push_back(ProducerOf(RMS_GATHERED_DFB, "rms_gathered"));
        writer.dfb_bindings.push_back(ConsumerOf(RMS_SCALE_SRC_DFB, "rms_scale_src"));
        writer.dfb_bindings.push_back(ProducerOf(RMS_SCALE_DFB, "rms_scale"));
        writer.dfb_bindings.push_back(ProducerOf(RMS_REDUCE_SCALER_DFB, "rms_reduce_scaler"));
        writer.dfb_bindings.push_back(ConsumerOf(RMS_REDUCE_SCALER_DFB, "rms_reduce_scaler"));
        writer.semaphore_bindings.push_back(
            SemaphoreBinding{.semaphore_spec_name = RMS_ARRIVAL_SEM, .accessor_name = "rms_arrival"});
        writer.semaphore_bindings.push_back(
            SemaphoreBinding{.semaphore_spec_name = RMS_SCALE_READY_SEM, .accessor_name = "rms_scale_ready"});
        writer.compile_time_args.emplace("rms_m_tiles", M_tiles);
        writer.compile_time_args.emplace("rms_local_tile_size", rms_local_tile_size);
        writer.compile_time_args.emplace("rms_reduce_tile_size", rms_reduce_tile_size);
        if (rms_vector_gamma) {
            // gamma is resident in the tensor it borrows and compute reads it in place, so this
            // binding only gives the buffer the producer it must have.
            writer.dfb_bindings.push_back(ProducerOf(RMS_GAMMA_DFB, "rms_gamma"));
        }
        writer.advanced_options.num_runtime_varargs = rms_num_varargs;
        for (const auto& core : cores) {
            writer_run_args.advanced_options.runtime_varargs[core] = rms_topology_varargs[producer_id_by_core.at(core)];
        }
    };

    if (rms_norm && !mcast_out) {
        const auto writer_groups =
            group_cores(producer_cores, [&](const CoreCoord& core) { return opposite(reader_noc_by_core.at(core)); });
        for (const auto& [noc, cores] : writer_groups) {
            const KernelSpecName name{fmt::format("rms_writer_noc{}", static_cast<int>(noc))};
            KernelSpec writer{
                .unique_id = name,
                .source =
                    "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
                    "writer_full_width_rms_norm.cpp",
                .hw_config = dm_config(DataMovementProcessor::RISCV_0, noc),
            };
            KernelRunArgs writer_run_args{.kernel = name};
            add_rms_transport(writer, writer_run_args, cores);
            for (const auto& core : cores) {
                kernels_by_core[core].push_back(name);
            }
            kernels.push_back(std::move(writer));
            run_args.kernel_run_args.push_back(std::move(writer_run_args));
        }
    }

    if (mcast_out) {
        // One multicast sender per NOC: hub0 on NOC0 covers producers [0, split_P), hub1 on NOC1 the
        // rest. With a single hub, split_P spans every producer so hub0 owns the whole N range.
        const uint32_t num_out_hubs = mcast_two_hub ? 2u : 1u;
        const uint32_t split_P = mcast_two_hub ? num_producers / 2 : num_producers;
        const CoreRange dest_bbox = output_core_range_set.bounding_box();
        const CoreCoord out_hub0_logical = dest_bbox.start_coord;
        const CoreCoord out_hub1_logical = dest_bbox.end_coord;
        // Both hubs come from the dest bbox corners, so a one-core grid collapses them into the
        // same core: it would be tagged Hub0 and hub1's half of N would never be written.
        TT_FATAL(
            !mcast_two_hub || out_hub0_logical != out_hub1_logical,
            "matmul_decode output_mcast_two_hub needs an output_core_grid of at least two cores, but got {}",
            output_core_range_set.str());
        const CoreCoord out_hub0_phys = device->worker_core_from_logical_core(out_hub0_logical);
        const CoreCoord out_hub1_phys = device->worker_core_from_logical_core(out_hub1_logical);
        auto out_role_of = [&](const CoreCoord& core) -> HubRole {
            if (core == out_hub0_logical) {
                return HubRole::Hub0;
            }
            if (mcast_two_hub && core == out_hub1_logical) {
                return HubRole::Hub1;
            }
            return HubRole::Plain;
        };

        const std::vector<CoreCoord> writer_cores = corerange_to_cores(writer_core_range_set, std::nullopt, true);
        for (const auto& core : writer_cores) {
            TT_FATAL(
                reader_noc_by_core.contains(core),
                "full_width_sharded matmul_decode output mcast: writer core {} has no reader, so its NOC cannot be "
                "chosen opposite the reader's",
                core.str());
        }
        const auto writer_groups = group_cores(writer_cores, [&](const CoreCoord& core) {
            return std::make_tuple(
                opposite(reader_noc_by_core.at(core)),
                inputB_core_range_set.contains(core),
                output_core_range_set.contains(core));
        });
        for (const auto& [key, cores] : writer_groups) {
            const auto& [noc, is_producer, is_dest] = key;
            const KernelSpecName name{fmt::format(
                "out_writer_noc{}{}{}", static_cast<int>(noc), is_producer ? "_producer" : "", is_dest ? "_dest" : "")};
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
                        {"Nc_tiles", inB_N_tiles_per_core},
                        {"N_tiles", N_tiles},
                        {"tile_size_bytes", out_tile_size},
                        {"num_dest_cores", static_cast<uint32_t>(output_core_range_set.num_cores())},
                        {"hub0_noc_x", static_cast<uint32_t>(out_hub0_phys.x)},
                        {"hub0_noc_y", static_cast<uint32_t>(out_hub0_phys.y)},
                        {"hub1_noc_x", static_cast<uint32_t>(out_hub1_phys.x)},
                        {"hub1_noc_y", static_cast<uint32_t>(out_hub1_phys.y)},
                        {"split_P", split_P},
                        {"num_producers", num_producers},
                        {"num_hubs", num_out_hubs},
                    },
                .runtime_arg_schema = {.runtime_arg_names = {"n_idx", "role"}},
                .hw_config = dm_config(DataMovementProcessor::RISCV_0, noc),
            };
            KernelRunArgs writer_run_args{.kernel = name};
            if (is_producer) {
                writer.compiler_options.defines.emplace("IS_PRODUCER", "1");
                writer.dfb_bindings.push_back(ConsumerOf(OUT_DFB, "out"));
                if (rms_norm) {
                    writer.compiler_options.defines.emplace("FUSE_RMS_NORM", "1");
                    add_rms_transport(writer, writer_run_args, cores);
                }
            }
            if (is_dest) {
                // Filled by the hubs' multicast; nothing local reads it.
                writer.compiler_options.defines.emplace("IS_DEST", "1");
                writer.dfb_bindings.push_back(ProducerOf(OUT_FULL_DFB, "out_full"));
                writer.dfb_bindings.push_back(ConsumerOf(OUT_FULL_DFB, "out_full"));
            }
            for (const auto& core : cores) {
                const auto it = producer_id_by_core.find(core);
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"n_idx", it != producer_id_by_core.end() ? it->second : 0u},
                     {"role", static_cast<uint32_t>(out_role_of(core))}});
                kernels_by_core[core].push_back(name);
            }
            kernels.push_back(std::move(writer));
            run_args.kernel_run_args.push_back(std::move(writer_run_args));
        }
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
    if (rms_vector_gamma) {
        const auto& gamma = tensor_args.rms_norm_gamma.value();
        tensor_parameters.push_back(TensorParameter{.unique_id = GAMMA, .spec = gamma.tensor_spec()});
        run_args.tensor_args.emplace(GAMMA, gamma.mesh_tensor());
    }
    for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
        run_args.advanced_options.prefetcher_pipe_args.emplace(
            pipe_names[p], PrefetcherPipeArgument{*prefetcher_pipes[p]});
    }

    ProgramSpec spec{
        .name = "matmul_decode_full_width_sharded",
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

std::vector<ttnn::MeshCoordinateRange> dispatch_ranges(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    if (!operation_attributes.mesh_coords.has_value()) {
        return tensor_coords.ranges();
    }
    std::vector<ttnn::MeshCoordinateRange> ranges;
    for (const auto& coord : tensor_coords.coords()) {
        if (std::find(operation_attributes.mesh_coords->begin(), operation_attributes.mesh_coords->end(), coord) !=
            operation_attributes.mesh_coords->end()) {
            ranges.emplace_back(coord);
        }
    }
    return ranges;
}

ttnn::device_operation::MeshWorkloadArtifacts
MatmulDecodeDeviceOperation::FullWidthSharded::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto artifacts = create_two_hub_artifacts(operation_attributes, tensor_args, tensor_return_value);
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : dispatch_ranges(operation_attributes, tensor_coords)) {
        workload.programs.push_back({.range = range, .spec = artifacts.spec, .run_params = artifacts.run_params});
    }
    return workload;
}

ProgramDescriptor MatmulDecodeDeviceOperation::FullWidthRingGather::create_descriptor(
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
        "matmul_decode FullWidthSharded: in0 gather = ring ({}, S={} cores, C={} cores)",
        a_grid.intersects(b_grid) ? "overlapping S/C" : "disjoint S/C",
        a_grid.num_cores(),
        b_grid.num_cores());
    return create_descriptor_ring_gather_full(
        operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
}

namespace {

using ring_walk::arriving_sender_ids_at;
using ring_walk::build_ring_walk;
using ring_walk::RG_ROLE_IDLE;
using ring_walk::RingWalk;

ProgramDescriptor create_descriptor_ring_gather_full(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const GlobalSemaphore* out_ready_semaphore,
    const GlobalSemaphore* barrier_semaphore) {
    log_warning(
        tt::LogOp,
        "matmul_decode is falling back to the general block matmul LLKs: ring-gather and fused all-gather are not "
        "supported by custom_mm");
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
        "matmul_decode ring gather: unexpected tile geometry");

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);

    IDevice* device = input_tensor_a.device();
    const bool all_gather = operation_attributes.all_gather;
    TT_FATAL(
        !all_gather || mesh_dispatch_coordinate.has_value(),
        "matmul_decode all_gather requires a mesh dispatch coordinate");

    const auto& packed = operation_attributes.packed_weight;
    auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    auto inputB_core_range_set =
        packed.has_value() ? packed->cores : input_tensor_b.memory_config().shard_spec().value().grid;
    auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;
    TT_FATAL(
        inputB_core_range_set == output_core_range_set,
        "matmul_decode ring gather: input tensor B and output tensor must have the same core range set");

    // Per-core sharding geometry. inA_K_tiles_per_core (= K_local in tiles) is a source shard's
    // width; each shard is M_tiles x inA_K_tiles_per_core.
    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height) &&
            inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "matmul_decode ring gather: input A shard shape must be [M_tiles*tile_h, k * tile_w]");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    uint32_t inB_N_tiles_per_core;
    if (packed.has_value()) {
        const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const uint32_t num_weight_cores = inputB_core_range_set.num_cores();
        TT_FATAL(
            N_tiles % num_weight_cores == 0,
            "matmul_decode ring gather (packed_weight): N in tiles ({}) must be divisible by weight core count ({})",
            N_tiles,
            num_weight_cores);
        inB_N_tiles_per_core = N_tiles / num_weight_cores;
    } else {
        const std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
        TT_FATAL(
            inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT) &&
                inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
            "matmul_decode ring gather: input B shard shape must be [K, per_core_N]");
        inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    }

    // No topological requirement on either grid: the ring walk is a logical sequence over
    // corerange_to_cores(..., row_wise=true), and NoC unicasts route through intermediate cores
    // at the hardware level without any relay kernel. Row-major snake grids (as used by the
    // packed-weight deepseek layer test) work as well as rectangles.
    TT_FATAL(
        inputA_core_range_set.num_cores() > 0 && output_core_range_set.num_cores() > 0,
        "matmul_decode ring gather: both input A grid and output grid must be non-empty");

    const std::vector<CoreCoord> S_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, /*row_wise=*/true);
    const std::vector<CoreCoord> C_cores = corerange_to_cores(output_core_range_set, std::nullopt, /*row_wise=*/true);
    const uint32_t num_senders = static_cast<uint32_t>(S_cores.size());

    TT_FATAL(
        num_senders == K_tiles / inA_K_tiles_per_core,
        "matmul_decode ring gather: sender count ({}) must equal K_tiles / inA_K_tiles_per_core ({} / {})",
        num_senders,
        K_tiles,
        inA_K_tiles_per_core);

    // Phase 1: single CW ring only. Bidirectional (CCW on RISCV_0/NOC_1) is a straightforward
    // extension -- build a second RingWalk over the second half of the senders, allocate a
    // cb_in2_ccw, and instantiate a second reader kernel.
    const RingWalk cw = build_ring_walk(device, /*sources=*/S_cores, /*hops=*/C_cores);

    // Kernels and CBs are placed on the walk's cores (S ∪ C) only. Cores that fall inside the
    // bounding box but neither shard nor compute get no kernel, no CB, and no L1 reservation.
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

    ProgramDescriptor desc;

    const uint32_t in0_cb_index = CBIndex::c_0;
    const uint32_t in1_cb_index = CBIndex::c_1;
    const uint32_t out_cb_index = CBIndex::c_2;
    const uint32_t in2_cw_cb_index = CBIndex::c_3;
    const uint32_t in2_ccw_cb_index = CBIndex::c_4;

    // cb_in0: local shard, allocated only on source cores over the input tensor's L1.
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

    // cb_in1: full weight, L1-resident, allocated on compute cores. In the packed case the
    // buffer is the fused tensor and address_offset re-bases it onto this weight's slab.
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_tiles * inB_N_tiles_per_core * in1_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
        .address_offset = packed.has_value() ? packed->tile_offset * in1_tile_size : 0,
    });

    // cb_out: on compute cores. Aliased over the output tensor unless all_gather is on,
    // in which case it is scratch (local N-shard) and the writer copies into the gathered tensor.
    CBDescriptor out_cb_desc{
        .total_size = M_tiles * inB_N_tiles_per_core * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    };
    if (!all_gather) {
        out_cb_desc.buffer = output_tensor.buffer();
    }
    desc.cbs.push_back(std::move(out_cb_desc));

    // cb_in2_cw: remote shard ring buffer for the CW direction. Sized to hold every shard this
    // core receives (max = num_senders across all cores in the ring walk). This is the CB whose
    // size scales with num_senders, but each core only fills num_recv slots.
    const uint32_t in2_cb_num_tiles = num_senders * M_tiles * inA_K_tiles_per_core;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_cb_num_tiles * in0_tile_size,
        .core_ranges = all_ring_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in2_cw_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    // cb_in2_ccw: zero-sized placeholder while CCW is not yet wired up. Sized to a single tile
    // so CB allocation succeeds; the compute kernel skips CCW when num_shards_ccw == 0.
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_tile_size,
        .core_ranges = all_ring_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in2_ccw_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    // Ring semaphores. sig_cw carries the pipeline credit for the CW ring. Allocated on the
    // whole ring so every core can be atomically bumped from its predecessor.
    constexpr uint32_t sig_cw_sem_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sig_cw_sem_id,
        .core_ranges = all_ring_cores,
        .initial_value = 0,
    });
    if (all_gather) {
        TT_FATAL(
            out_ready_semaphore != nullptr && barrier_semaphore != nullptr,
            "matmul_decode all_gather requires workload-scoped semaphores");
        // Cross-device semaphores are allocated once by AllGatherFullWidth and
        // kept alive by its WorkloadDescriptor. The activation-ring semaphore
        // above remains a normal per-program semaphore.
    }

    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t local_shard_num_tiles = shard_num_tiles;

    // ---- Reader kernel (CW) ----
    KernelDescriptor reader_cw;
    reader_cw.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_full_width_ring_gather.cpp";
    reader_cw.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_cw.core_ranges = all_ring_cores;
    reader_cw.compile_time_args = {
        shard_num_tiles,
        in0_tile_size,
        sig_cw_sem_id,
        local_shard_num_tiles,
    };
    reader_cw.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in2", in2_cw_cb_index},
    };
    reader_cw.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    };
    // Position of a core in the CW walk (if any).
    std::map<CoreCoord, uint32_t> cw_pos;
    for (uint32_t p = 0; p < cw.cores.size(); ++p) {
        cw_pos[cw.cores[p]] = p;
    }
    // Reader on every core in S ∪ C. `signal_in0` publishes cb_in0 for compute to consume:
    // - source-only: harmless (push_back with no compute consumer, cb_in0 is allocated).
    // - source-and-compute (overlap): required so compute.wait_front(cb_in0) resolves.
    // - hop (compute-only, no own shard): 0 -- cb_in0 isn't allocated on this core.
    reader_cw.runtime_args.reserve(S_or_C.size());
    for (const auto& core : S_or_C) {
        const auto it = cw_pos.find(core);
        if (it == cw_pos.end()) {
            reader_cw.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{RG_ROLE_IDLE, 0, 0, 0, 0, 0});
            continue;
        }
        const uint32_t p = it->second;
        const CoreCoord next = cw.next_phys[p];
        const uint32_t signal_in0 = cw.is_source[p] ? 1u : 0u;
        // Reader kernel only distinguishes "has own shard" (SOURCE) vs "no own shard" (HOP);
        // it doesn't care whether we're also a compute core. Fold the roles here so the kernel
        // stays simple.
        const uint32_t reader_role = cw.is_source[p] ? /*SOURCE*/ 1u : /*HOP*/ 2u;
        reader_cw.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                reader_role,
                cw.num_recv[p],
                cw.num_sends[p],
                static_cast<uint32_t>(next.x),
                static_cast<uint32_t>(next.y),
                signal_in0});
    }
    desc.kernels.push_back(std::move(reader_cw));

    // ---- Compute kernel ----
    TT_FATAL(
        M_tiles <= 8,
        "matmul_decode ring gather: M_tiles must be <= 8 so the output block fits in DST, but got {}",
        M_tiles);

    KernelDescriptor compute_kd;
    compute_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
        "compute_full_width_ring_gather.cpp";
    compute_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kd.core_ranges = output_core_range_set;

    compute_kd.compile_time_args = {
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core,
        // num_senders is the max any core's num_arriving_cw can be (compute-only cores see all
        // S shards; overlap cores see S-1). Passed as a compile-time constant only so the
        // kernel can size compile-time init decisions; the actual per-launch loop bound is a
        // runtime arg (num_arriving_cw) since it varies per core.
        /*num_senders=*/num_senders,
    };
    compute_kd.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in2_cw", in2_cw_cb_index},
        {"cb_in2_ccw", in2_ccw_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_out", out_cb_index},
    };
    compute_kd.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (all_gather) {
        compute_kd.defines.emplace_back("ENABLE_ALL_GATHER", "1");
    }
    // Per-core runtime args:
    //   [0] has_local_shard  (1 iff this core is also in S -- overlap)
    //   [1] local_sender_id  (source index if has_local_shard, else don't-care)
    //   [2] num_arriving_cw  (S for compute-only, S-1 for overlap)
    //   [3] num_arriving_ccw (0 while CCW ring is not wired up)
    //   [4..4+num_arriving_cw-1] cw_sender_ids (arrival order)
    //   [4+num_arriving_cw..] ccw_sender_ids (currently empty)
    compute_kd.runtime_args.reserve(C_cores.size());
    for (const auto& core : C_cores) {
        const auto it = cw_pos.find(core);
        TT_FATAL(
            it != cw_pos.end(),
            "matmul_decode ring gather: compute core {} is not on the CW walk (internal error)",
            core.str());
        const uint32_t p = it->second;
        const std::vector<uint32_t> ids = arriving_sender_ids_at(p, cw);
        const uint32_t expected = num_senders - cw.is_source[p];
        TT_FATAL(
            ids.size() == expected,
            "matmul_decode ring gather: compute core at walk pos {} sees {} sender IDs, expected {} "
            "(is_source={})",
            p,
            ids.size(),
            expected,
            cw.is_source[p]);
        KernelDescriptor::CoreRuntimeArgs rt;
        rt.reserve(4 + ids.size());
        rt.push_back(cw.is_source[p] ? 1u : 0u);                   // has_local_shard
        rt.push_back(cw.is_source[p] ? cw.own_sender_id[p] : 0u);  // local_sender_id
        rt.push_back(static_cast<uint32_t>(ids.size()));           // num_arriving_cw
        rt.push_back(0);                                           // num_arriving_ccw
        rt.insert(rt.end(), ids.begin(), ids.end());
        compute_kd.runtime_args.emplace_back(core, std::move(rt));
    }
    desc.kernels.push_back(std::move(compute_kd));

    if (all_gather) {
        const auto route = make_all_gather_fabric_route(input_tensor_a, *mesh_dispatch_coordinate);
        const auto available_cores =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().at(0));
        const auto available_core_list = corerange_to_cores(available_cores, std::nullopt, true);
        const auto output_core_list = corerange_to_cores(output_core_range_set, std::nullopt, true);
        const std::set<CoreCoord> ring_core_set(S_or_C.begin(), S_or_C.end());
        std::set<CoreCoord> output_core_set(output_core_list.begin(), output_core_list.end());
        std::vector<CoreCoord> mux_cores;
        for (const auto& core : available_core_list) {
            if (!output_core_set.contains(core) && !ring_core_set.contains(core)) {
                mux_cores.push_back(core);
            }
        }
        const bool use_mux = C_cores.size() > 1;
        std::array<std::vector<uint32_t>, 2> mux_links_by_dir;
        uint32_t dst_index = 0;
        const auto sender_node = input_tensor_a.device()->get_fabric_node_id(*mesh_dispatch_coordinate);
        for (uint32_t dir = 0; dir < 2; ++dir) {
            const uint32_t range = dir == 0 ? route.range_fwd : route.range_bwd;
            if (range == 0) {
                continue;
            }
            const auto valid_links = tt::tt_fabric::get_forwarding_link_indices(
                *input_tensor_a.device(), sender_node, route.dst_nodes[dst_index++]);
            TT_FATAL(!valid_links.empty(), "matmul_decode all_gather: no forwarding link for mux direction {}", dir);
            mux_links_by_dir[dir] = valid_links;
        }
        const uint32_t required_mux_cores =
            static_cast<uint32_t>(mux_links_by_dir[0].size() + mux_links_by_dir[1].size());
        TT_FATAL(
            C_cores.size() <= 1 || mux_cores.size() >= required_mux_cores,
            "matmul_decode all_gather needs {} free mux cores, but only {} are available",
            required_mux_cores,
            mux_cores.size());
        std::array<std::vector<CoreCoord>, 2> mux_core_groups;
        uint32_t next_mux_core = 0;
        uint32_t max_clients_per_mux = 0;
        for (uint32_t dir = 0; dir < 2; ++dir) {
            for (uint32_t mux_idx = 0; mux_idx < mux_links_by_dir[dir].size(); ++mux_idx) {
                mux_core_groups[dir].push_back(mux_cores[next_mux_core++]);
            }
            if (!mux_links_by_dir[dir].empty()) {
                max_clients_per_mux = std::max(
                    max_clients_per_mux,
                    div_up(static_cast<uint32_t>(C_cores.size()), static_cast<uint32_t>(mux_links_by_dir[dir].size())));
            }
        }
        tt::tt_fabric::FabricMuxConfig mux_config(
            /*num_full_size_channels=*/max_clients_per_mux,
            /*num_header_only_channels=*/0,
            /*num_buffers_full_size_channel=*/2,
            /*num_buffers_header_only_channel=*/0,
            /*buffer_size_bytes_full_size_channel=*/
            tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*input_tensor_a.device()),
            /*base_l1_address=*/device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
        KernelDescriptor writer;
        writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
            "writer_full_width_all_gather.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = output_core_range_set;
        writer.named_compile_time_args = all_gather_named_compile_time_args(
            out_cb_index,
            M_tiles,
            inB_N_tiles_per_core,
            route.ring_index,
            operation_attributes.ring_size,
            static_cast<uint32_t>(out_ready_semaphore->address()),
            static_cast<uint32_t>(barrier_semaphore->address()),
            route.start_fwd,
            route.range_fwd,
            route.start_bwd,
            route.range_bwd,
            /*ag_rt_arg_base=*/0,
            /*num_shards=*/1,
            /*shard_sem_id=*/0,
            /*staging_cb_id=*/0,
            use_mux,
            mux_config.get_num_buffers(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL),
            mux_config.get_buffer_size_bytes(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL),
            mux_config.get_status_address(),
            mux_config.get_termination_signal_address(),
            max_clients_per_mux);
        writer.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_1,
        };
        const KernelHandle writer_id = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(writer));
        CoreRangeSet mux_core_range;
        if (use_mux) {
            std::vector<CoreRange> active_mux_ranges;
            for (uint32_t dir = 0; dir < 2; ++dir) {
                for (const auto& mux_core : mux_core_groups[dir]) {
                    active_mux_ranges.emplace_back(mux_core);
                }
            }
            mux_core_range = CoreRangeSet(active_mux_ranges);
            KernelDescriptor mux_kernel;
            mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
            mux_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
            mux_kernel.core_ranges = mux_core_range;
            mux_kernel.compile_time_args = mux_config.get_fabric_mux_compile_time_args();
            mux_kernel.config =
                DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
            mux_kernel.runtime_args.reserve(required_mux_cores);
            uint32_t route_dst_index = 0;
            for (uint32_t dir = 0; dir < 2; ++dir) {
                if ((dir == 0 && route.range_fwd == 0) || (dir == 1 && route.range_bwd == 0)) {
                    continue;
                }
                const auto dst = route.dst_nodes[route_dst_index++];
                for (uint32_t mux_idx = 0; mux_idx < mux_core_groups[dir].size(); ++mux_idx) {
                    const auto mux_core = mux_core_groups[dir][mux_idx];
                    mux_kernel.runtime_args.emplace_back(
                        mux_core,
                        mux_config.get_fabric_mux_run_time_args(
                            input_tensor_a.device()->get_fabric_node_id(*mesh_dispatch_coordinate),
                            dst,
                            mux_links_by_dir[dir][mux_idx],
                            desc,
                            mux_core));
                }
            }
            desc.kernels.push_back(std::move(mux_kernel));
        }
        const uint32_t gathered_n_tiles = inB_N_tiles_per_core * operation_attributes.ring_size;
        const uint32_t local_n_tiles = inB_N_tiles_per_core * static_cast<uint32_t>(C_cores.size());
        for (uint32_t i = 0; i < C_cores.size(); ++i) {
            // The gathered tensor remains WIDTH_SHARDED across C_cores. A local result from
            // compute core i belongs at global N tile
            //   ring_index * N_local_tiles + i * N_tiles_per_core.
            // Convert that global tile to the destination output shard and its in-shard offset.
            const uint32_t global_n_tile = route.ring_index * local_n_tiles + i * inB_N_tiles_per_core;
            const uint32_t output_core_idx = global_n_tile / gathered_n_tiles;
            const uint32_t output_tile_offset = global_n_tile % gathered_n_tiles;
            TT_FATAL(
                output_core_idx < C_cores.size() && output_tile_offset + inB_N_tiles_per_core <= gathered_n_tiles,
                "matmul_decode all_gather output placement is invalid: source core {}, global tile {}, "
                "destination core {}, offset {}, shard width {}",
                i,
                global_n_tile,
                output_core_idx,
                output_tile_offset,
                gathered_n_tiles);
            const CoreCoord output_core = C_cores[output_core_idx];
            const auto forwarding_links =
                all_gather_forwarding_links(input_tensor_a, *mesh_dispatch_coordinate, route, i);
            if (use_mux) {
                std::array<CoreCoord, 2> mux_core_by_dir{};
                std::array<CoreCoord, 2> termination_masters{};
                std::array<uint32_t, 2> mux_worker_ids{};
                std::array<uint32_t, 2> mux_client_counts{};
                for (uint32_t dir = 0; dir < 2; ++dir) {
                    if ((dir == 0 && route.range_fwd == 0) || (dir == 1 && route.range_bwd == 0)) {
                        continue;
                    }
                    const uint32_t mux_group = i % static_cast<uint32_t>(mux_core_groups[dir].size());
                    const uint32_t mux_group_start = mux_group;
                    const uint32_t mux_group_count = div_up(
                        static_cast<uint32_t>(C_cores.size()) - mux_group_start,
                        static_cast<uint32_t>(mux_core_groups[dir].size()));
                    mux_core_by_dir[dir] = mux_core_groups[dir][mux_group];
                    termination_masters[dir] = C_cores[mux_group_start];
                    mux_worker_ids[dir] = i / static_cast<uint32_t>(mux_core_groups[dir].size());
                    mux_client_counts[dir] = mux_group_count;
                }
                set_all_gather_writer_runtime_args(
                    desc,
                    writer_id,
                    C_cores[i],
                    device,
                    input_tensor_a,
                    *mesh_dispatch_coordinate,
                    output_tensor,
                    route,
                    output_core,
                    output_tile_offset,
                    {},
                    {},
                    {},
                    mux_config,
                    mux_core_by_dir,
                    termination_masters,
                    mux_worker_ids,
                    mux_client_counts);
            } else {
                set_all_gather_writer_runtime_args(
                    desc,
                    writer_id,
                    C_cores[i],
                    device,
                    input_tensor_a,
                    *mesh_dispatch_coordinate,
                    output_tensor,
                    route,
                    output_core,
                    output_tile_offset,
                    forwarding_links);
            }
        }
    }

    log_debug(
        tt::LogOp,
        "matmul_decode ring gather: num_senders={}, num_compute={}, M_tiles={}, K_tiles={}, "
        "inA_K_tiles_per_core={}, inB_N_tiles_per_core={}",
        num_senders,
        C_cores.size(),
        M_tiles,
        K_tiles,
        inA_K_tiles_per_core,
        inB_N_tiles_per_core);

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MatmulDecodeDeviceOperation::AllGatherFullWidth::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    TT_FATAL(operation_attributes.all_gather, "internal error: all-gather workload factory used without all_gather");
    TT_FATAL(
        operation_attributes.prefetcher_pipes.empty(),
        "matmul_decode all_gather is not supported with prefetcher_pipes (tensor prefetcher)");

    auto* mesh_device = tensor_args.input_tensor_a.device();
    const auto subdevice_id = mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);

    // In L1_SMALL when the device reserves it, so the semaphores do not fragment L1.
    const auto sem_buffer_type = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0
                                     ? tt::tt_metal::BufferType::L1_SMALL
                                     : tt::tt_metal::BufferType::L1;
    tt::tt_metal::WorkloadDescriptor workload;
    workload.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    workload.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    const auto& out_ready_semaphore = workload.semaphores[0];
    const auto& barrier_semaphore = workload.semaphores[1];

    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    workload.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        auto descriptor = create_descriptor_ring_gather_full(
            operation_attributes, tensor_args, tensor_return_value, coord, &out_ready_semaphore, &barrier_semaphore);
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(descriptor)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::matmul_decode
