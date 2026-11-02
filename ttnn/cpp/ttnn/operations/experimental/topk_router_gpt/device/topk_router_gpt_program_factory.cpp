// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_device_operation.hpp"
#include "topk_router_gpt_device_operation_types.hpp"

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::topk_router_gpt {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// name / DataFormat / is_tile / tiles_per_dfb
using DfbSpec = std::tuple<DFBSpecName, tt::DataFormat, bool, uint32_t>;

void push_dfbs(ProgramSpec& spec, const std::vector<DfbSpec>& specs) {
    for (const auto& [name, data_format, is_tile, tiles_per_dfb] : specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = name,
            .entry_size = bytes_per_tile,
            .num_entries = tiles_per_dfb,
            .data_format_metadata = data_format,
        });
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts
TopkRouterGptDeviceOperation::TopkRouterGptProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const KernelSpecName DM0{"dm0"};
    const KernelSpecName DM1{"dm1"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName DFB_WEIGHT{"weight"};
    const DFBSpecName DFB_INPUT{"input"};
    const DFBSpecName DFB_PARTIAL_RECV{"partial_recv"};
    const DFBSpecName DFB_LOCAL_OUT{"local_out"};
    const DFBSpecName DFB_BIAS{"bias"};
    const DFBSpecName DFB_INDEX{"index"};
    const DFBSpecName DFB_TOPK_VAL{"topk_val"};
    const DFBSpecName DFB_GATHERED_VAL{"gathered_val"};
    const DFBSpecName DFB_GATHERED_IND{"gathered_ind"};
    const DFBSpecName DFB_INTERMED_VAL{"intermed_val"};
    const DFBSpecName DFB_INTERMED_IND{"intermed_ind"};
    const DFBSpecName DFB_SOFTMAX_MASK{"softmax_mask"};
    const DFBSpecName DFB_SOFTMAX_TMP{"softmax_tmp"};
    const DFBSpecName DFB_REDUCE_SCALAR{"reduce_scalar"};
    const DFBSpecName DFB_BCAST_SCALER{"bcast_scaler"};
    const DFBSpecName DFB_FINAL_OUT{"final_out"};
    const DFBSpecName DFB_DISPATCH{"dispatch"};

    const SemaphoreSpecName SEM_PARTIAL_READY{"partial_ready"};
    const SemaphoreSpecName SEM_TOPK_READY{"topk_ready"};

    const TensorParamName INPUT{"input"};
    const TensorParamName WEIGHT{"weight"};
    const TensorParamName BIAS{"bias"};
    const TensorParamName INDICES_RM{"indices_rm"};
    const TensorParamName WEIGHTS_RM{"weights_rm"};

    const auto& input = tensor_args.input_tensor.mesh_tensor();
    const auto& weight = tensor_args.weight_tensor.mesh_tensor();
    const auto& bias = tensor_args.bias_tensor.mesh_tensor();
    const auto& indices_rm = std::get<0>(tensor_return_value).mesh_tensor();
    const auto& weights_rm = std::get<1>(tensor_return_value).mesh_tensor();

    ProgramSpec spec{.name = "topk_router_gpt"};

    auto* device = tensor_args.input_tensor.device();

    // Get the cores for the program
    const auto dram_bank2core_coords =
        device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    const uint32_t num_cores = dram_bank2core_coords.size();
    auto all_cores = tt::tt_metal::CoreRangeSet(dram_bank2core_coords);

    constexpr uint32_t num_groups = 4;
    // Wormhole exposes 12 DRAM-aligned workers and uses two senders per
    // expert group. Blackhole P150 exposes 8, so split each K dimension between
    // one sender and one worker instead. The rest of the four-group routing
    // and collection pipeline is identical.
    const uint32_t cores_per_group = num_cores >= 12 ? 3 : 2;
    const uint32_t num_senders = cores_per_group - 1;
    const uint32_t required_cores = num_groups * cores_per_group;
    TT_FATAL(
        num_cores >= required_cores,
        "topk_router_gpt requires at least {} DRAM-aligned cores, got {}",
        required_cores,
        num_cores);

    // Tensor shapes
    const auto& input_shape = tensor_args.input_tensor.logical_shape();
    const uint32_t hidden_dim = input_shape[1];
    const uint32_t num_experts = operation_attributes.num_experts;
    constexpr uint32_t tile_hw = 32;

    const uint32_t total_k_tiles = hidden_dim / tile_hw;
    const uint32_t n_tiles = num_experts / tile_hw;
    const uint32_t k_tiles_per_core_base = total_k_tiles / cores_per_group;
    const uint32_t k_tiles_remainder = total_k_tiles % cores_per_group;
    const uint32_t max_k_tiles = k_tiles_per_core_base + (k_tiles_remainder > 0 ? 1 : 0);

    // DFBs used in the topk_router_gpt operation
    // Every DFB MUST sit at the same L1 offset on all cores because the DM1 kernel reads its
    // own partial_recv (and, on workers, gathered_val / gathered_ind) base address and uses it
    // as the NOC write destination for other cores. If the layouts differed between cores, the
    // L1 offsets would diverge silently. Every kernel runs on every core, so every DFB is placed
    // on all cores and allocated in the order declared below, which keeps the layout uniform.
    /*
        -------------------------------------------------------------------
        |     Name           |   Dtype    | Tile? | Tiles/DFB | Used on     |
        -------------------------------------------------------------------
        | weight             | Float16_b  | true  | max_k     | All         |
        | input              | Float16_b  | true  | max_k     | All         |
        | partial_recv       | Float16_b  | true  | senders   | All         |
        | local_out          | Float16_b  | true  | 1         | All         |
        | bias               | Float16_b  | true  | 1         | Workers     |
        | index              | Float16_b  | true  | 1         | Workers     |
        | topk_val           | Float16_b  | true  | 1         | Workers     |
        | gathered_val       | Float16_b  | true  | 4         | Workers     |
        | gathered_ind       | Float16_b  | true  | 4         | Workers     |
        | intermed_val       | Float16_b  | true  | 2         | Collector   |
        | intermed_ind       | Float16_b  | true  | 1         | Collector   |
        | softmax_mask       | Float16_b  | true  | 1         | Collector   |
        | softmax_tmp        | Float16_b  | true  | 1         | Collector   |
        | reduce_scalar      | Float16_b  | true  | 1         | Collector   |
        | bcast_scaler       | Float16_b  | true  | 1         | Collector   |
        | final_out          | Float16_b  | true  | 2         | Collector   |
        | dispatch           | Float16_b  | false | var       | Collector   |
        -------------------------------------------------------------------
    */

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &dram_bank2core_coords](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;
            }
            return pa.x > pb.x;
        });

    // Map ring positions to group roles
    const uint32_t collector_ring_pos = num_senders;
    const uint32_t collector_bank_id = ring_pos2bank_id[collector_ring_pos];
    const auto collector_logical = dram_bank2core_coords[collector_bank_id];
    const auto collector_physical = device->worker_core_from_logical_core(collector_logical);

    push_dfbs(
        spec,
        {
            {DFB_WEIGHT, tt::DataFormat::Float16_b, true, max_k_tiles},
            {DFB_INPUT, tt::DataFormat::Float16_b, true, max_k_tiles},
            {DFB_PARTIAL_RECV, tt::DataFormat::Float16_b, true, num_senders},
            {DFB_LOCAL_OUT, tt::DataFormat::Float16_b, true, 1},
        });

    // Worker DFBs (includes collector)
    push_dfbs(
        spec,
        {
            {DFB_BIAS, tt::DataFormat::Float16_b, true, 1},
            {DFB_INDEX, tt::DataFormat::Float16_b, true, 1},
            {DFB_TOPK_VAL, tt::DataFormat::Float16_b, true, 1},
            {DFB_GATHERED_VAL, tt::DataFormat::Float16_b, true, num_groups},
            {DFB_GATHERED_IND, tt::DataFormat::Float16_b, true, num_groups},
        });

    // Collector-only DFBs
    push_dfbs(
        spec,
        {
            {DFB_INTERMED_VAL, tt::DataFormat::Float16_b, true, 2},
            {DFB_INTERMED_IND, tt::DataFormat::Float16_b, true, 1},
            {DFB_SOFTMAX_MASK, tt::DataFormat::Float16_b, true, 1},
            {DFB_SOFTMAX_TMP, tt::DataFormat::Float16_b, true, 1},
            {DFB_REDUCE_SCALAR, tt::DataFormat::Float16_b, true, 1},
            {DFB_BCAST_SCALER, tt::DataFormat::Float16_b, true, 1},
            {DFB_FINAL_OUT, tt::DataFormat::Float16_b, true, 2},
        });

    // Dispatch scratch (collector only, non-tile)
    uint32_t k_padded = tt::round_up(operation_attributes.k, 8);
    uint32_t dispatch_scratch_size = 2 * tile_hw * k_padded * 2;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DFB_DISPATCH,
        .entry_size = dispatch_scratch_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    });

    // Tensors: inputs are read by dm0, outputs written by dm1.
    spec.tensor_parameters = {
        {.unique_id = INPUT, .spec = input.tensor_spec()},
        {.unique_id = WEIGHT, .spec = weight.tensor_spec()},
        {.unique_id = BIAS, .spec = bias.tensor_spec()},
        {.unique_id = INDICES_RM, .spec = indices_rm.tensor_spec()},
        {.unique_id = WEIGHTS_RM, .spec = weights_rm.tensor_spec()},
    };

    const uint32_t tile_size_bf16 = tt::tile_size(tt::DataFormat::Float16_b);

    const KernelSpec::CompileTimeArgs named_compile_time_args = {
        {"num_cores", num_cores},
        {"num_groups", num_groups},
        {"cores_per_group", cores_per_group},
        {"num_senders", num_senders},
        {"collector_physical_x", static_cast<uint32_t>(collector_physical.x)},
        {"collector_physical_y", static_cast<uint32_t>(collector_physical.y)},
        {"topk_k", operation_attributes.k},
        {"k_padded", k_padded},
        {"n_tiles", n_tiles},
        {"tile_size_bf16", tile_size_bf16},
    };

    // Shared runtime-arg schema across all 3 kernels (each reads only what it needs).
    const KernelSpec::RuntimeArgSchema runtime_arg_schema = {
        .runtime_arg_names =
            {
                "dram_bank_id",
                "vchannel",
                "is_sender",
                "is_worker",
                "is_collector",
                "num_k_tiles",
                "k_tile_offset",
                "n_tile_id",
                "worker_phys_x",
                "worker_phys_y",
                "sender_slot",
                "worker_gather_slot",
            },
    };

    // Create kernels for the program.  Pushed dm0, dm1, compute; all three read the same
    // per-core runtime-arg block set below.
    spec.kernels.push_back(KernelSpec{
        .unique_id = DM0,
        .source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm0.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O2},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_WEIGHT,
                    .accessor_name = "weight",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INPUT,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_BIAS,
                    .accessor_name = "bias",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "input",
                },
                TensorBinding{
                    .tensor_parameter_name = WEIGHT,
                    .accessor_name = "weight",
                },
                TensorBinding{
                    .tensor_parameter_name = BIAS,
                    .accessor_name = "bias",
                },
            },
        .compile_time_args = named_compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = DM1,
        .source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/dm1.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O2},
        .dfb_bindings =
            {
                // Senders peek partial_recv's write pointer as the NOC destination on their worker;
                // workers fill it (FIFO producer) from the senders' NOC writes.
                DFBBinding{
                    .dfb_spec_name = DFB_PARTIAL_RECV,
                    .accessor_name = "partial_recv",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_LOCAL_OUT,
                    .accessor_name = "local_out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // index: dm1 both generates the index tile and reads it back, so it is its sole toucher.
                DFBBinding{
                    .dfb_spec_name = DFB_INDEX,
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INDEX,
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_TOPK_VAL,
                    .accessor_name = "topk_val",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // Non-collector workers peek gathered_val / gathered_ind's write pointer as the NOC
                // destination on the collector; the collector fills them (FIFO producer).
                DFBBinding{
                    .dfb_spec_name = DFB_GATHERED_VAL,
                    .accessor_name = "gathered_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_GATHERED_IND,
                    .accessor_name = "gathered_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_SOFTMAX_MASK,
                    .accessor_name = "softmax_mask",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_BCAST_SCALER,
                    .accessor_name = "bcast_scaler",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_FINAL_OUT,
                    .accessor_name = "final_out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // dispatch: dm1-only scratch for the row-major outputs.
                DFBBinding{
                    .dfb_spec_name = DFB_DISPATCH,
                    .accessor_name = "dispatch",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_DISPATCH,
                    .accessor_name = "dispatch",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_PARTIAL_READY,
                    .accessor_name = "partial_ready",
                },
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_TOPK_READY,
                    .accessor_name = "topk_ready",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INDICES_RM,
                    .accessor_name = "indices_rm",
                },
                TensorBinding{
                    .tensor_parameter_name = WEIGHTS_RM,
                    .accessor_name = "weights_rm",
                },
            },
        .compile_time_args = named_compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DFB_WEIGHT,
                    .accessor_name = "weight",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INPUT,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_PARTIAL_RECV,
                    .accessor_name = "partial_recv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_LOCAL_OUT,
                    .accessor_name = "local_out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_BIAS,
                    .accessor_name = "bias",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_TOPK_VAL,
                    .accessor_name = "topk_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_GATHERED_VAL,
                    .accessor_name = "gathered_val",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_GATHERED_IND,
                    .accessor_name = "gathered_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // intermed_val / intermed_ind / softmax_tmp / reduce_scalar are compute-internal
                // staging buffers: compute both packs into them and unpacks from them.
                DFBBinding{
                    .dfb_spec_name = DFB_INTERMED_VAL,
                    .accessor_name = "intermed_val",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INTERMED_VAL,
                    .accessor_name = "intermed_val",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INTERMED_IND,
                    .accessor_name = "intermed_ind",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_INTERMED_IND,
                    .accessor_name = "intermed_ind",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_SOFTMAX_MASK,
                    .accessor_name = "softmax_mask",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_SOFTMAX_TMP,
                    .accessor_name = "softmax_tmp",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_SOFTMAX_TMP,
                    .accessor_name = "softmax_tmp",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_REDUCE_SCALAR,
                    .accessor_name = "reduce_scalar",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_REDUCE_SCALAR,
                    .accessor_name = "reduce_scalar",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_BCAST_SCALER,
                    .accessor_name = "bcast_scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DFB_FINAL_OUT,
                    .accessor_name = "final_out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = named_compile_time_args,
        .runtime_arg_schema = runtime_arg_schema,
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
                .sfpu_precision_mode = Precision::Precise,
                .enable_32_bit_dest = true,
                .double_buffer_dest = true,
            },
    });

    // Create semaphores. dm1 signals them remotely (sender -> worker, worker -> collector) and
    // waits on them locally, so they live on every core.
    spec.semaphores = {
        SemaphoreSpec{
            .unique_id = SEM_PARTIAL_READY,
            .target_nodes = all_cores,
        },
        SemaphoreSpec{
            .unique_id = SEM_TOPK_READY,
            .target_nodes = all_cores,
        },
    };

    spec.work_units = {
        WorkUnitSpec{
            .name = "main",
            .kernels = {DM0, DM1, COMPUTE},
            .target_nodes = all_cores,
        },
    };

    // VChannel computation with conflict avoidance
    std::vector<uint32_t> vchannels;
    vchannels.reserve(num_cores);
    for (uint32_t bank_id = 0; bank_id < num_cores; bank_id++) {
        uint32_t vchannel = bank_id & 0x3;
        auto it = std::find_if(
            dram_bank2core_coords.begin(), dram_bank2core_coords.begin() + bank_id, [&](const auto& core_prev) {
                return core_prev.y == dram_bank2core_coords[bank_id].y;
            });
        if (it != dram_bank2core_coords.begin() + bank_id) {
            size_t j = std::distance(dram_bank2core_coords.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);
    }

    // Set the runtime arguments for the kernels
    // Shared across all 3 kernels (each reads only what it needs). The five tensor addresses
    // are tensor bindings and the two semaphores are semaphore bindings, so the framework
    // patches the addresses on a cache hit. Everything else derives from the tensor specs and
    // the device's DRAM bank assignment, all of which a cache hit reproduces.
    KernelRunArgs::RuntimeArgValues runtime_arg_values;
    for (uint32_t ring_pos = 0; ring_pos < required_cores; ring_pos++) {
        uint32_t bank_id = ring_pos2bank_id[ring_pos];
        const auto& core = dram_bank2core_coords[bank_id];

        uint32_t group_id = ring_pos / cores_per_group;
        uint32_t pos_in_group = ring_pos % cores_per_group;

        uint32_t k_tiles = k_tiles_per_core_base + (pos_in_group < k_tiles_remainder ? 1 : 0);
        uint32_t k_tile_offset = 0;
        for (uint32_t j = 0; j < pos_in_group; j++) {
            k_tile_offset += k_tiles_per_core_base + (j < k_tiles_remainder ? 1 : 0);
        }

        bool is_sender = (pos_in_group < num_senders);
        bool is_worker = (pos_in_group == num_senders);
        bool is_collector = (bank_id == collector_bank_id);

        uint32_t worker_ring_pos = (group_id * cores_per_group) + num_senders;
        uint32_t worker_bank_id_val = ring_pos2bank_id[worker_ring_pos];
        const auto worker_physical = device->worker_core_from_logical_core(dram_bank2core_coords[worker_bank_id_val]);

        AddRuntimeArgsForNode(
            runtime_arg_values,
            core,
            {
                {"dram_bank_id", bank_id},
                {"vchannel", vchannels[bank_id]},
                {"is_sender", is_sender ? 1u : 0u},
                {"is_worker", is_worker ? 1u : 0u},
                {"is_collector", is_collector ? 1u : 0u},
                {"num_k_tiles", k_tiles},
                {"k_tile_offset", k_tile_offset},
                {"n_tile_id", group_id},
                {"worker_phys_x", static_cast<uint32_t>(worker_physical.x)},
                {"worker_phys_y", static_cast<uint32_t>(worker_physical.y)},
                {"sender_slot", pos_in_group},
                {"worker_gather_slot", group_id},
            });
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = DM0,
            .runtime_arg_values = runtime_arg_values,
        },
        KernelRunArgs{
            .kernel = DM1,
            .runtime_arg_values = runtime_arg_values,
        },
        KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values = std::move(runtime_arg_values),
        },
    };
    run_args.tensor_args = {
        {INPUT, input},
        {WEIGHT, weight},
        {BIAS, bias},
        {INDICES_RM, indices_rm},
        {WEIGHTS_RM, weights_rm},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
