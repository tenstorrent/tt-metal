// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The placement program of the expert rows path (MoEComputePlaceFactory): moe_compute's outputs from the rows and the
// routing table, on the height_shards x width_shards combine cores. place.cpp writes the metadata outputs and each
// core's shard of the double buffer; with a combine (SingleDevice, FullCcl) rows_combine.cpp on the same cores sends
// every routed row to the combine output, over the fabric with FullCcl.
#include <tt-metalium/tensor_accessor_args.hpp>

#include "moe_compute_device_operation.hpp"
#include "moe_compute_program_factory.hpp"
#include "moe_expert_rows_plan.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

namespace ttnn::experimental::prim {

using namespace expert_rows_host;

namespace {

constexpr uint32_t kOutputsWithCombine = 6;
// The placement program: place.cpp, then with a combine rows_combine.cpp (then, with FullCcl, the fabric mux kernels)
constexpr uint32_t kCombineKernelIdx = 1;
constexpr uint32_t kCombineCrossDeviceSemaphoreArgIdx = 2;  // rows_combine.cpp's final barrier semaphore
enum PlaceCb : uint32_t { kPlaceScratchCb = 0, kCombineScratchCb = 1, kPacketHeaderCb = 2 };

// The placement program on the device at mesh_coordinate.
tt::tt_metal::ProgramDescriptor build_place_program_descriptor(
    const MoEComputeParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const MoEComputePlaceInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_barrier_semaphore,
    const std::optional<GlobalSemaphore>& final_barrier_semaphore) {
    const bool combine = args.path != MoEComputePath::ComputeOnly;
    const bool ccl = args.path == MoEComputePath::FullCcl;
    TT_FATAL(
        !ccl || (init_barrier_semaphore.has_value() && final_barrier_semaphore.has_value()),
        "moe_compute expert rows placement program: FullCcl needs the combine's barrier semaphores");
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const ttnn::Tensor& row_buffer = tensor_args.row_buffer_tensor;
    const ttnn::Tensor& table = tensor_args.routing_table_tensor;
    const auto& input_shape = tensor_args.tilize_input_tensor.logical_shape();
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t row_bytes = hidden_size * kBf16Bytes;
    const uint32_t tokens = input_shape[0] * input_shape[1];
    const uint32_t top_k = tensor_args.tilize_expert_indices_tensor.logical_shape()[-1];
    const uint32_t local_experts = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    const uint32_t height_shards = args.num_token_parallel_cores, width_shards = args.num_data_parallel_cores;
    const PlaceLayout place = place_layout(tokens, top_k, local_experts);
    TT_FATAL(
        table.buffer()->size() >= place.table.bytes,
        "moe_compute expert rows placement program: routing table is {} B, expected {}",
        table.buffer()->size(),
        place.table.bytes);

    const ttnn::Tensor& counts_out = tensor_return_value[kExpertTokenCounts];
    const ttnn::Tensor& activation_out = tensor_return_value[kExpertActivation];
    const ttnn::Tensor& e_t_out = tensor_return_value[kETPages];
    const ttnn::Tensor& double_buffer = tensor_return_value[kTilizeOutput];
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords virtual_coords = grid_coords(mesh_device, grid);

    std::vector<CoreCoord> cores;
    if (!combine) {
        cores = get_moe_combine_cores(
            mesh_device, height_shards, width_shards, hidden_size, CoreRangeSet{}, args.bh_ring_size);
    } else {
        TT_FATAL(
            args.combine_params.has_value() && args.combine_params->local_combine == !ccl &&
                tensor_return_value.size() == kOutputsWithCombine,
            "moe_compute expert rows placement program: SingleDevice / FullCcl need their combine parameters and {} "
            "outputs; got combine parameters {}, local_combine {}, {} outputs",
            kOutputsWithCombine,
            args.combine_params.has_value(),
            args.combine_params.has_value() && args.combine_params->local_combine,
            tensor_return_value.size());
        const auto& worker_cores = args.combine_params->worker_cores;
        TT_FATAL(
            worker_cores.size() >= height_shards * width_shards,
            "moe_compute expert rows placement program: {} combine cores, need {}",
            worker_cores.size(),
            height_shards * width_shards);
        cores.assign(worker_cores.begin(), worker_cores.begin() + height_shards * width_shards);
    }
    const CoreRangeSet core_set = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));

    // place.cpp
    create_cb(desc, core_set, kPlaceScratchCb, place.scratch_bytes, tt::DataFormat::UInt32, place.scratch_bytes);
    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs place_ct = {
        {"cb_scratch", kPlaceScratchCb},
        {"tokens", tokens},
        {"top_k", top_k},
        {"local_experts", local_experts},
        {"height_shards", height_shards},
        {"segment_bytes", hidden_size / width_shards * kBf16Bytes},
        {"half_rows", double_buffer_half_rows(double_buffer.tensor_spec(), hidden_size, width_shards)},
        {"count_words", place.count_words},
        {"e_t_entry_words", place.e_t_entry_words},
        {"e_t_page_bytes", place.e_t_page_bytes},
        {"activation_row_bytes", place.activation_row_bytes},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"program_cores", static_cast<uint32_t>(cores.size())},
        {"row_bytes", row_bytes},
        {"scratch_e_t", place.scratch_e_t},
        {"scratch_activation", place.scratch_activation},
        {"scratch_count_row", place.scratch_count_row},
        {"scratch_table", place.scratch_table},
    };
    std::vector<uint32_t> place_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*row_buffer.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*e_t_out.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*activation_out.buffer()).append_to(place_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(place_ct_pos);
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + "place.cpp",
        .core_ranges = core_set,
        .compile_time_args = std::move(place_ct_pos),
        .named_compile_time_args = std::move(place_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    });
    for (uint32_t core_index = 0; core_index < cores.size(); ++core_index) {
        tt::tt_metal::KernelDescriptor::RTArgList place_args;
        place_args.push_back(row_buffer.buffer());
        place_args.push_back(e_t_out.buffer());
        place_args.push_back(activation_out.buffer());
        place_args.push_back(counts_out.buffer());
        place_args.push_back(double_buffer.buffer());
        place_args.push_back(table.buffer());
        place_args.append({core_index / width_shards, core_index % width_shards, core_index});
        place_args.append(virtual_coords.xs);
        place_args.append(virtual_coords.ys);
        desc.kernels.back().emplace_runtime_args(cores[core_index], place_args);
    }
    if (!combine) {
        return desc;
    }

    // rows_combine.cpp, with FullCcl its fabric side
    const auto& combine_params = *args.combine_params;
    const ttnn::Tensor& output = tensor_return_value[kCombineOutput];
    TT_FATAL(
        output.buffer()->page_size() == row_bytes,
        "moe_compute expert rows placement program: the combine output's pages are {} B, expected one row of {} B",
        output.buffer()->page_size(),
        row_bytes);
    const uint32_t senders = cores.size();
    const CombineLayout layout = combine_layout(tokens, top_k, hidden_size, senders);
    create_cb(desc, core_set, kCombineScratchCb, layout.bytes, tt::DataFormat::UInt32, layout.bytes);
    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs combine_ct = {
        {"cb_scratch", kCombineScratchCb},
        {"num_senders", senders},
        {"tokens", tokens},
        {"row_bytes", row_bytes},
        {"row_stride", layout.row_stride},
        {"batch_rows", layout.batch_rows},
        {"sender_rows_max", layout.sender_rows_max},
        {"top_k", top_k},
        {"local_experts", local_experts},
        {"scratch_head", layout.scratch_head},
        {"scratch_tokens", layout.scratch_tokens},
        {"scratch_k", layout.scratch_k},
        {"scratch_order", layout.scratch_order},
        {"scratch_stage", layout.scratch_stage},
    };
    tt::tt_metal::KernelDescriptor::Defines combine_defines;
    std::optional<CombineFabric> fabric;
    if (ccl) {
        fabric = plan_combine_fabric(
            desc,
            *mesh_device,
            mesh_coordinate,
            all_mesh_coordinates,
            combine_params.topology,
            combine_params.axis,
            combine_params.num_links,
            combine_params.mux_core_range_set,
            cores,
            kPacketHeaderCb,
            combine_max_packet_size_bytes(*mesh_device, row_buffer.dtype()));
        combine_ct.insert(
            combine_ct.end(), fabric->named_compile_time_args.begin(), fabric->named_compile_time_args.end());
        // a caller that owns the output and the final barrier semaphore orders the devices' writes itself
        const bool use_init_semaphore = !tensor_args.optional_output_tensor.has_value() ||
                                        !combine_params.optional_cross_device_semaphore.has_value();
        combine_ct.emplace_back("use_init_semaphore", static_cast<uint32_t>(use_init_semaphore));
        combine_defines = fabric->defines;
        combine_defines.emplace_back("COMBINE_FABRIC", "1");
    }
    std::vector<uint32_t> combine_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(combine_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*row_buffer.buffer()).append_to(combine_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table.buffer()).append_to(combine_ct_pos);
    TT_FATAL(
        desc.kernels.size() == kCombineKernelIdx,
        "moe_compute expert rows placement program: the combine is kernel {}, expected {}",
        desc.kernels.size(),
        kCombineKernelIdx);
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + "rows_combine.cpp",
        .core_ranges = core_set,
        .compile_time_args = std::move(combine_ct_pos),
        .named_compile_time_args = std::move(combine_ct),
        .defines = std::move(combine_defines),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    });
    auto& combine_kernel = desc.kernels.back();
    combine_kernel.emplace_common_runtime_args({output.buffer(), row_buffer.buffer(), table.buffer()});
    for (uint32_t sender = 0; sender < senders; ++sender) {
        tt::tt_metal::KernelDescriptor::RTArgList combine_args;
        combine_args.push_back(sender);
        if (ccl) {
            // persistent GlobalSemaphores held by the WorkloadDescriptor
            combine_args.push_back(static_cast<uint32_t>(init_barrier_semaphore->address()));   // smuggled-rta-ok
            combine_args.push_back(static_cast<uint32_t>(final_barrier_semaphore->address()));  // smuggled-rta-ok
            combine_args.push_back(static_cast<uint32_t>(sender == 0));
            combine_args.append(fabric->worker_runtime_args[sender]);
        }
        combine_kernel.emplace_runtime_args(cores[sender], combine_args);
    }
    if (ccl) {
        fabric->append_mux_kernels(desc);
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoEComputePlaceFactory::create_workload_descriptor(
    const MoEComputeParams& args,
    const MoEComputePlaceInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    std::optional<GlobalSemaphore> init_barrier_semaphore;
    std::optional<GlobalSemaphore> final_barrier_semaphore;
    if (args.path == MoEComputePath::FullCcl) {
        std::tie(init_barrier_semaphore, final_barrier_semaphore) = add_selective_reduce_combine_barrier_semaphores(
            workload,
            tensor_args.tilize_input_tensor.device(),
            CoreRangeSet(ttsl::Span<const CoreCoord>(args.combine_params->worker_cores)),
            args.combine_params->optional_cross_device_semaphore);
    }
    const std::vector<ttnn::MeshCoordinate> all_coordinates = tensor_coords.coords();
    for (const auto& coord : all_coordinates) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_place_program_descriptor(
                 args,
                 coord,
                 all_coordinates,
                 tensor_args,
                 tensor_return_value,
                 init_barrier_semaphore,
                 final_barrier_semaphore)});
    }
    return workload;
}

void MoEComputePlaceFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoEComputeParams& args,
    const MoEComputePlaceInputs& /*tensor_args*/,
    std::vector<ttnn::Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    // Buffer addresses are bindings. Only the caller-owned cross-device semaphore address can move.
    if (args.path != MoEComputePath::FullCcl || !args.combine_params->optional_cross_device_semaphore.has_value()) {
        return;
    }
    override_cross_device_semaphore_arg(
        program,
        kCombineKernelIdx,
        kCombineCrossDeviceSemaphoreArgIdx,
        *args.combine_params->optional_cross_device_semaphore);
}

MoEComputePlaceDeviceOperation::program_factory_t MoEComputePlaceDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEComputePlaceFactory{};
}

void MoEComputePlaceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    MoEComputeDeviceOperation::validate_on_program_cache_hit(args, tensor_args.moe_compute_inputs());
}

void MoEComputePlaceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    MoEComputeDeviceOperation::validate_on_program_cache_miss(args, tensor_args.moe_compute_inputs());
}

MoEComputePlaceDeviceOperation::spec_return_value_t MoEComputePlaceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return MoEComputeDeviceOperation::compute_output_specs(args, tensor_args.moe_compute_inputs());
}

MoEComputePlaceDeviceOperation::topology_return_value_t MoEComputePlaceDeviceOperation::compute_output_topologies(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return MoEComputeDeviceOperation::compute_output_topologies(args, tensor_args.moe_compute_inputs());
}

MoEComputePlaceDeviceOperation::tensor_return_value_t MoEComputePlaceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return MoEComputeDeviceOperation::create_output_tensors(args, tensor_args.moe_compute_inputs());
}

ttsl::hash::hash_t MoEComputePlaceDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    MoEComputeParams keyed = args;
    keyed.layer_id = 0;
    keyed.activation_type = detail::MoEActivationFunction::SILU;
    keyed.activation_limit = std::nullopt;
    keyed.math_fidelity = tt::tt_metal::MathFidelity::LoFi;
    keyed.fp32_dest_acc_en = false;
    return ttsl::hash::hash_objects_with_default_seed(keyed, tensor_args);
}

}  // namespace ttnn::experimental::prim
