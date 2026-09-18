// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_schedule_test.hpp"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::experimental::prim {

struct SpMatmulScheduleTestParams {
    ttnn::prim::MatmulParams matmul_struct;
    std::vector<uint32_t> schedule_words;
    bool ag_mode = false;

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "schedule_words", "ag_mode");
    auto attribute_values() const {
        return std::forward_as_tuple(this->matmul_struct, this->schedule_words, this->ag_mode);
    }
};

struct SpMatmulScheduleTestInputs {
    Tensor input;
    Tensor weight;
};

struct SpMatmulScheduleTestProgramFactory {
    using shared_variables_t = ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SpMatmulScheduleTestParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SpMatmulScheduleTestInputs& tensor_args,
        Tensor& output) {
        tt::tt_metal::distributed::MeshWorkload mesh_workload;
        std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;
        for (const auto& coord : tensor_coords.coords()) {
            auto cached_program = create_at(args, coord, tensor_args, output);
            mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
            shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
        }
        return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
    }

private:
    // Private like the other fused CCL factories: a public `cached_program_t` would make the single-device
    // ProgramFactoryConcept check proceed into its (non-SFINAE) lambda and hard-error on the missing `create`.
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const SpMatmulScheduleTestParams& args,
        const ttnn::MeshCoordinate& /*mesh_coord*/,
        const SpMatmulScheduleTestInputs& tensor_args,
        Tensor& output) {
        using namespace ttnn::experimental::ccl;

        const auto& program_config = args.matmul_struct.program_config.value();
        const auto& cfg = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(program_config);
        tt::tt_metal::Program program{};
        tt::tt_metal::IDevice* device = tensor_args.input.device();

        std::optional<MatmulFusedOpSignaler> matmul_fused_op_signaler;
        if (args.ag_mode) {
            // SP_ALL_GATHER without an all-gather: the two direction semaphores are created by the matmul factory
            // (init_fused_op on the in0 sender cores) and stay at 0; the schedule's wait_count is 0 (validated), and
            // the "original sharded input" read for is_local iterations is the input itself.
            matmul_fused_op_signaler = MatmulFusedOpSignaler(MatmulFusedOpSignalerType::SP_ALL_GATHER);
            matmul_fused_op_signaler->init_sp_schedule(
                args.schedule_words, static_cast<uint32_t>(tensor_args.input.buffer()->address()));
        } else {
            // The SP_REDUCE_SCATTER path does a per-sub-batch all-core barrier + semaphore increment on the RS
            // receiver cores. Without a CCL, aim it at one dummy core outside the matmul grid (nobody waits on it).
            const CoreCoord grid = cfg.compute_with_storage_grid_size;
            const CoreCoord device_grid = device->compute_with_storage_grid_size();
            CoreCoord dummy_core;
            if (grid.x < device_grid.x) {
                dummy_core = CoreCoord(grid.x, 0);
            } else if (grid.y < device_grid.y) {
                dummy_core = CoreCoord(0, grid.y);
            } else {
                TT_THROW(
                    "sp_matmul_schedule_test: matmul grid {}x{} fills the device grid {}x{}; no room for the dummy RS "
                    "signal core",
                    grid.x,
                    grid.y,
                    device_grid.x,
                    device_grid.y);
            }
            ReduceScatterFusedOpSignaler rs_signaler;
            rs_signaler.init_reduce_scatter(program, device, CoreRange(dummy_core, dummy_core));
            rs_signaler.init_fused_op();

            matmul_fused_op_signaler = MatmulFusedOpSignaler(MatmulFusedOpSignalerType::SP_REDUCE_SCATTER);
            matmul_fused_op_signaler->init_reduce_scatter(
                rs_signaler.fused_op_receiver_cores_noc,
                rs_signaler.fused_op_receiver_signal_semaphores,
                rs_signaler.fused_op_signaler_mode);
            matmul_fused_op_signaler->init_sp_schedule(args.schedule_words);
        }

        auto matmul_cached_program = ttnn::prim::matmul_multi_core_reuse_mcast_2d_optimized_helper(
            program,
            tensor_args.input,
            tensor_args.weight,
            /*bias=*/std::nullopt,
            output,
            args.matmul_struct.bcast_batch.value(),
            args.matmul_struct.compute_kernel_config.value(),
            program_config,
            args.matmul_struct.untilize_out,
            matmul_fused_op_signaler,
            args.matmul_struct.transpose_a,
            args.matmul_struct.transpose_b);

        return cached_program_t{
            std::move(matmul_cached_program.program), std::move(matmul_cached_program.shared_variables)};
    }

public:
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SpMatmulScheduleTestParams& args,
        const SpMatmulScheduleTestInputs& tensor_args,
        Tensor& output) {
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
            std::vector<Tensor> matmul_output_tensors = {output};
            ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
                program,
                shared_vars,
                args.matmul_struct,
                {.input_tensors = {tensor_args.input, tensor_args.weight},
                 .optional_input_tensors = {std::optional<const Tensor>{}},
                 .optional_output_tensors = {}},
                matmul_output_tensors);
            if (args.ag_mode) {
                // The matmul's own override does not know about the alternate in0 buffer.
                ttnn::prim::override_sp_in0_alt_addr(
                    program, shared_vars, static_cast<uint32_t>(tensor_args.input.buffer()->address()));
            }
        }
    }
};

struct SpMatmulScheduleTestDeviceOperation {
    using operation_attributes_t = SpMatmulScheduleTestParams;
    using tensor_args_t = SpMatmulScheduleTestInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SpMatmulScheduleTestProgramFactory>;
    using shared_variables_t = SpMatmulScheduleTestProgramFactory::shared_variables_t;

    static ttnn::prim::MatmulInputs matmul_inputs(const tensor_args_t& tensor_args) {
        return {
            .input_tensors = {tensor_args.input, tensor_args.weight},
            .optional_input_tensors = {std::optional<const Tensor>{}},
            .optional_output_tensors = {}};
    }

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
        ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(
            args.matmul_struct, matmul_inputs(tensor_args));
        TT_FATAL(
            args.matmul_struct.program_config.has_value() &&
                std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
                    args.matmul_struct.program_config.value()),
            "sp_matmul_schedule_test needs a MatmulMultiCoreReuseMultiCastProgramConfig");
        const auto& cfg = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
            args.matmul_struct.program_config.value());
        TT_FATAL(!cfg.fuse_batch, "sp_matmul_schedule_test needs fuse_batch=false");

        const auto a_shape_padded =
            operations::matmul::utilities::get_matmul_tensor_padded_shape(tensor_args.input, false);
        const uint32_t num_sub_batches =
            static_cast<uint32_t>(a_shape_padded.volume() / (a_shape_padded[-1] * a_shape_padded[-2]));
        const auto& words = args.schedule_words;
        TT_FATAL(
            words.size() == num_sub_batches,
            "sp_matmul_schedule_test: {} schedule words for {} sub-batches",
            words.size(),
            num_sub_batches);
        std::vector<bool> seen_in0(num_sub_batches, false);
        std::vector<bool> seen_out(num_sub_batches, false);
        for (uint32_t w : words) {
            const uint32_t in0_idx = w & 0xFF;
            const uint32_t out_idx = (w >> 8) & 0xFF;
            TT_FATAL(in0_idx < num_sub_batches && !seen_in0[in0_idx], "in0_idx {} not a permutation", in0_idx);
            TT_FATAL(out_idx < num_sub_batches && !seen_out[out_idx], "out_idx {} not a permutation", out_idx);
            if (args.ag_mode) {
                TT_FATAL(
                    (w >> 24) == 0,
                    "sp_matmul_schedule_test(ag_mode): wait_count must be 0 (nobody increments the semaphores), got "
                    "word {:#x}",
                    w);
            } else {
                TT_FATAL(
                    (w >> 16) == 0, "sp_matmul_schedule_test: wait/local bits must be 0 (no CCL), got word {:#x}", w);
            }
            seen_in0[in0_idx] = true;
            seen_out[out_idx] = true;
        }
    }

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args) {
        return ttnn::prim::MatmulDeviceOperation::compute_output_specs(
            args.matmul_struct, {.input_tensors = {tensor_args.input, tensor_args.weight}})[0];
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args) {
        return ttnn::prim::MatmulDeviceOperation::create_output_tensors(
            args.matmul_struct, {.input_tensors = {tensor_args.input, tensor_args.weight}})[0];
    }

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args) {
        return tt::tt_metal::operation::hash_operation<SpMatmulScheduleTestDeviceOperation>(
            args.matmul_struct, args.schedule_words, args.ag_mode, tensor_args.input, tensor_args.weight);
    }
};

}  // namespace ttnn::experimental::prim

namespace ttnn::experimental {

Tensor sp_matmul_schedule_test(
    const Tensor& in0_view,
    const Tensor& in1,
    const std::vector<uint32_t>& schedule_words,
    bool transpose_b,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    bool ag_mode) {
    using OperationType = ttnn::experimental::prim::SpMatmulScheduleTestDeviceOperation;
    TT_FATAL(in0_view.device() != nullptr, "sp_matmul_schedule_test: input must be on device");

    operations::matmul::MatmulProgramConfig pc = program_config;
    operations::matmul::normalize_program_config(pc, in0_view.device()->compute_with_storage_grid_size());

    ttnn::prim::MatmulParams params;
    params.program_config = pc;
    params.output_mem_config = memory_config.value_or(in0_view.memory_config());
    params.output_dtype = in0_view.dtype();
    if (compute_kernel_config.has_value()) {
        params.compute_kernel_config = *compute_kernel_config;
    }
    params.transpose_b = transpose_b;
    auto matmul_struct = ttnn::prim::create_matmul_attributes(in0_view, in1, params, {});

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{std::move(matmul_struct), schedule_words, ag_mode},
        OperationType::tensor_args_t{.input = in0_view, .weight = in1});
}

}  // namespace ttnn::experimental
