// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// rotary_embedding_indexed chain mode (bring-up option, deepseek_v41_flash_d_p P.22, ROPE_CHAINS): the RoPE of a
// head's tile-aligned channel region with the layout / dtype moves around it in ONE program:
//   - input_heads = H: the input is [B, 1, S, H W] TILE (a projection's output) and head h is read from tile columns
//     h W/32 .. (h + 1) W/32 - 1 in place (no nlp_create_q_heads split); or the input is [B, H, S, W] ROW_MAJOR
//     (tilized on read); or [B, H, S, W] TILE.
//   - output_layout ROW_MAJOR: [B, H, S, W] rows (untilized in the op), bf16.
//   - output_dtype bfloat8_b / bfloat4_b (TILE): the bf16 result packed onto that grid (bfloat8_b = ttnn.typecast of
//     the bf16 result; bfloat4_b = ttnn.bringup.bfp_round's round to nearest even).
//   - group_heads = G: the output [B, H / G, S, G W] TILE, head h at group h / G, columns (h % G) W.. (what
//     nlp_concat_heads of [B H / G, G, S, W] writes).
// The rotary arithmetic is the stock op's (rotary_embedding_llama compute), so the values equal the unfused chain's.
// Per-chip pre-sliced cos / sin (kv_actual_global 0, row 0); one program per mesh range; no per-core runtime args.

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::rotary_embedding_indexed {

struct RopeChainDeviceOperation {
    struct operation_attributes_t {
        uint32_t input_heads;  // H when the input is [B, 1, S, H W] (heads read in place); 0 = [B, H, S, W]
        uint32_t rotary_dim;
        uint32_t rotary_offset;
        tt::tt_metal::Layout output_layout;
        tt::tt_metal::DataType output_dtype;
        uint32_t group_heads;  // >= 1
        tt::tt_metal::MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
        std::optional<CoreCoord> max_grid;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
        std::optional<Tensor> output;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output);
        // Cache hit: the four input addresses and the output address (common runtime args) for this call.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& coord);
    };

    struct DescriptorAdapterOperation {
        using operation_attributes_t = RopeChainDeviceOperation::operation_attributes_t;
        using tensor_args_t = RopeChainDeviceOperation::tensor_args_t;
        using spec_return_value_t = RopeChainDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = RopeChainDeviceOperation::tensor_return_value_t;
    };

    struct MeshWorkloadFactory {
        using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
            DescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<ProgramFactory>;
        using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& args,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<MeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::rotary_embedding_indexed

namespace ttnn::prim::bringup {

ttnn::Tensor rotary_embedding_indexed_chain(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t input_heads,
    uint32_t rotary_dim,
    uint32_t rotary_offset,
    tt::tt_metal::Layout output_layout,
    tt::tt_metal::DataType output_dtype,
    uint32_t group_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    std::optional<CoreCoord> max_grid);

}  // namespace ttnn::prim::bringup
