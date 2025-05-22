#include "ttnn/operations/experimental/ccl/reduce_scatter_matmul/device/rs_matmul_op.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl {

void AllGatherRS::validate_on_program_cache_hit(
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    auto& input_tensor = input_tensors[0];
    auto& weight_tensor = input_tensors[1];
    this->matmul_struct.validate({input_tensor, weight_tensor}, {std::nullopt}, {});
    this->rs_struct.validate_on_program_cache_hit(operation_attributes, tensor_args);
}

void AllGatherRS::validate_on_program_cache_miss(
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    auto& input_tensor = input_tensors[0];
    auto& weight_tensor = input_tensors[1];
    this->matmul_struct.validate({input_tensor, weight_tensor}, {std::nullopt}, {});
    this->rs_struct.validate_on_program_cache_miss(operation_attributes, tensor_args);
}

std::vector<ttnn::TensorSpec> AllGatherRS::compute_output_specs(
    const std::vector<Tensor>& input_tensors,
    const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
    const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args) {
    // All Gather shape
    ttnn::TensorSpec reduce_scatter_output_spec =
        this->rs_struct.compute_output_specs(operation_attributes, tensor_args);
    auto& input_tensor = input_tensors[0];
    auto& weight_tensor = input_tensors[1];

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensor, weight_tensor}, {})[0];

    return {matmul_output_specs, reduce_scatter_output_spec};
}

std::vector<Tensor> rs_matmul(
    const ttnn::Tensor& input_tensor,                           // mm0 used
    const ttnn::Tensor& weight_tensor,                          // mm1 used
    const ttnn::Tensor& rs_tensor,                              // rs1
    ttnn::Tensor& intermediate_packet_buffer,                   // rs2
    uint32_t dim,                                               // rs3
    const GlobalSemaphore& cross_device_semaphore,              // rs4
    const uint32_t cluster_axis,                                // rs 5
    const MeshDevice& mesh_device,                              // rs 6
    const uint32_t num_links,                                   // rs 7 default 1
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,  // rs 8 default std::nullopt
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,  // mm4 used but default std::nullopt
    const std::optional<const ttnn::DeviceComputeKernelConfig>
        compute_kernel_config,                                      // mm8 used but default std::nullopt
    const std::optional<const GlobalCircularBuffer>& global_cb,     // mm12 used but default std::nullopt
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,  // rs and mm13 used same but default std::nullopt
    const std::optional<const ttnn::CoreGrid> core_grid,            // mm9 may use but default std::nullopt
    const bool transpose_a,                                         // mm2 set false
    const bool transpose_b,                                         // mm3 set false
    const std::optional<const DataType> dtype,                      // mm5 set false
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,  // mm6 std::nullopt
    const std::optional<const std::string>& activation,                                  // mm7 set false
    const std::optional<const tt::tt_metal::Tile>& output_tile,                          // mm10 std::nullopt
    const std::optional<Tensor>& optional_output_tensor                                  // mm11 std::nullopt
) {
    return {input_tensor};
}
}  // namespace ttnn::operations::experimental::ccl
