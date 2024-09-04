// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::experimental::matmul {


void AttnMatmulDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]

    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to matmul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    TT_FATAL((ashape[0] == 1), "Input q_len must be 1!");
    TT_FATAL((bshape[1] == 1), "Number of kv_heads must be 1!");  // TODO: May need to uplift to support falcon-40B
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");

    bool read_from_kv_cache = false;
    if (this->num_tokens.has_value() or this->transpose_hw.has_value()) {
        TT_FATAL(
            (this->num_tokens.has_value() and this->transpose_hw.has_value()),
            "Must provide num_tokens and transpose_hw flag if we are reading from cache for in1!");
        TT_FATAL(this->num_tokens.value() % 32 == 0, "Number of tokens must be divisble by 32!");
        read_from_kv_cache = true;
    }

    if (read_from_kv_cache) {
        if (this->transpose_hw.value()) {
            TT_FATAL(
                ashape[3] == bshape[3] &&
                "For pre-attention matmul, dimension K for B is in B.shape[3], so A.shape[3] must match B.shape[3]");  // A.K == B.K
        } else {
            TT_FATAL(
                ashape[3] == this->num_tokens &&
                "For post-attention matmul, dimension K (A.shape[3]) is the kv_seq_len in this case and must match the "
                "length of the cache we read");  // A.K == B.K
        }
    } else {
        TT_FATAL(
            ashape[3] == bshape[2] &&
            "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in attn_matmul op");  // A.K == B.K
    }
}

std::vector<tt::tt_metal::Shape> AttnMatmulDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();

    uint32_t N = bshape[3];
    if (this->transpose_hw.value_or(false)) {
        N = this->num_tokens.value();
    }

    return {tt::tt_metal::Shape{1, ashape[1], ashape[2], N}};
}

std::vector<Tensor> AttnMatmulDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {create_device_tensor(
            compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            Layout::TILE,
            input_tensor.device(),
            this->output_mem_config)};
}

operation::ProgramWithCallbacks AttnMatmulDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    return multi_core_attn_matmul(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        this->num_tokens,
        this->transpose_hw,
        this->compute_with_storage_grid_size,
        this->compute_kernel_config);
}

const operation::Hash AttnMatmulDeviceOperation::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensors.at(0).storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensors.at(0).get_storage()),__FILE__, __LINE__));
    TT_ASSERT(std::holds_alternative<DeviceStorage>(input_tensors.at(1).storage()), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(input_tensors.at(1).get_storage()),__FILE__, __LINE__));
    return operation::hash_operation<AttnMatmulDeviceOperation>(
        this->transpose_hw,
        this->output_mem_config,
        this->output_dtype,
        std::get<DeviceStorage>(input_tensors.at(0).storage()).memory_config(),
        input_tensors.at(0).dtype(),
        std::get<DeviceStorage>(input_tensors.at(1).storage()).memory_config(),
        input_tensors.at(1).dtype());
}


}  // namespace ttnn::operations::experimental::transformer
