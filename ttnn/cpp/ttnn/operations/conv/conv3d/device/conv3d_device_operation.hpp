#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
// #include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::conv::conv3d {

namespace detail {
inline std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& kernel_size) {
    uint32_t T_out = T_in + 2 * padding[0] - (kernel_size[0] - 1);
    uint32_t H_out = H_in + 2 * padding[1] - (kernel_size[1] - 1);
    uint32_t W_out = W_in + 2 * padding[2] - (kernel_size[2] - 1);
    return {T_out, H_out, W_out};
}
}  // namespace detail

struct Conv3dConfig {
    // This constructor matches the PyBind11 .def(...)
    // It also provides default values so you can call it with fewer arguments if desired.
    Conv3dConfig(
        DataType dtype_ = DataType::BFLOAT16,
        DataType weights_dtype_ = DataType::BFLOAT16,
        Layout output_layout_ = Layout::ROW_MAJOR,
        uint32_t T_out_block_ = 1,
        uint32_t W_out_block_ = 1,
        uint32_t H_out_block_ = 1,
        uint32_t C_out_block_ = 0,
        uint32_t C_in_block_ = 0,
        uint32_t output_channels_ = 0,
        const std::array<uint32_t, 3> kernel_size_ = {1, 1, 1},
        const std::array<uint32_t, 3> stride_ = {1, 1, 1},
        const std::array<uint32_t, 3> padding_ = {0, 0, 0},
        const std::string padding_mode_ = "zeros",
        uint32_t groups_ = 1,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        dtype(dtype_),
        weights_dtype(weights_dtype_),
        output_layout(output_layout_),
        T_out_block(T_out_block_),
        W_out_block(W_out_block_),
        H_out_block(H_out_block_),
        C_out_block(C_out_block_),
        C_in_block(C_in_block_),
        output_channels(output_channels_),
        kernel_size(kernel_size_),
        stride(stride_),
        padding(padding_),
        padding_mode(padding_mode_),
        groups(groups_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    DataType dtype;
    DataType weights_dtype;
    Layout output_layout;
    uint32_t T_out_block;
    uint32_t W_out_block;
    uint32_t H_out_block;
    uint32_t C_out_block;
    uint32_t C_in_block;
    uint32_t output_channels;
    std::array<uint32_t, 3> kernel_size;
    std::array<uint32_t, 3> stride;
    std::array<uint32_t, 3> padding;
    std::string padding_mode;
    uint32_t groups;
    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "dtype",
        "weights_dtype",
        "output_layout",
        "T_out_block",
        "W_out_block",
        "H_out_block",
        "C_out_block",
        "C_in_block",
        "output_channels",
        "kernel_size",
        "stride",
        "padding",
        "padding_mode",
        "groups",
        "compute_with_storage_grid_size");

    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->dtype),
            std::cref(this->weights_dtype),
            std::cref(this->output_layout),
            std::cref(this->T_out_block),
            std::cref(this->W_out_block),
            std::cref(this->H_out_block),
            std::cref(this->C_out_block),
            std::cref(this->C_in_block),
            std::cref(this->output_channels),
            std::cref(this->kernel_size),
            std::cref(this->stride),
            std::cref(this->padding),
            std::cref(this->padding_mode),
            std::cref(this->groups),
            std::cref(this->compute_with_storage_grid_size));
    }
};

struct Conv3dOp {
    Conv3dConfig config;
    MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::conv::conv3d
