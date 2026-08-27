// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
namespace ttnn::operations::op_slicing {

struct Op2DSliceConfig {
    // Determines the dimension along which the input & output tensors are sliced.
    // Slices based on [N, H, W, C] shape.
    // Using width slicing is more efficient as it reduces memory usage. This is because the overlap of data between
    // cores is minimized in width slicing, reducing the size of the Halo output. If the Height & Width dimensions are
    // similar, then use Width slicing. Use Height slicing if the Height dimension is significantly larger than the
    // Width dimension.
    enum class SliceType : uint8_t {
        DRAM_HEIGHT,
        DRAM_WIDTH,
        L1_FULL,  // This option can be used to force conv2d with a DRAM Input to move it to L1, and output will be in
                  // L1.
        // Appended rather than grouped with the DRAM_* values above so the existing enumerators keep
        // their numeric values.
        DRAM_CHANNEL  // Slice along the channel dimension. Only legal for ops with no cross-channel reduction (see
                      // OpSliceAttr::channel_slice_granularity). Unlike height/width slicing there is no halo, so no
                      // data is duplicated between slices; it is the only usable axis when both spatial dimensions are
                      // too small to slice (e.g. a wide depthwise short convolution, output 1x23x8192).
    };
    SliceType slice_type = SliceType::DRAM_WIDTH;

    // Number of slices that the output tensor should be divided into.
    uint32_t num_slices = 0;
};

class OpSliceAttr {
public:
    using OptionalRefTensor = std::optional<std::reference_wrapper<ttnn::Tensor>>;
    using RefTensor = std::reference_wrapper<ttnn::Tensor>;

    virtual ~OpSliceAttr() = default;
    using IOShape = std::tuple<uint32_t, uint32_t>;
    virtual std::tuple<IOShape, IOShape> get_input_slice(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const = 0;

    virtual uint32_t get_L1_usage(
        const IOShape& output_slice_start,
        const IOShape& output_slice_end,
        const op_slicing::Op2DSliceConfig& slice_config) const = 0;

    virtual tt::tt_metal::MemoryConfig get_input_memory_config(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const = 0;
    virtual std::vector<ttnn::Tensor> run_L1_op(
        const ttnn::Tensor& sliced_input_tensor,
        const IOShape& output_slice_start,
        const IOShape& output_slice_end) = 0;
    virtual std::string name() const = 0;

    // ---- Channel slicing (SliceType::DRAM_CHANNEL) -----------------------------------------------
    // Opt-in: an op supports channel slicing only if each output channel can be computed without
    // reducing over the channels held by other slices. True for pooling (fully per-channel) and for
    // depthwise/grouped convolution when slices land on group boundaries; false for a dense
    // convolution, where every output channel reduces over all input channels.
    //
    // Returns the required channel alignment of a slice boundary, or 0 if the op does not support
    // channel slicing (the default, so existing ops keep their spatial-only behaviour).
    virtual uint32_t channel_slice_granularity() const;

    // As get_L1_usage / run_L1_op, but for a slice covering the full spatial extent and the
    // half-open output channel range [channel_start, channel_end). Only called when
    // channel_slice_granularity() > 0.
    virtual uint32_t get_L1_usage_for_channel_slice(
        uint32_t channel_start, uint32_t channel_end, const op_slicing::Op2DSliceConfig& slice_config) const;
    virtual std::vector<ttnn::Tensor> run_L1_op_channel_slice(
        const ttnn::Tensor& sliced_input_tensor, uint32_t channel_start, uint32_t channel_end);
};

Op2DSliceConfig determine_slice_config(
    OpSliceAttr* op_slice_attr,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& output_shape,
    std::optional<Op2DSliceConfig> slice_config_,
    tt::tt_metal::Layout output_layout,
    tt::tt_metal::distributed::MeshDevice* device);

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensor,
    OpSliceAttr* op_slice_attr,
    std::optional<Op2DSliceConfig> dram_slice_config_);

}  // namespace ttnn::operations::op_slicing
