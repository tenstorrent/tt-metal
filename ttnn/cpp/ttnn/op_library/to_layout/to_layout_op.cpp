// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/op_library/to_layout/to_layout_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_eager/tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {

namespace operations {

namespace core {

enum class ToLayoutProgramType {
    Tilize,
    PadAndTilize,
    Untilize,
    UntilizeAndUnpad,
};

const auto requires_padding_change = [](ttnn::Layout layout, const ttnn::Shape& shape) -> bool {
    const auto intended_shape = shape;
    const auto padded_shape = shape.with_tile_padding();
    if (layout == ttnn::ROW_MAJOR_LAYOUT and intended_shape != padded_shape) {
        return true;
    } else if (
        layout == ttnn::TILE_LAYOUT and (padded_shape.rank() < 2 or padded_shape[-1] % ttnn::TILE_SIZE != 0 or
                                         padded_shape[-2] % ttnn::TILE_SIZE != 0)) {
        return true;
    } else {
        return false;
    }
};

inline ToLayoutProgramType get_program_type(const ToLayout& operation, const std::vector<Tensor>& input_tensors) {
    const auto& tensor = input_tensors.at(0);
    const auto& layout = operation.program_config.layout;
    const auto& dtype = operation.program_config.dtype;

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        if (not requires_padding_change(layout, tensor.get_shape())) {
            return ToLayoutProgramType::Untilize;
        } else {
            return ToLayoutProgramType::UntilizeAndUnpad;
        }
    } else if (layout == ttnn::TILE_LAYOUT) {
        if (not requires_padding_change(layout, tensor.get_shape())) {
            return ToLayoutProgramType::Tilize;
        } else {
            return ToLayoutProgramType::PadAndTilize;
        }
    } else {
        TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
    }
}

void ToLayout::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& tensor = input_tensors.at(0);
    const auto& layout = this->program_config.layout;
    const auto& dtype = this->program_config.dtype;

    const std::set<ttnn::Layout> supported_layouts = {
        ttnn::ROW_MAJOR_LAYOUT,
        ttnn::TILE_LAYOUT,
    };

    if (supported_layouts.find(layout) == supported_layouts.end()) {
        TT_THROW(
            "ttnn::to_layout: Unsupported layout conversion from {} to {}!", input_tensors.at(0).get_layout(), layout);
    }

    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        TT_ASSERT(tensor.get_dtype() == dtype, "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
    } else if (layout == ttnn::TILE_LAYOUT) {
        if (not requires_padding_change(layout, tensor.get_shape())) {
            if (tensor.is_sharded()) {
                const auto shard_shape = get_memory_config(tensor).value().shard_spec.value().shape;
                if (shard_shape[0] % ttnn::TILE_SIZE != 0 or shard_shape[1] % ttnn::TILE_SIZE != 0) {
                    TT_THROW(
                        "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                        "TILE_SIZE!");
                }
            }
        }
    } else {
        TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
    }
}

std::vector<tt::tt_metal::Shape> ToLayout::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& tensor = input_tensors.at(0);
    const auto& layout = this->program_config.layout;

    const auto intended_shape = tensor.get_shape();

    std::vector<uint32_t> output_shape;
    if (layout == ttnn::TILE_LAYOUT and intended_shape.rank() < 2) {
        output_shape.push_back(1);
    }
    for (auto index = 0; index < intended_shape.rank(); ++index) {
        output_shape.push_back(intended_shape[index]);
    }

    auto padded_output_shape = output_shape;
    for (auto index = output_shape.size() - 2; index < output_shape.size(); ++index) {
        padded_output_shape[index] = ttnn::pad_to_multiple_of_tile_size(padded_output_shape[index]);
    }

    auto program_type = get_program_type(*this, input_tensors);
    switch (program_type) {
        case ToLayoutProgramType::Tilize: return {tt::tt_metal::Shape{output_shape, padded_output_shape}};
        case ToLayoutProgramType::PadAndTilize: return {tt::tt_metal::Shape{output_shape, padded_output_shape}};
        case ToLayoutProgramType::Untilize: return {tt::tt_metal::Shape{output_shape}};
        case ToLayoutProgramType::UntilizeAndUnpad: return {tt::tt_metal::Shape{output_shape}};
        default: {
            TT_THROW("ttnn::to_layout: Unsupported program type!");
        }
    }
}

std::vector<Tensor> ToLayout::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    if (this->program_config.memory_config.is_sharded()) {
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->program_config.dtype,
            this->program_config.layout,
            input_tensors.at(0).device(),
            this->program_config.memory_config)};
    }
    return operation::generic_create_output_tensors(
        *this,
        input_tensors,
        this->program_config.dtype,
        this->program_config.layout,
        this->program_config.memory_config);
}

operation::ProgramWithCallbacks ToLayout::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto program_type = get_program_type(*this, input_tensors);
    switch (program_type) {
        case ToLayoutProgramType::Tilize: {
            return tilize_multi_core(input_tensor, output_tensor);
        }
        case ToLayoutProgramType::PadAndTilize: {
            std::vector<uint32_t> input_tensor_start{};  // unused
            std::vector<uint32_t> output_shape{};        // unused
            if (not input_tensor.is_sharded() or not output_tensor.is_sharded()) {
                return tilize_with_val_padding_single_core(
                    input_tensor, output_tensor, output_shape, input_tensor_start, 0);
            }
            return tilize_with_val_padding_multi_core(input_tensor, output_tensor, output_shape, input_tensor_start, 0);
        }
        case ToLayoutProgramType::Untilize: {
            bool use_pack_untilize = true;
            return untilize_multi_core(
                input_tensor, output_tensor, use_pack_untilize, get_fp32_dest_acc_en(this->compute_kernel_config));
        }
        case ToLayoutProgramType::UntilizeAndUnpad: {
            bool use_pack_untilize = true;
            std::vector<uint32_t> output_tensor_start{};  // unused
            std::vector<uint32_t> output_tensor_end{};    // unused'
            if (not input_tensor.is_sharded() or not output_tensor.is_sharded()) {
                return untilize_with_unpadding_single_core(
                    input_tensor,
                    output_tensor,
                    output_tensor_start,
                    output_tensor_end,
                    use_pack_untilize,
                    get_fp32_dest_acc_en(this->compute_kernel_config));
            }
            return untilize_with_unpadding_multi_core(
                input_tensor,
                output_tensor,
                output_tensor_start,
                output_tensor_end,
                use_pack_untilize,
                get_fp32_dest_acc_en(this->compute_kernel_config));
        }
        default: {
            TT_THROW("ttnn::to_layout: Unsupported program type!");
        }
    }
    return {};
}

Tensor to_layout(
    const Tensor& input_tensor,
    const Layout layout,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [layout, memory_config, dtype](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors) mutable -> std::vector<Tensor> {
            auto input_tensor = input_tensors.at(0);

            if (input_tensor.get_layout() == layout) {
#ifdef DEBUG
                if (dtype.has_value() and dtype.value() != input_tensor.get_dtype()) {
                    tt::log_warning(
                        tt::LogOp,
                        "ttnn::to_layout: dtype is specified but the input_tensor is already in the requested "
                        "layout! "
                        "So, "
                        "the dtype won't be changed!");
                }
                if (memory_config.has_value() and memory_config.value() != get_memory_config(input_tensor).value()) {
                    tt::log_warning(
                        tt::LogOp,
                        "ttnn::to_layout: memory_config is specified but the input_tensor is already in the "
                        "requested "
                        "layout! So, the memory_config won't be changed!");
                }
#endif
                return {input_tensor};
            }
            if (ttnn::is_tensor_on_device_or_multidevice(input_tensor)) {
                return operation::run(
                    ToLayout{ToLayoutProgramConfig{
                        layout,
                        memory_config.value_or(input_tensor.memory_config()),
                        dtype.value_or(input_tensor.get_dtype())}},
                    {input_tensors.at(0)});

            } else {
                const auto intended_shape = input_tensor.get_shape();

                std::vector<uint32_t> output_shape;
                if (layout == ttnn::TILE_LAYOUT and intended_shape.rank() < 2) {
                    output_shape.push_back(1);
                }
                for (auto index = 0; index < intended_shape.rank(); ++index) {
                    output_shape.push_back(intended_shape[index]);
                }

                auto padded_output_shape = output_shape;
                for (auto index = output_shape.size() - 2; index < output_shape.size(); ++index) {
                    padded_output_shape[index] = ttnn::pad_to_multiple_of_tile_size(padded_output_shape[index]);
                }

                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting layout on host!");
                if (not requires_padding_change(layout, input_tensor.get_shape())) {
                    return {input_tensor.to(layout)};

                } else if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                    auto tensor = ttnn::unsqueeze_to_4D(input_tensor);
                    tensor = tensor.to(layout);
                    tensor = tensor.unpad_from_tile(tensor.get_shape().value().without_padding());
                    return {ttnn::reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape}))};

                } else if (layout == ttnn::TILE_LAYOUT) {
                    auto tensor = ttnn::unsqueeze_to_4D(input_tensor);
                    std::vector<uint32_t> padded_4D_output_shape;
                    padded_4D_output_shape.push_back(tensor.get_shape()[-4]);
                    padded_4D_output_shape.push_back(tensor.get_shape()[-3]);
                    padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-2]));
                    padded_4D_output_shape.push_back(ttnn::pad_to_multiple_of_tile_size(tensor.get_shape()[-1]));
                    tensor = tensor.pad(padded_4D_output_shape, {0, 0, 0, 0}, 0).to(layout);
                    return {ttnn::reshape(tensor, ttnn::Shape(tt::tt_metal::Shape{output_shape, padded_output_shape}))};

                } else {
                    TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
                }
            }
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace core

}  // namespace operations

}  // namespace ttnn
