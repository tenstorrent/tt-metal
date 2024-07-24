// SPDX-FileCopyrightText: © 2024 BOS
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace test_ops {

    Concat::program_factory_t Concat::select_program_factory(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        if (tensor_args[0].is_sharded())
            return MultiCore{};
        
        TT_FATAL(false, "This program currently only supports Multi-core, so please shard the input!")
        return SingleCore{};
    }

    void Concat::validate_on_program_cache_miss(const operation_attributes_t &attributes, const tensor_args_t &tensor_args) {
        Concat::validate_on_program_cache_hit(attributes, tensor_args);

        const auto &first_input = tensor_args.input_tensors[0];
        tt::tt_metal::Shape shape_first = first_input.get_legacy_shape();
        TT_FATAL(attributes.dim < shape_first.rank(), "Concat dim specified is larger than input tensor rank.");
        bool shard_first = tensor_args.input_tensors[0].is_sharded();

        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            TT_FATAL(tesnor_args.input_tensors[input_id].buffer(), "Operand to concat needs to be allocated to a buffer on device.");
            TT_FATAL(tesnor_args.input_tensors[input_id].device(), "Operand to concat needs to be on device.");
            TT_FATAL(tesnor_args.input_tensors[input_id].device() == first_input.device(), "Operands to concat need to be on the same device.");
            TT_FATAL(tesnor_args.input_tensors[input_id].get_layout() == first_input.get_layout(), "All Tensors should have same layouts.");
            TT_FATAL(tesnor_args.input_tensors[input_id].get_dtype() == first_input.get_dtype(), "All Tensors should have same dtypes.");

            if (tesnor_args.input_tensors[input_id].get_layout() == Layout::ROW_MAJOR && attributes.dim == shape_first.rank() - 1) {
                TT_FATAL(
                    (tesnor_args.input_tensors[input_id].get_legacy_shape()[attributes.dim] * tesnor_args.input_tensors[input_id].element_size()) % ADDRESS_ALIGNMENT == 0,
                    "This concat implementation requires aligned last dim when concatting on last dim");
            }

            TT_FATAL(tesnor_args.input_tensors[input_id].is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
            if (shard_first)
                TT_FATAL((tesnor_args.input_tensors[input_id].get_layout() == Layout::ROW_MAJOR), "Only row major supported for sharded concat.");
        }
    }

    void Concat::validate_on_program_cache_hit(const operation_attributes_t &attributes, const tensor_args_t &tensor_args) {
        const auto &first_input = tensor_args.input_tensors[0];
        const uint32_t num_input_tensors = tensor_args.input_tensors.size();
        
        TT_ASSERT(num_input_tensors > 1, "Must have at least 2 tensors to be concatenated!");

        tt::tt_metal::Shape shape_first = first_input.get_legacy_shape();
        shape_first[attributes.dim] = 0;
        for (uint32_t input_id = 1; input_id < num_input_tensors; input_id++){
            tt::tt_metal::Shape cur_shape = tensor_args.input_tensors[input_id].get_legacy_shape();
            cur_shape[attributes.dim] = 0;
            TT_ASSERT(curr_shape == shape_first, "Shape mismatch: Shape of Input tensor {} does not match Input Tensor 0", input_id);
        }
    }

    ttnn::Shape Concat::compute_output_shapes(const tensor_args_t& tensor_args, const uint32_t dim) {
        tt::tt_metal::Shape shape_out = tensor_args[0].get_legacy_shape();
        shape_out[dim] = 0;
        for (const Tensor &tensor : tensor_args) {
            tt::tt_metal::Shape cur_shape = tensor.get_legacy_shape();
            shape_out[dim] += cur_shape[dim];
        }
        return shape_out;
    }

    Tensor Concat::create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

        const Tensor &ref_in_tensor = tensor_args.at(0);
        auto output_shape = compute_output_shapes(tensor_args, operation_attributes.dim);

        return {create_device_tensor(
                            output_shape,
                            ref_in_tensor.get_dtype(),
                            ref_in_tensor.get_layout(),
                            ref_in_tensor.device(),
                            this->output_mem_config)
               };
    }


}
}
}