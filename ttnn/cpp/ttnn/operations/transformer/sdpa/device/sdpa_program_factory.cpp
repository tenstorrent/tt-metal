// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_dense.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"

namespace ttnn::prim {

// Dense, chunked, MLA and windowed prefill SDPA: the precision recipe program (device/recipe/). Buffer addresses are
// common runtime args bound to their buffers, so a program-cache hit patches them without calling this.
tt::tt_metal::ProgramDescriptor SDPAOperation::SDPAProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return ttnn::operations::transformer::sdpa::detail::dense_recipe_descriptor(
        operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::prim
