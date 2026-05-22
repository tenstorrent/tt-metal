// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "split_query_key_value_and_split_heads_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct SplitFusedQKVAndSplitHeadsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
        const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
