// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/operations/experimental/high_bw_all_gather/device/high_bw_all_gather_device_operation.hpp"
#include "fabric_all_gather_device_operation_types.hpp"
#include "fabric_all_gather_factory.hpp"

namespace ttnn::operations::experimental::fabric_all_gather {

// high_bw_all_gather's operation (validation, program hash, output spec and topology) with this op's program. The
// operation type is part of the program-cache key, so the two ops never share a cached program.
struct FabricAllGatherDeviceOperation : high_bw_all_gather::HighBwAllGatherDeviceOperation {
    using program_factory_t = std::variant<FabricAllGatherFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return FabricAllGatherFactory{};
    }
};

}  // namespace ttnn::operations::experimental::fabric_all_gather
