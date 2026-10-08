// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"

#include "exchange_histories_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct ExchangeHistoriesProgramFactory {
    // Allocates the line barrier and arrival semaphores once per workload, then one program per coordinate:
    // each program's fabric routes depend on the device's place on the sequence-parallel line.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ExchangeHistoriesParams&,
        const ExchangeHistoriesInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
