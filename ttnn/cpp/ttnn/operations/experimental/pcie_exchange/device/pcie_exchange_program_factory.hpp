// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "pcie_exchange_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::pcie_exchange {

struct PcieExchangeProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::KernelHandle dm_kernel{};
        tt::tt_metal::KernelHandle compute_kernel{};
        bool has_compute = false;
        tt::tt_metal::CBHandle cb_in{};
        tt::tt_metal::CBHandle cb_landing{};
        tt::tt_metal::CBHandle cb_out{};
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const PcieExchangeParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const PcieExchangeInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const PcieExchangeParams& args,
        const PcieExchangeInputs& tensor_args,
        Tensor& output);

private:
    // Private on purpose: a public cached_program_t would select the single-program factory concept.
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const PcieExchangeParams& args,
        const ttnn::MeshCoordinate& coord,
        const PcieExchangeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::experimental::pcie_exchange
