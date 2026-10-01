// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "pcie_exchange_device_operation_types.hpp"
#include "pcie_exchange_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::pcie_exchange {

struct PcieExchangeDeviceOperation {
    using operation_attributes_t = PcieExchangeParams;
    using tensor_args_t = PcieExchangeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PcieExchangeProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Shard cores of a width-sharded tensor in shard order (row-major grid walk).
    static std::vector<tt::tt_metal::CoreCoord> shard_cores(const Tensor& t);
    // Output core set for ALL_GATHER: the first num_devices * N cores of the compute grid, row-major.
    static tt::tt_metal::CoreRangeSet gather_output_grid(const Tensor& input, uint32_t num_devices);
};

}  // namespace ttnn::operations::experimental::pcie_exchange

namespace ttnn::prim {

Tensor pcie_exchange(
    const Tensor& input,
    const Tensor& exchange,
    ttnn::operations::experimental::pcie_exchange::PcieExchangeMode mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& ctrl = std::nullopt,
    bool bfp8_payload = false);

}  // namespace ttnn::prim
