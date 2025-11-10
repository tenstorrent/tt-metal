// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {

namespace operations {
namespace core {

struct ToDeviceOperation : public ttnn::experimental::lazy::LazyOperation {
    using tensor_args_t = Tensor;
    distributed::MeshDevice* mesh_device_;
    const MemoryConfig mem_config_;
    std::optional<ttnn::QueueId> cq_id_;

    ToDeviceOperation(
        distributed::MeshDevice* mesh_device, MemoryConfig&& mem_config, std::optional<ttnn::QueueId> cq_id) :
        mesh_device_(mesh_device), mem_config_(std::move(mem_config)), cq_id_(cq_id) {}

    virtual std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const ttnn::experimental::lazy::LazyOperationInputs& inputs) override {
        TT_FATAL(inputs.size() == 1, "ToDeviceOperation expects exactly one input");
        TT_FATAL(inputs.at(0)->is_materialized(), "We need a materialized tensor to move to device");
        std::optional<MemoryConfig> mem_config = std::make_optional(mem_config_);
        return {inputs.at(0)->materialized_tensor().to_device(mesh_device_, std::move(mem_config), cq_id_)};
    }

    virtual std::string_view name() const override { return "ToDeviceOperation"; }

    virtual tt::stl::hash::hash_t operation_type_id() const override {
        return tt::stl::hash::type_hash<ToDeviceOperation>;
    }

    virtual ~ToDeviceOperation() = default;
};

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor);

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, int rank);

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<QueueId> queue_id = std::nullopt);

ttnn::Tensor from_device(
    const ttnn::Tensor& tensor, bool blocking = true, std::optional<QueueId> queue_id = std::nullopt);

void deallocate(Tensor& tensor, bool force = true);

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);

}  // namespace core
}  // namespace operations

using operations::core::deallocate;
using operations::core::from_device;
using operations::core::reallocate;
using operations::core::squeeze_from_4D;
using operations::core::to_device;
using operations::core::unsqueeze_to_4D;

constexpr auto to_dtype = ttnn::register_operation<"ttnn::to_dtype", ttnn::operations::core::ToDtype>();
constexpr auto to_memory_config =
    ttnn::register_operation<"ttnn::to_memory_config", ttnn::operations::core::ToMemoryConfig>();
constexpr auto to_layout = ttnn::register_operation<"ttnn::to_layout", ttnn::operations::core::ToLayout>();

}  // namespace ttnn
