// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/distributed/tensor_topology.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(
        Storage storage,
        TensorSpec tensor_spec,
        DistributedTensorConfig distributed_tensor_config,
        TensorTopology tensor_topology);

    // Getters and setters.
    const Storage& get_storage() const;
    Storage& get_storage();
    const TensorSpec& get_tensor_spec() const;
    const DistributedTensorConfig& get_distributed_tensor_config() const;
    const TensorTopology& get_tensor_topology() const;

private:
    Storage storage_;
    TensorSpec tensor_spec_;
    DistributedTensorConfig distributed_tensor_config_;
    TensorTopology tensor_topology_;
};

}  // namespace tt::tt_metal
