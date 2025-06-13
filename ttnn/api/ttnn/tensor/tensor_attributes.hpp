// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(Storage storage, TensorSpec tensor_spec, DistributedTensorConfig distributed_tensor_config);

    // Getters and setters.
    const Storage& get_storage() const;
    Storage& get_storage();
    const TensorSpec& get_tensor_spec() const;
    const DistributedTensorConfig& get_distributed_tensor_config() const;

private:
    Storage storage_;
    TensorSpec tensor_spec_;
    DistributedTensorConfig distributed_tensor_config_;
};

}  // namespace tt::tt_metal
