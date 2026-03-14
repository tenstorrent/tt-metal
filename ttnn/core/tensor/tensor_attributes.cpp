// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <variant>
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

bool GhostSpecAccessGuard::check_ghost_spec_ = true;
std::string_view GhostSpecAccessGuard::current_white_listed_function = "";

void GhostSpecAccessGuard::fault() {
    if (!check_ghost_spec_) {
        log_error(
            tt::LogAlways, "Ghost spec access detected in whitelisted function: {}", current_white_listed_function);
        return;
    }

    TT_THROW("Ghost spec access");
}

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)), tensor_topology_(std::move(tensor_topology)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const {
    if (const auto* device_storage = std::get_if<DeviceStorage>(&storage_); device_storage != nullptr) {
        if (!device_storage->is_allocated()) {
            GhostSpecAccessGuard::fault();
        }
    }
    return tensor_spec_;
}

const TensorTopology& TensorAttributes::get_tensor_topology() const { return tensor_topology_; }

TensorAttributes TensorAttributes::with_tensor_topology(TensorTopology tensor_topology) const {
    TensorAttributes result = *this;
    result.tensor_topology_ = std::move(tensor_topology);
    return result;
}

}  // namespace tt::tt_metal
