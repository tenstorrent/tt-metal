// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
struct RuntimeArgsData;
}

#include <vector>
#include <cstdint>
#include <cstddef>

namespace ttnn::ccl {
/*
 * Tracks tensors (by user defined index) and the runtime arg indices that are used to
 * reference their address.
 *
 * Future work will let a user track runtime args by various attributes, such as shape, dtype, etc,
 * -> whatever the user decides to track in runtime args that may change across invocations
 */
struct tensor_address_runtime_args_overrider {
public:
    using runtime_args_t = std::vector<uint32_t>;

    /*
     * Add a tensor to be tracked by this overrider.
     *
     * @return: the index assigned to this tensor, for use in future lookups by user
     */
    size_t add_tensor();

    /*
     * Add a runtime arg index to the tensor's runtime args
     *
     * @param tensor_idx: the index of the tensor to add the runtime arg index to, assigned by add_tensor()
     * @param runtime_arg_index: the index of the runtime arg to add
     */
    void add_runtime_arg_index(size_t tensor_idx, size_t runtime_arg_index);

    /*
     * Get the runtime arg indices that are associated with the specified tensor\
     *
     * @param tensor_idx: the index of the tensor to get the runtime args for, assigned by add_tensor()
     * @return: the runtime args that are associated with the tensor
     */
    std::vector<size_t> get_runtime_arg_indices(size_t tensor_idx) const;

    /*
     * Get the runtime arg indices that are used to reference the tensor's address
     *
     * @param tensor_idx: the index of the tensor to get the runtime args for, assigned by add_tensor()
     * @param new_value: the new value to set the tensor's associated runtime args to
     * @param runtime_args_to_modify: the runtime args to modify. These will be modified in place.
     */
    void override_runtime_args(
        size_t tensor_idx, uint32_t new_value, tt::tt_metal::RuntimeArgsData& runtime_args_to_modify) const;

    /*
     * Get the number of tensors in the overrider
     */
    size_t size() const;

private:
    std::vector<std::vector<size_t>> tensor_address_runtime_arg_indices;
};
}  // namespace ttnn::ccl
