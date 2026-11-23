// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// The structural properties the KV cache kernels assume of a 1-element metadata tensor (slot, chunk start, chunk end)
// they read on device: on the cache's device, UINT32, ROW_MAJOR, a single element, not sharded, and in the memory
// config of the one TensorAccessor every metadata read goes through. Checked host-side so a malformed tensor fails
// with a clear message instead of a read of the wrong address; the values stay off the host dispatch path.
inline void validate_kv_cache_metadata_tensor(
    const Tensor& meta,
    const Tensor& cache,
    const tt::tt_metal::MemoryConfig& accessor_memory_config,
    const char* name) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using ::ttnn::StorageType;
    TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
    TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
    TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
    TT_FATAL(
        meta.logical_volume() == 1,
        "metadata tensor {} must be a single element (got {})",
        name,
        meta.logical_volume());
    TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);
    // The kernels resolve meta.mesh_buffer().address() against cache.device(); a tensor on a different mesh device
    // would bake the wrong address and fail obscurely downstream.
    TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
    TT_FATAL(
        meta.memory_config() == accessor_memory_config,
        "metadata tensors with different memory configs are not supported: one TensorAccessor serves every metadata "
        "read, so {} must match slot_idx (got buffer types {} and {})",
        name,
        meta.memory_config().buffer_type(),
        accessor_memory_config.buffer_type());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
