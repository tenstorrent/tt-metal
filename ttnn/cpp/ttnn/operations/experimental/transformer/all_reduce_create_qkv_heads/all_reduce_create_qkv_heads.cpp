// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads.hpp"

#include "device/all_reduce_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::ccl::transformer {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteAllReduceCreateQkvHeads::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const ttnn::Tensor& batch_offset,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& multi_device_global_semaphore,
    // create qkv heads non-optional parameters
    uint32_t num_heads,
    const std::optional<ttnn::MemoryConfig>& all_reduce_memory_config,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
    // create qkv heads optional parameters
    std::optional<uint32_t> num_kv_heads,
    std::optional<uint32_t> slice_size,
    const std::optional<MemoryConfig>& final_memory_config,
    std::optional<DataType> dtype,
    bool use_noc1_only) {
    MemoryConfig out_memory_config = all_reduce_memory_config.value_or(input_tensor.memory_config());
    const uint32_t num_kv_heads_val = num_kv_heads.value_or(num_heads);

    TT_FATAL(
        input_tensor.padded_shape().size() == 4,
        "Input Tensor dim must be 4 while given dim = {}",
        input_tensor.padded_shape().size());
    TT_FATAL(
        input_tensor.padded_shape()[3] % (num_heads + 2 * num_kv_heads_val) == 0,
        "Input shape {} must be divisible by num_heads + 2*num_kv_heads = {}",
        input_tensor.padded_shape()[3],
        num_heads + 2 * num_kv_heads_val);

    uint32_t head_dim = input_tensor.padded_shape()[3] / (num_heads + 2 * num_kv_heads_val);

    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-reduce invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    uint32_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    auto output_tensors = ttnn::prim::all_reduce_create_qkv_heads(
        input_tensor,
        buffer_tensor,
        batch_offset,
        num_preferred_links.has_value() ? num_preferred_links.value() : 1,
        num_devices,
        out_memory_config,
        topology,
        multi_device_global_semaphore,
        worker_subdevice_id_opt,
        head_dim,
        use_noc1_only,
        num_heads,
        num_kv_heads.value_or(1),
        false,  // input_on_subcoregrids
        slice_size,
        final_memory_config.value_or(input_tensor.memory_config()),
        dtype.value_or(input_tensor.dtype()),
        cluster_axis);

    return {output_tensors.all_reduce, output_tensors.q, output_tensors.k, output_tensors.v};
}

}  // namespace ttnn::operations::experimental::ccl::transformer
