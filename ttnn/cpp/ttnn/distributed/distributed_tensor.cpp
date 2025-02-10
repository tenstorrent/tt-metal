// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_tensor.hpp"
#include "tt-metalium/assert.hpp"

namespace ttnn::distributed {

std::vector<Tensor> TensorToMesh::map(const Tensor& tensor) const {
    // This function should never be called directly, it's just to satisfy the linker
    TT_THROW("Pure virtual function 'map' called - please use or define concrete implementations instead.");
}

tt::tt_metal::DistributedTensorConfig TensorToMesh::config() const {
    // This function should never be called directly, it's just to satisfy the linker
    TT_THROW("Pure virtual function 'config' called - please use or define concrete implementations instead.");
}

Tensor MeshToTensor::compose(const std::vector<Tensor>& tensors) const {
    // This function should never be called directly, it's just to satisfy the linker
    TT_THROW("Pure virtual function 'compose' called  - please use or define concrete implementations instead.");
}

std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device) {
    return std::make_unique<ReplicateTensorToMesh>(mesh_device.num_devices());
}

std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim) {
    return std::make_unique<ShardTensorToMesh>(mesh_device.num_devices(), dim);
}

std::unique_ptr<TensorToMesh> shard_tensor_to_2d_mesh_mapper(
    MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config) {
    TT_FATAL(
        config.row_dim.has_value() || config.col_dim.has_value(),
        "Sharding a tensor to 2D mesh requires at least one dimension to shard");
    TT_FATAL(
        mesh_shape.num_rows <= mesh_device.shape().num_rows &&  //
            mesh_shape.num_cols <= mesh_device.shape().num_cols,
        "Device mesh shape does not match the provided mesh shape.");
    return std::make_unique<ShardTensorTo2dMesh>(mesh_shape, config);
}

std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(int dim) {
    return std::make_unique<ConcatMeshToTensor>(dim);
}

std::unique_ptr<MeshToTensor> concat_2d_mesh_to_tensor_composer(MeshDevice& mesh_device, const Concat2dConfig& config) {
    TT_FATAL(
        config.row_dim != config.col_dim,
        "Dimensions in 'dims' must be different; got row_dim: {}, col_dim: {}",
        config.row_dim,
        config.col_dim);
    return std::make_unique<Concat2dMeshToTensor>(mesh_device, config);
}

Tensor distribute_tensor(
    const Tensor& tensor, const TensorToMesh& mapper, std::optional<std::reference_wrapper<MeshDevice>> mesh_device) {
    TT_FATAL(
        tensor.storage_type() != tt::tt_metal::StorageType::MULTI_DEVICE &&
            tensor.storage_type() != tt::tt_metal::StorageType::MULTI_DEVICE_HOST,
        "TensorToMesh does not support multi-device or multi-device host tensors; got storage type: {}",
        tensor.storage_type());
    std::vector<Tensor> tensors = mapper.map(tensor);
    Tensor output = aggregate_as_tensor(tensors, mapper.config());
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()));
    }
    return output;
}

Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer) {
    return is_multi_device_tensor(tensor) ? composer.compose(get_tensors_from_multi_device_storage(tensor))
                                          : composer.compose({tensor});
}

}  // namespace ttnn::distributed
