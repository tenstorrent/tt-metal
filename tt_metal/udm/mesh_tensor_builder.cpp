// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/mesh_tensor_builder.hpp"
#include "tt_metal/udm/mesh_builder.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/api/tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include <array>

namespace tt::tt_metal::experimental::udm {

// Align mesh and grid ranks by prepending 1s to the shorter shape
class MeshTensorBuilder::Impl {
public:
    explicit Impl(
        const tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
        const tt::tt_metal::Shape& tensor_shape_in_pages,
        const tt::tt_metal::distributed::MeshShape& distribution_shape,
        const std::vector<std::optional<int>>& shard_dims) :
        mesh_buffer_(mesh_buffer), distribution_shape_(distribution_shape), shard_dims_(shard_dims) {
        // tensor_shape_in_pages is already computed externally using TensorLayout
        // Extract and reconstruct mesh tensor shape from local tensor shape (already in pages)
        extract_tensor_shapes(tensor_shape_in_pages);

        accessor_args_ = tt::tt_metal::TensorAccessorArgs(mesh_buffer_);

        // Create mesh builder to get fabric node id mapping
        mesh_builder_ = std::make_unique<MeshBuilder>(mesh_buffer_);
    }

    std::vector<uint32_t> get_compile_time_args() const {
        auto compile_time_args = accessor_args_.get_compile_time_args();

        // MeshTensorAccessor args layout:
        // 1. TensorAccessorArgs (variable size) - already in compile_time_args
        // 2. buffer_address (uint32_t)
        // 3. aligned_page_size
        // 4. mesh_dspec_rank
        // 5. mesh_tensor_shape_in_pages[mesh_dspec_rank]
        // 6. mesh_tensor_strides_in_pages[mesh_dspec_rank]
        // 7. tensor_shape_in_pages[mesh_dspec_rank]
        // 8. tensor_strides_in_pages[mesh_dspec_rank]
        // 9. mesh_shape[mesh_dspec_rank]
        // 10. mesh_strides[mesh_dspec_rank]
        // 11. num_grids
        // 12. fabric_mesh_ids[num_grids]
        // 13. fabric_chip_ids[num_grids]

        compile_time_args.push_back(static_cast<uint32_t>(mesh_buffer_.address()));
        compile_time_args.push_back(mesh_buffer_.get_reference_buffer()->aligned_page_size());

        compile_time_args.push_back(mesh_tensor_rank_);
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_tensor_shape_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_tensor_strides_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(tensor_shape_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(tensor_strides_in_pages_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_shape_[i]);
        }
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            compile_time_args.push_back(mesh_strides_[i]);
        }

        // Get fabric node mapping from mesh builder
        auto fabric_node_args = mesh_builder_->get_fabric_nodes_compile_args();

        // Fabric node args layout:
        // 1. num_grids
        // 2. fabric_mesh_ids[num_grids]
        // 3. fabric_chip_ids[num_grids]

        // Append all fabric node args directly
        compile_time_args.insert(compile_time_args.end(), fabric_node_args.begin(), fabric_node_args.end());

        return compile_time_args;
    }

    uint64_t get_buffer_address() const { return mesh_buffer_.address(); }

    uint32_t get_aligned_page_size() const { return mesh_buffer_.get_reference_buffer()->aligned_page_size(); }

    const tt::tt_metal::Shape& get_mesh_tensor_shape_in_pages() const {
        std::vector<uint32_t> shape_vec;
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            shape_vec.push_back(mesh_tensor_shape_in_pages_[i]);
        }
        static thread_local tt::tt_metal::Shape cached_shape;
        cached_shape = tt::tt_metal::Shape(std::move(shape_vec));
        return cached_shape;
    }

    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer() const { return mesh_buffer_; }

    MeshBuilder& mesh_builder() { return *mesh_builder_; }

private:
    // Helper: Expand sharded tensor dimensions by multiplying by mesh extent
    // (doesn't change rank, so tensor dimension indices remain stable)
    void expand_sharded_tensor_dims(std::array<uint32_t, MAX_RANK>& mesh_tensor_shape) const {
        for (int mesh_dim = static_cast<int>(shard_dims_.size()) - 1; mesh_dim >= 0; --mesh_dim) {
            if (shard_dims_[mesh_dim].has_value()) {
                int tensor_dim = shard_dims_[mesh_dim].value();
                mesh_tensor_shape[tensor_dim] *= distribution_shape_[mesh_dim];
            }
        }
    }

    // Helper: Expand replicated tensor dimensions by multiplying existing tensor dims
    // Rule:
    // - Replicate on last mesh dim → multiply last tensor dim
    // - Replicate on non-last mesh dim → multiply first tensor dim
    void expand_replicated_tensor_dims(std::array<uint32_t, MAX_RANK>& mesh_tensor_shape, uint32_t& rank) const {
        uint32_t num_mesh_dims = shard_dims_.size();
        for (size_t mesh_dim = 0; mesh_dim < num_mesh_dims; ++mesh_dim) {
            if (!shard_dims_[mesh_dim].has_value()) {
                // This is a replicated mesh dim - determine which tensor dim to multiply
                int tensor_dim;
                if (mesh_dim == num_mesh_dims - 1) {
                    // Last mesh dim → multiply last tensor dim
                    tensor_dim = rank - 1;
                } else {
                    // Non-last mesh dim → multiply first tensor dim
                    tensor_dim = 0;
                }
                log_info(tt::LogTest, "mesh_dim {} distribution_shape_ {}", mesh_dim, distribution_shape_[mesh_dim]);
                mesh_tensor_shape[tensor_dim] *= distribution_shape_[mesh_dim];
            }
        }
    }

    std::pair<std::array<uint32_t, MAX_RANK>, uint32_t> reconstruct_mesh_tensor_shape(
        const Shape& local_tensor_shape) const {
        uint32_t rank = local_tensor_shape.rank();
        TT_FATAL(rank <= MAX_RANK, "Tensor rank exceeds MAX_RANK");

        std::array<uint32_t, MAX_RANK> mesh_tensor_shape{};
        for (size_t i = 0; i < rank; ++i) {
            mesh_tensor_shape[i] = local_tensor_shape[i];
        }

        // Apply distribution in two passes to avoid index shifting issues:
        // 1. First expand sharded dims (tensor dimension indices are stable)
        // 2. Then expand replicated dims (inserts new dimensions)
        expand_sharded_tensor_dims(mesh_tensor_shape);
        expand_replicated_tensor_dims(mesh_tensor_shape, rank);

        return {mesh_tensor_shape, rank};
    }

    void extract_tensor_shapes(const tt::tt_metal::Shape& tensor_shape_in_pages) {
        // Extract local tensor shape dimensions
        uint32_t local_rank = tensor_shape_in_pages.rank();
        TT_FATAL(local_rank <= MAX_RANK, "Tensor rank exceeds MAX_RANK");
        for (size_t i = 0; i < local_rank; ++i) {
            tensor_shape_in_pages_[i] = tensor_shape_in_pages[i];
        }

        // Apply distribution sharding and replication to get global mesh tensor shape
        auto [reconstructed_shape, reconstructed_rank] = reconstruct_mesh_tensor_shape(tensor_shape_in_pages);
        mesh_tensor_shape_in_pages_ = reconstructed_shape;
        mesh_tensor_rank_ = reconstructed_rank;

        // Adjust tensor_shape_in_pages to match mesh_tensor_rank
        adjust_shape_ranks(tensor_shape_in_pages_, local_rank, mesh_tensor_shape_in_pages_, mesh_tensor_rank_, 1);

        // Compute strides
        compute_strides(mesh_tensor_shape_in_pages_, mesh_tensor_rank_, mesh_tensor_strides_in_pages_);
        compute_strides(tensor_shape_in_pages_, mesh_tensor_rank_, tensor_strides_in_pages_);

        // Compute mesh shape (mesh_tensor_shape / tensor_shape) and mesh strides
        for (uint32_t i = 0; i < mesh_tensor_rank_; ++i) {
            mesh_shape_[i] = mesh_tensor_shape_in_pages_[i] / tensor_shape_in_pages_[i];
        }
        compute_strides(mesh_shape_, mesh_tensor_rank_, mesh_strides_);
    }

    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer_;
    tt::tt_metal::TensorAccessorArgs accessor_args_;

    tt::tt_metal::distributed::MeshShape distribution_shape_;
    std::vector<std::optional<int>> shard_dims_;  // nullopt = replicate, value = shard on that tensor dim

    uint32_t mesh_tensor_rank_ = 0;
    std::array<uint32_t, MAX_RANK> mesh_tensor_shape_in_pages_{};
    std::array<uint32_t, MAX_RANK> mesh_tensor_strides_in_pages_{};
    std::array<uint32_t, MAX_RANK> tensor_shape_in_pages_{};
    std::array<uint32_t, MAX_RANK> tensor_strides_in_pages_{};
    std::array<uint32_t, MAX_RANK> mesh_shape_{};    // mesh device shape (mesh_tensor_shape / tensor_shape)
    std::array<uint32_t, MAX_RANK> mesh_strides_{};  // strides for mesh device space

    // Mesh builder for fabric node id mapping
    std::unique_ptr<MeshBuilder> mesh_builder_;
};

MeshTensorBuilder::MeshTensorBuilder(
    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
    const tt::tt_metal::Shape& tensor_shape_in_pages,
    const tt::tt_metal::distributed::MeshShape& distribution_shape,
    const std::vector<std::optional<int>>& shard_dims) :
    impl_(std::make_unique<Impl>(mesh_buffer, tensor_shape_in_pages, distribution_shape, shard_dims)) {}

MeshTensorBuilder::~MeshTensorBuilder() = default;

MeshTensorBuilder::MeshTensorBuilder(MeshTensorBuilder&&) noexcept = default;
MeshTensorBuilder& MeshTensorBuilder::operator=(MeshTensorBuilder&&) noexcept = default;

std::vector<uint32_t> MeshTensorBuilder::get_compile_time_args() const { return impl_->get_compile_time_args(); }

uint64_t MeshTensorBuilder::get_buffer_address() const { return impl_->get_buffer_address(); }

uint32_t MeshTensorBuilder::get_aligned_page_size() const { return impl_->get_aligned_page_size(); }

const tt::tt_metal::Shape& MeshTensorBuilder::get_mesh_tensor_shape_in_pages() const {
    return impl_->get_mesh_tensor_shape_in_pages();
}

const tt::tt_metal::distributed::MeshBuffer& MeshTensorBuilder::mesh_buffer() const { return impl_->mesh_buffer(); }

MeshBuilder& MeshTensorBuilder::mesh_builder() { return impl_->mesh_builder(); }

}  // namespace tt::tt_metal::experimental::udm
