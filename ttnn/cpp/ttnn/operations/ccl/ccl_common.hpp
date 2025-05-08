// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/program.hpp>
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "erisc_datamover_builder_helper.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

namespace ttnn {
namespace ccl {

struct SyncModeSpec {
    uint32_t num_signals = 0;
    CoreCoord core;
    std::vector<uint32_t> sem_ids;
    std::vector<uint32_t> wait_counts;

    void add_signal(uint32_t sem_id, uint32_t wait_count);
};

class EriscDatamoverBuilder;

// Creates a mesh workload by calling the `create_program` function for each coordinate in the `tensor_coords` set.
tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload_from_programs(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const std::function<tt::tt_metal::operation::ProgramWithCallbacks(const ttnn::MeshCoordinate&)>& create_program);

struct SenderRecieverConfig {
    uint32_t device_index = 0;
    std::optional<chip_id_t> sender_device_id;
    std::optional<chip_id_t> receiver_device_id;
};

SenderRecieverConfig get_device_sender_receiver_config(
    const IDevice* target_device, const std::vector<IDevice*>& devices, ttnn::ccl::Topology topology);

SenderRecieverConfig get_device_sender_receiver_config_in_ring(
    const MeshCoordinate& mesh_coord, const distributed::MeshDevice* mesh_device, uint32_t cluster_axis, int ring_size);

std::vector<ttnn::Tensor> unpad_output_tensor(
    const std::vector<ttnn::Tensor>& output_tensor,
    const uint32_t num_devices,
    const ttnn::SmallVector<uint32_t>& unpad_elements,
    const int dim);

class LineTopology {
   public:
    LineTopology(
        size_t line_size,
        size_t line_index);

    bool is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction direction) const;
    bool is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction direction) const;

    bool is_at_end_of_line() const;

    size_t line_size() const;

    size_t line_index() const;

    size_t get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction direction) const;

    ttnn::ccl::Topology topology() const;

   private:
    size_t _line_size;
    size_t _line_index;
};


// Eventual home: ccl_topology_descriptors
struct RingTopology {
    RingTopology(
        tt::tt_metal::IDevice const* device,
        Topology topology,
        std::optional<uint32_t> sender_device_id,
        std::optional<uint32_t> receiver_device_id,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index);

    bool is_first_device_in_line(bool in_clockwise_direction) const;
    bool is_last_device_in_line(bool in_clockwise_direction) const;

    const tt::tt_metal::IDevice*device;

    std::vector<CoreCoord> eth_sender_cores;
    std::vector<CoreCoord> eth_receiver_cores;

    uint32_t num_links;
    uint32_t ring_size;
    uint32_t ring_index;
    bool is_linear;
};

struct TensorPartition {
    TensorPartition(
        uint32_t partition_size,
        uint32_t partition_index)
        : partition_size(partition_size),
          partition_index(partition_index) {}

    uint32_t partition_size;
    uint32_t partition_index;
};

class CclOpTensorConfig {
   public:
    static std::unique_ptr<CclOpTensorConfig> build_all_gather_tensor_config(Tensor const& tensor);

    CclOpTensorConfig(Tensor const& tensor);
    uint32_t get_page_size() const;
    uint32_t get_tile_size() const;
    tt::tt_metal::Tile get_tile() const;

    uint32_t get_buffer_start_address() const;

    virtual ~CclOpTensorConfig()=default;

   protected:
    uint32_t page_size;
    uint32_t tile_size;
    tt::tt_metal::Tile tile;
    uint32_t buffer_start_address;
    tt::DataFormat df;
};

class CclOpInterleavedTensorConfig final : public virtual CclOpTensorConfig {
   public:
    CclOpInterleavedTensorConfig(Tensor const& input_tensor);
};

class CclOpShardedTensorConfig final : public virtual CclOpTensorConfig {
   public:
    CclOpShardedTensorConfig(Tensor const& tensor);

    tt::tt_metal::ShardSpec const& get_shard_spec() const;

   private:
    uint32_t page_size;
    tt::tt_metal::ShardSpec const shard_spec;
};

struct CclTensorSlicer {
    CclTensorSlicer(
        const tt::tt_metal::Shape& tensor_shape,
        const tt::tt_metal::Shape& dim_slice_factors,
        // tt::tt_metal::Shape page_shape,
        std::size_t num_pages,
        std::size_t elem_size,
        std::size_t page_size_in_bytes) :
        tensor_shape(tensor_shape),
        dim_slice_factors_per_rank(dim_slice_factors),
        // page_shape(page_shape),
        num_pages(num_pages),
        page_size_in_bytes(page_size_in_bytes),
        elem_size(elem_size) {
        TT_ASSERT(
            tensor_shape.rank() == dim_slice_factors.rank(),
            "Tensor shape and dim slice factors must have the same size");
        TT_ASSERT(
            std::all_of(dim_slice_factors.cbegin(), dim_slice_factors.cend(), [](uint32_t factor) { return factor > 0; }),
            "All factors must be greater than 0");
    }

    std::size_t get_num_pages_per_slice() const {
        std::size_t n = std::accumulate(
            dim_slice_factors_per_rank.cbegin(), dim_slice_factors_per_rank.cend(), 1, std::multiplies<uint32_t>());
        for (uint32_t i = 0; i < (tensor_shape.rank() - dim_slice_factors_per_rank.rank()); ++i) {
            n *= tensor_shape[i];
        }
        return n;
    }

    tt::tt_metal::Shape const tensor_shape;
    tt::tt_metal::Shape const dim_slice_factors_per_rank;
    // tt::tt_metal::Shape const page_shape;
    std::size_t const num_pages;

    // tt::tt_metal::Shape rank_slice_shape;

    std::size_t const page_size_in_bytes;
    std::size_t const elem_size;
};

// To be replaced by the CclTensorSlicer class, which should be reusable between sharded and interleaved
// specs and also provides a simpler interface to reason about
struct LegacyCclTensorSlicer {
    LegacyCclTensorSlicer() :
        input_page_size(0),
        num_rows(0),
        num_cols(0),
        row_offset(0),
        col_offset(0),
        num_tiles(0),
        input_start_page_idx(0),
        output_addr_offset(0),
        col_idx(0),
        row_idx(0),
        output_page_offset(0),
        output_start_page_idx(0),
        output_start_addr_offset(0),
        row_major(false),
        slice_dim_is_width(false),
        is_sharded(false) {}

    LegacyCclTensorSlicer(
        uint32_t input_page_size,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t row_offset,
        uint32_t col_offset,
        uint32_t num_tiles,
        uint32_t input_start_page_idx,
        uint32_t output_addr_offset,
        uint32_t col_idx,
        uint32_t row_idx,
        uint32_t output_page_offset,
        uint32_t output_start_page_idx,
        uint32_t output_start_addr_offset,
        bool row_major,
        bool slice_dim_is_width,
        bool is_sharded) :
        input_page_size(input_page_size),
        num_rows(num_rows),
        num_cols(num_cols),
        row_offset(row_offset),
        col_offset(col_offset),
        num_tiles(num_tiles),
        input_start_page_idx(input_start_page_idx),
        output_addr_offset(output_addr_offset),
        col_idx(col_idx),
        row_idx(row_idx),
        output_page_offset(output_page_offset),
        output_start_page_idx(output_start_page_idx),
        output_start_addr_offset(output_start_addr_offset),
        row_major(row_major),
        slice_dim_is_width(slice_dim_is_width),
        is_sharded(is_sharded) {}

    virtual void increment(uint32_t num_pages) = 0;

    uint32_t input_page_size;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t row_offset;
    uint32_t col_offset;
    uint32_t num_tiles;
    uint32_t input_start_page_idx;
    uint32_t output_addr_offset;
    uint32_t col_idx;
    uint32_t row_idx;
    uint32_t output_page_offset;
    uint32_t output_start_page_idx;
    uint32_t output_start_addr_offset;
    bool row_major;
    bool slice_dim_is_width;
    bool is_sharded;
};



inline namespace v1 {
struct TensorSlice {
    using ords_t = tt_xy_pair;
    ords_t tensor_shape;
    ords_t tensor_slice_shape;
    ords_t tensor_slice_offset;
    ords_t worker_slice_shape;
    ords_t worker_slice_offset;
    std::size_t dim;
};
};

// Workers iterate over tensor slices in a sequence along a
// single, specified dimension. Workers iterator over the tensor
// slice in wrapped mode
std::vector<TensorSlice> generate_slice_sequence_on_dim(
    TensorSlice::ords_t tensor_shape,
    TensorSlice::ords_t worker_slice_shape,
    std::size_t fracture_dim,
    std::size_t num_slices,
    std::int64_t start_slice_index,
    std::int64_t end_slice_index,
    std::size_t worker_index
);

// Uniform Tensor Worker Slice
struct InterleavedTensorWorkerSlice {
    InterleavedTensorWorkerSlice(
        tt_xy_pair const& tensor_shape,
        tt_xy_pair const& tensor_slice_shape,
        tt_xy_pair const& worker_slice_shape,
        tt_xy_pair const& worker_slice_offset,
        bool worker_slice_is_wrapped=false) :
        tensor_shape(tensor_shape),
        tensor_slice_shape(tensor_slice_shape),
        worker_slice_shape(worker_slice_shape),
        worker_slice_offset(worker_slice_offset),
        worker_slice_is_wrapped(worker_slice_is_wrapped) {}

    // Could probably be solved in some closed form

    std::size_t compute_num_worker_slice_iterations(std::size_t num_workers) const {
        auto slice_offset = coord_t(worker_slice_offset.x, worker_slice_offset.y);
        auto const& slice_shape = coord_t(worker_slice_shape.x, worker_slice_shape.y);
        auto const& outer_slice_shape = coord_t(tensor_slice_shape.x, tensor_slice_shape.y);
        uint32_t num_iterations = 0;
        while (slice_offset.y < tensor_slice_shape.y && slice_offset.x < tensor_slice_shape.x) {

            slice_offset = worker_slice_is_wrapped ? ccl::advance_wrapped_slice_row_major(slice_offset, slice_shape, outer_slice_shape, num_workers) :
                ccl::advance_slice_row_major(slice_offset, slice_shape, outer_slice_shape, num_workers);
            num_iterations++;
        }

        return num_iterations;
    }

    std::size_t get_worker_slice_num_pages() const {
        return worker_slice_shape.x * worker_slice_shape.y;
    }

    void print() const {
        log_trace(tt::LogOp, "----- printing worker slice -----");
        log_trace(tt::LogOp, "tensor_shape: ({},{})", tensor_shape.x, tensor_shape.y);
        log_trace(tt::LogOp, "tensor_slice_shape: ({},{})", tensor_slice_shape.x, tensor_slice_shape.y);
        log_trace(tt::LogOp, "worker_slice_shape: ({},{})", worker_slice_shape.x, worker_slice_shape.y);
        log_trace(tt::LogOp, "worker_slice_offset: ({},{})", worker_slice_offset.x, worker_slice_offset.y);
        log_trace(tt::LogOp, "worker_slice_is_wrapped: {}", worker_slice_is_wrapped);
        log_trace(tt::LogOp, "worker_slice_num_pages: {}", get_worker_slice_num_pages());
    }

    tt_xy_pair tensor_shape;
    tt_xy_pair tensor_slice_shape;
    tt_xy_pair worker_slice_shape;
    tt_xy_pair worker_slice_offset;

    bool worker_slice_is_wrapped;
};

template <class DERIVED_SLICER_T>
class RingReduceScatterBaseTensorSlicer : public LegacyCclTensorSlicer {
    public:
    RingReduceScatterBaseTensorSlicer(
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index, bool wrapped) {
        TT_ASSERT(global_worker_index < this->worker_slice_shapes.size(), "Invalid worker index {} in `worker_slice_shapes` of size {}", global_worker_index, worker_slice_shapes.size());
        TT_ASSERT(global_worker_index < this->worker_slice_offsets.size(), "Invalid worker index {} in `worker_slice_offsets` of size {}", global_worker_index, worker_slice_offsets.size());
        return ccl::InterleavedTensorWorkerSlice(
            this->flattened_tensor_shape,
            this->tensor_slice_shape,
            this->worker_slice_shapes[global_worker_index],
            this->worker_slice_offsets[global_worker_index],
            wrapped);
    }

    [[deprecated("deprecated code path for reduce scatter. Use nerw get_worker_slice API instead")]]
    virtual void increment(uint32_t num_pages) override {
        TT_THROW("deprecated code path for ");
    }

   public:
    std::vector<tt_xy_pair> get_worker_slice_shapes() const { return this->worker_slice_shapes; }
    uint32_t get_worker_slice_size_bytes(std::size_t worker_index) {
        TT_ASSERT(this->worker_slice_shapes.size() > worker_index, "Invalid worker index {} in `worker_slice_shapes` of size {}", worker_index, worker_slice_shapes.size());
        auto worker_slice_shape = this->worker_slice_shapes.at(worker_index);
        return worker_slice_shape.x * worker_slice_shape.y * this->input_page_size;
    }

    void create_worker_slice_shape_for_row_major_layout(tt_xy_pair const& tensor_slice_shape, uint32_t num_workers) {
        TT_THROW("Row major interleaved not supported by Reduce Scatter");
    }

    // Static methods
    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        ttnn::Shape const& tensor_shape,
        tt_xy_pair const& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_row_major_layout(
        tt_xy_pair const& tensor_slice_shape_in_elems, uint32_t num_workers, uint32_t max_slice_size_in_elements);


    protected:
    tt_xy_pair flattened_tensor_shape;
    tt_xy_pair tensor_slice_shape;
    std::vector<tt_xy_pair> worker_slice_shapes;
    // For RowMajor - offset is in elements
    // For Tile - offset is in tiles
    std::vector<tt_xy_pair> worker_slice_offsets;

};

class RingReduceScatterTensorSlicer : public RingReduceScatterBaseTensorSlicer<RingReduceScatterTensorSlicer> {
   public:
    RingReduceScatterTensorSlicer(
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index) {
        return this->RingReduceScatterBaseTensorSlicer::get_worker_slice(global_worker_index, false);} // False: Use the non wrapped version of the worker slice

    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        ttnn::Shape const& tensor_shape,
        tt_xy_pair const& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

};

// Define a class RingReduceScatterWrappedTensor slicer that inherits from RingReduceScatterBaseTensorSlicer and overwrites the compute_worker_slice_offsets and create_worker_slice_shapes_for_tile_layout functions
class RingReduceScatterWrappedTensorSlicer : public RingReduceScatterBaseTensorSlicer<RingReduceScatterWrappedTensorSlicer> {
   public:
    RingReduceScatterWrappedTensorSlicer(
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index) {
        return this->RingReduceScatterBaseTensorSlicer::get_worker_slice(global_worker_index, true);} // True: Use the wrapped version of the worker slice

    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        ttnn::Shape const& tensor_shape,
        tt_xy_pair const& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

};

class InterleavedRingAllGatherTensorSlicer : public LegacyCclTensorSlicer {
   public:
    InterleavedRingAllGatherTensorSlicer(
         const Tensor & input_tensor,  const Tensor & output_tensor, int slice_dim, uint32_t slice_idx) :
        LegacyCclTensorSlicer() {
        this->row_major = input_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR;
        this->slice_dim_is_width = input_tensor.get_padded_shape().rank() - 1 == slice_dim;
        this->is_sharded = input_tensor.is_sharded();

        this->input_page_size = input_tensor.buffer()->page_size();

        if (row_major) {
            this->num_cols = input_tensor.get_padded_shape()[-1];
            auto input_shape = input_tensor.get_padded_shape();
            auto output_shape = output_tensor.get_padded_shape();
            this->num_rows =
                std::accumulate(input_shape.cbegin() + slice_dim, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
            this->row_offset =
                std::accumulate(
                    output_shape.cbegin() + slice_dim, output_shape.cend() - 1, 1, std::multiplies<uint32_t>()) -
                num_rows;
        } else {
            auto input_shape = input_tensor.get_padded_shape();
            auto output_shape = output_tensor.get_padded_shape();
            auto input_tile = input_tensor.tensor_spec().tile();
            auto output_tile = output_tensor.tensor_spec().tile();
            this->num_cols = input_shape[-1] / input_tile.get_width();
            uint32_t num_output_cols = output_tensor.get_padded_shape()[-1] / output_tile.get_width();
            this->num_rows =
                std::accumulate(
                    input_shape.cbegin() + slice_dim, input_shape.cend() - 1, 1, std::multiplies<uint32_t>()) /
                input_tile.get_height();
            this->row_offset =
                (std::accumulate(
                     output_shape.cbegin() + slice_dim, output_shape.cend() - 1, 1, std::multiplies<uint32_t>()) / output_tile.get_height() - num_rows) *
                num_output_cols;
            this->col_offset = num_output_cols - num_cols;
            this->num_tiles = num_rows * num_cols;
        }

        if (row_major) {
            if (slice_dim_is_width) {
                this->output_addr_offset = input_page_size;
            } else {
                this->output_page_offset = num_rows;
            }
        } else {
            if (slice_dim_is_width) {
                this->output_page_offset = num_cols;
            } else {
                this->output_page_offset = num_tiles;
            }
        }
        this->output_start_page_idx = slice_idx /*ring_index*/ * output_page_offset;
        this->output_start_addr_offset = slice_idx /*ring_index*/ * output_addr_offset;
    }

    virtual void increment(uint32_t num_pages) override {
        if (num_pages /*pages_per_worker*/ > 0) {
            if (row_major) {
                uint32_t num_rows_shifted = row_idx + num_pages /*pages_per_worker*/;
                uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                this->output_start_page_idx += num_pages /*pages_per_worker*/ + num_blocks_shifted * row_offset;
                this->row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
            } else {
                uint32_t num_cols_shifted = col_idx + num_pages /*pages_per_worker*/;
                uint32_t num_rows_shifted = num_cols_shifted / num_cols;
                uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                this->output_start_page_idx += num_pages /*pages_per_worker*/ + num_rows_shifted * col_offset +
                                                num_blocks_shifted * row_offset;
                this->col_idx = num_cols_shifted % num_cols;
                this->row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
            }
        }
        this->input_start_page_idx += num_pages /*pages_per_worker*/;
    }
};


tt::tt_metal::KernelHandle generate_edm_kernel(
   tt::tt_metal::Program& program,
    tt::tt_metal::IDevice const* device,
    tt::tt_fabric::FabricEriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    tt::tt_metal::NOC noc_id);

tt::tt_metal::KernelHandle generate_edm_kernel(
   tt::tt_metal::Program& program,
    IDevice const* device,
    EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    tt::tt_metal:: NOC noc_id);

void generate_edm_kernels_for_ring_or_linear_topology(
   tt::tt_metal::Program& program,
    IDevice const* device,
    RingTopology const& topology_config,
    std::vector<ccl::EriscDatamoverBuilder> const& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id);

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    std::size_t num_buffers_per_channel,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
    EriscDataMoverTerminationMode termination_mode);


std::vector<TensorSlice> generate_slice_sequence_on_dim_v2(
    TensorSlice::ords_t tensor_shape,
    TensorSlice::ords_t worker_slice_shape,
    TensorSlice::ords_t worker_slice_offset,
    std::size_t fracture_dim,
    std::size_t num_slices,
    std::int64_t start_slice_index,
    std::int64_t end_slice_index_exclusive,
    std::size_t worker_index
);

class GenericWrappedTensorSlicer {
public:
    GenericWrappedTensorSlicer(
        const Tensor& input_tensor,
        const Tensor& output_tensor,
        int slice_dim,
        uint32_t partition_index,
        uint32_t partition_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index);

    ttnn::ccl::v2::TensorSlice get_worker_slice_v2(std::size_t global_worker_index);

    // method to compute offsets in a wrapped layout
    std::vector<tt_xy_pair> compute_worker_slice_offsets(
        const std::vector<tt_xy_pair>& worker_slice_shapes,
        tt_xy_pair const& tensor_slice_shape);

    // method to create worker slice shapes in a tile layout
    std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        const ttnn::Shape& tensor_shape,
        tt_xy_pair const& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

private:
    void initialize(
        const Tensor& input_tensor,
        const Tensor& output_tensor,
        int slice_dim,
        uint32_t partition_index,
        uint32_t partition_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    tt_xy_pair calculate_tensor_slice_shape(const Tensor& input_tensor, int slice_dim, uint32_t partition_size);
    Shape4D<uint32_t> calculate_tensor_slice_offset(const Tensor& input_tensor, int slice_dim, uint32_t partition_index);

    // Class member variables
    tt_xy_pair flattened_tensor_shape;
    tt_xy_pair tensor_slice_shape;
    Shape4D<uint32_t> tensor_slice_offset;
    std::vector<tt_xy_pair> worker_slice_shapes;
    std::vector<tt_xy_pair> worker_slice_offsets;
    uint32_t input_page_size;
    bool row_major;
    uint32_t partition_index;
    uint32_t partition_size;
};


class GenericWrappedTensorSlicerV2 {
public:
    GenericWrappedTensorSlicerV2(
        const Tensor& input_tensor,
        int slice_dim,
        uint32_t partition_index,
        uint32_t partition_size,
        uint32_t total_num_workers);

    ttnn::ccl::v2::TensorSlice get_worker_slice_v2(std::size_t global_worker_index);

    // method to compute offsets in a wrapped layout
    std::vector<Shape4D<uint32_t>> compute_worker_slice_offsets(std::vector<Shape4D<uint32_t>> const& worker_slice_shapes);

    // method to create worker slice shapes in a tile layout
    std::vector<Shape4D<uint32_t>> create_worker_slice_shapes_for_tile_layout(
        Shape4D<uint32_t> const& tensor_slice_shape_in_tiles,
        uint32_t num_workers);

private:
    void initialize(
        const Tensor& input_tensor,
        int slice_dim,
        uint32_t partition_index,
        uint32_t partition_size,
        uint32_t total_num_workers);

    Shape4D<uint32_t> calculate_tensor_slice_shape(Shape4D<uint32_t> const& tensor_shape, int slice_dim, uint32_t partition_size);
    Shape4D<uint32_t> calculate_tensor_slice_offset(Shape4D<uint32_t> const& tensor_shape, int slice_dim, uint32_t partition_index);

    // Class member variables
    Shape4D<uint32_t> tensor_shape;
    Shape4D<uint32_t> tensor_slice_shape;
    Shape4D<uint32_t> tensor_slice_offset;
    std::vector<Shape4D<uint32_t>> worker_slice_shapes;
    std::vector<Shape4D<uint32_t>> worker_slice_offsets;
    uint32_t input_page_size;
    bool row_major;
    uint32_t partition_index;
    uint32_t partition_size;
};

std::tuple<size_t, size_t, bool> get_forward_backward_configuration(size_t ring_size, size_t ring_index, Topology topology);

}  // namespace ccl
}  // namespace ttnn
