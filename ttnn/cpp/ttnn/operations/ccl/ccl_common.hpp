// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <numeric>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::ccl {

bool is_fabric_2d();

std::optional<ttnn::DeviceComputeKernelConfig> resolve_fp32_acc_compute_kernel_config(
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType input_dtype);

// Warn about ideal packet size
void validate_packet_size(tt::ARCH arch, size_t packet_size, uint32_t page_size);

uint32_t get_topological_dimension(const Tensor& tensor, const std::optional<uint32_t>& cluster_axis);

tt::tt_fabric::Topology get_usable_topology(
    const Tensor& tensor,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& cluster_axis = std::nullopt);

// Is every hop along this mesh axis wired in same dir? Ex: a 1x8 view of a 2x4 board makes axis-1 turn corners.
bool is_axis_straight(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t axis);

// Is the link that would close this mesh axis into a ring wired? Always false for an axis of 2 or
// fewer devices, which closes on the link it already uses.
bool is_axis_wrap_wired(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t axis);

// Where to allocate a GlobalSemaphore whose value is CARRIED across invocations (a barrier or arrival
// counter, written once and thereafter only incremented). Prefers L1_SMALL, which keeps these long-lived
// buffers out of general L1, and falls back to general L1 on a device opened without an L1_SMALL region.
tt::tt_metal::BufferType prefer_l1_small_buffer_type(const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Companion to prefer_l1_small_buffer_type: when it fell back to L1, name the device-open parameter
// that would have kept the semaphores in L1_SMALL.
void warn_if_semaphores_fall_back_to_l1(tt::tt_metal::BufferType sem_buffer_type);

// Floor of the L1_SMALL region, the ceiling a fabric mux's map is sized against. Derived
// from immutable limits: worker L1 is fixed and the L1_SMALL bank is never shrunk, whereas the L1 bank
// size IS reduced by a sub-device manager's bottom-up reservation
// (sub_device_manager_tracker.cpp), which would drag a floor computed as base + L1 bank size below the
// real one and cost a V1 mux buffers -- or make a V2 mux reject a map that physically fits.
size_t l1_small_floor_address(const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Resolve the topology (Ring vs Linear) for a single mesh axis
tt::tt_fabric::Topology get_axis_topology(
    const Tensor& tensor, tt::tt_fabric::FabricConfig fabric_config, uint32_t axis);

tt::tt_fabric::Topology convert_2d_to_1d_topology(tt::tt_fabric::Topology topology);

uint32_t get_linearized_index_from_physical_coord(
    const Tensor& tensor, const MeshCoordinate& physical_coord, const std::optional<uint32_t>& cluster_axis);

std::optional<MeshCoordinate> get_physical_neighbor_from_physical_coord(
    const Tensor& tensor,
    const MeshCoordinate& physical_coord,
    int offset,
    ttnn::ccl::Topology topology,
    const std::optional<uint32_t>& cluster_axis);

enum class LineDirection : uint8_t {
    FORWARD,
    BACKWARD,
};

enum class CoreAllocationStrategy {
    ROW_MAJOR,
    COL_MAJOR,
};

struct WorkerCoreSelection {
    CoreRangeSet core_range_set;
    std::vector<CoreCoord> cores;
    // Selected cores that core_grid_offset shifted off the device's worker grid. Kernels cannot be placed on these.
    std::vector<CoreCoord> unplaceable_cores;

    bool all_placeable() const { return unplaceable_cores.empty(); }
};

// Selects worker cores without asserting on the result. Callers that can adapt to a selection which does not fit at
// core_grid_offset use this; everyone else should use choose_worker_cores(), which rejects such a selection.
WorkerCoreSelection try_choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    distributed::MeshDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    CoreAllocationStrategy strategy = CoreAllocationStrategy::ROW_MAJOR);

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    distributed::MeshDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    CoreAllocationStrategy strategy = CoreAllocationStrategy::ROW_MAJOR);

class EriscDatamoverBuilder;

class LineTopology {
public:
    LineTopology(size_t line_size, size_t line_index);

    bool is_first_device_in_line(ttnn::ccl::LineDirection direction) const;
    bool is_last_device_in_line(ttnn::ccl::LineDirection direction) const;

    bool is_at_end_of_line() const;

    size_t line_size() const;

    size_t line_index() const;

    size_t get_distance_to_end_of_line(ttnn::ccl::LineDirection direction) const;

    ttnn::ccl::Topology topology() const;

private:
    size_t _line_size;
    size_t _line_index;
};

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

    virtual ~LegacyCclTensorSlicer() = default;

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
    std::size_t dim{};
};
};  // namespace v1

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
    std::size_t worker_index);

// Uniform Tensor Worker Slice
struct InterleavedTensorWorkerSlice {
    InterleavedTensorWorkerSlice(
        const tt_xy_pair& tensor_shape,
        const tt_xy_pair& tensor_slice_shape,
        const tt_xy_pair& worker_slice_shape,
        const tt_xy_pair& worker_slice_offset,
        bool worker_slice_is_wrapped = false) :
        tensor_shape(tensor_shape),
        tensor_slice_shape(tensor_slice_shape),
        worker_slice_shape(worker_slice_shape),
        worker_slice_offset(worker_slice_offset),
        worker_slice_is_wrapped(worker_slice_is_wrapped) {}

    // Could probably be solved in some closed form

    std::size_t compute_num_worker_slice_iterations(std::size_t num_workers) const {
        auto slice_offset = coord_t(worker_slice_offset.x, worker_slice_offset.y);
        const auto& slice_shape = coord_t(worker_slice_shape.x, worker_slice_shape.y);
        const auto& outer_slice_shape = coord_t(tensor_slice_shape.x, tensor_slice_shape.y);
        uint32_t num_iterations = 0;
        while (slice_offset.y < tensor_slice_shape.y && slice_offset.x < tensor_slice_shape.x) {
            slice_offset =
                worker_slice_is_wrapped
                    ? ccl::advance_wrapped_slice_row_major(slice_offset, slice_shape, outer_slice_shape, num_workers)
                    : ccl::advance_slice_row_major(slice_offset, slice_shape, outer_slice_shape, num_workers);
            num_iterations++;
        }

        return num_iterations;
    }

    std::size_t get_worker_slice_num_pages() const { return worker_slice_shape.x * worker_slice_shape.y; }

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
private:
    friend DERIVED_SLICER_T;
    RingReduceScatterBaseTensorSlicer(
        const Tensor& input_tensor,
        const Tensor& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

public:
    ~RingReduceScatterBaseTensorSlicer() override = default;

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index, bool wrapped) {
        TT_ASSERT(
            global_worker_index < this->worker_slice_shapes.size(),
            "Invalid worker index {} in `worker_slice_shapes` of size {}",
            global_worker_index,
            worker_slice_shapes.size());
        TT_ASSERT(
            global_worker_index < this->worker_slice_offsets.size(),
            "Invalid worker index {} in `worker_slice_offsets` of size {}",
            global_worker_index,
            worker_slice_offsets.size());
        return ccl::InterleavedTensorWorkerSlice(
            this->flattened_tensor_shape,
            this->tensor_slice_shape,
            this->worker_slice_shapes[global_worker_index],
            this->worker_slice_offsets[global_worker_index],
            wrapped);
    }

    [[deprecated("deprecated code path for reduce scatter. Use nerw get_worker_slice API instead")]] void increment(
        uint32_t /*num_pages*/) override {
        TT_THROW("deprecated code path for ");
    }

    std::vector<tt_xy_pair> get_worker_slice_shapes() const { return this->worker_slice_shapes; }
    uint32_t get_worker_slice_size_bytes(std::size_t worker_index) {
        TT_ASSERT(
            this->worker_slice_shapes.size() > worker_index,
            "Invalid worker index {} in `worker_slice_shapes` of size {}",
            worker_index,
            worker_slice_shapes.size());
        auto worker_slice_shape = this->worker_slice_shapes.at(worker_index);
        return worker_slice_shape.x * worker_slice_shape.y * this->input_page_size;
    }

    void create_worker_slice_shape_for_row_major_layout(
        const tt_xy_pair& /*tensor_slice_shape*/, uint32_t /*num_workers*/) {
        TT_THROW("Row major interleaved not supported by Reduce Scatter");
    }

    // Static methods
    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        const ttnn::Shape& tensor_shape,
        const tt_xy_pair& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_row_major_layout(
        const tt_xy_pair& tensor_slice_shape_in_elems, uint32_t num_workers, uint32_t max_slice_size_in_elements);

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
    ~RingReduceScatterTensorSlicer() override = default;
    RingReduceScatterTensorSlicer(
        const Tensor& input_tensor,
        const Tensor& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index) {
        return this->RingReduceScatterBaseTensorSlicer::get_worker_slice(global_worker_index, false);
    }  // False: Use the non wrapped version of the worker slice

    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        const ttnn::Shape& tensor_shape,
        const tt_xy_pair& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);
};

// Define a class RingReduceScatterWrappedTensor slicer that inherits from RingReduceScatterBaseTensorSlicer and
// overwrites the compute_worker_slice_offsets and create_worker_slice_shapes_for_tile_layout functions
class RingReduceScatterWrappedTensorSlicer
    : public RingReduceScatterBaseTensorSlicer<RingReduceScatterWrappedTensorSlicer> {
public:
    ~RingReduceScatterWrappedTensorSlicer() override = default;
    RingReduceScatterWrappedTensorSlicer(
        const Tensor& input_tensor,
        const Tensor& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index) {
        return this->RingReduceScatterBaseTensorSlicer::get_worker_slice(global_worker_index, true);
    }  // True: Use the wrapped version of the worker slice

    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        const std::vector<tt_xy_pair>& worker_slice_shapes, const tt_xy_pair& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        const ttnn::Shape& tensor_shape,
        const tt_xy_pair& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);
};

class InterleavedRingAllGatherTensorSlicer : public LegacyCclTensorSlicer {
public:
    ~InterleavedRingAllGatherTensorSlicer() override = default;
    InterleavedRingAllGatherTensorSlicer(
        const Tensor& input_tensor, const Tensor& output_tensor, int slice_dim, uint32_t slice_idx) {
        this->row_major = input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR;
        this->slice_dim_is_width = input_tensor.padded_shape().rank() - 1 == slice_dim;
        this->is_sharded = input_tensor.is_sharded();

        this->input_page_size = input_tensor.buffer()->page_size();

        if (row_major) {
            this->num_cols = input_tensor.padded_shape()[-1];
            const auto& input_shape = input_tensor.padded_shape();
            const auto& output_shape = output_tensor.padded_shape();
            this->num_rows = std::accumulate(
                input_shape.cbegin() + slice_dim, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
            this->row_offset =
                std::accumulate(
                    output_shape.cbegin() + slice_dim, output_shape.cend() - 1, 1, std::multiplies<uint32_t>()) -
                num_rows;
        } else {
            auto input_shape = input_tensor.padded_shape();
            const auto& output_shape = output_tensor.padded_shape();
            auto input_tile = input_tensor.tensor_spec().tile();
            auto output_tile = output_tensor.tensor_spec().tile();
            this->num_cols = input_shape[-1] / input_tile.get_width();
            uint32_t num_output_cols = output_tensor.padded_shape()[-1] / output_tile.get_width();
            this->num_rows =
                std::accumulate(
                    input_shape.cbegin() + slice_dim, input_shape.cend() - 1, 1, std::multiplies<uint32_t>()) /
                input_tile.get_height();
            this->row_offset =
                (std::accumulate(
                     output_shape.cbegin() + slice_dim, output_shape.cend() - 1, 1, std::multiplies<uint32_t>()) /
                     output_tile.get_height() -
                 num_rows) *
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

    void increment(uint32_t num_pages) override {
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
                this->output_start_page_idx +=
                    num_pages /*pages_per_worker*/ + num_rows_shifted * col_offset + num_blocks_shifted * row_offset;
                this->col_idx = num_cols_shifted % num_cols;
                this->row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
            }
        }
        this->input_start_page_idx += num_pages /*pages_per_worker*/;
    }
};

tt::tt_metal::KernelHandle generate_edm_kernel(
    tt::tt_metal::Program& program,
    const EriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id);

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    std::size_t num_buffers_per_channel,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
    EriscDataMoverTerminationMode termination_mode);

std::tuple<size_t, size_t, bool> get_forward_backward_configuration(
    size_t ring_size, size_t ring_index, Topology topology);

// Forward/backward devices are assumed to be neighbors for 1D fabric for now
std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>> get_forward_backward_line_unicast_configuration(
    const distributed::MeshCoordinate& src_device_coord,
    const std::optional<distributed::MeshCoordinate>& forward_device_coord,
    const std::optional<distributed::MeshCoordinate>& backward_device_coord,
    distributed::MeshDevice* mesh_device);

std::tuple<uint32_t, uint32_t> get_forward_backward_line_mcast_distance(
    size_t ring_size, size_t ring_index, Topology topology, bool static_alternate);

// Forward/backward devices are assumed to be neighbors for 1D fabric for now
std::tuple<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_forward_backward_line_mcast_configuration(
    const distributed::MeshCoordinate& src_device_coord,
    const std::optional<distributed::MeshCoordinate>& forward_device_coord,
    const std::optional<distributed::MeshCoordinate>& backward_device_coord,
    uint32_t num_targets_forward,
    uint32_t num_targets_backward,
    distributed::MeshDevice* mesh_device);


void fabric_mux_connection_ct_args(
    uint32_t num_workers_per_direction,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args);

void fabric_mux_connection_rt_args(
    bool mux_connection_valid,
    bool is_termination_master,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args,
    std::optional<uint32_t> = std::nullopt);

// ProgramDescriptor variant of the Program& helper above; produces the same 17-word client ABI. New semaphores are
// appended to desc.semaphores with IDs from find_available_semaphore_id, so they do not collide with IDs already
// allocated on worker_logical_core. A supplied termination_master_semaphore_id is reused, not allocated.
void fabric_mux_connection_rt_args(
    bool mux_connection_valid,
    bool is_termination_master,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::ProgramDescriptor& desc,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args,
    std::optional<uint32_t> termination_master_semaphore_id = std::nullopt);

// Fabric transfer time in device clock cycles, as a {bandwidth_cycles, latency_cycles} pair.
// bandwidth_cycles represents steady-state, latency_cycles is pipeline fill.
//   arch:          Wormhole or Blackhole
//   fabric_config: fabric config
//   clock_rate_mhz: device AICLK, used to convert ns -> cycles
//   data_bytes:    total bytes traversing the link
//   num_links:     number of parallel ethernet links
//   num_hops:      number of device hops
std::pair<int, int> estimate_fabric_transfer_cycles(
    tt::ARCH arch,
    tt::tt_fabric::FabricConfig fabric_config,
    int clock_rate_mhz,
    uint64_t data_bytes,
    uint32_t num_links,
    uint32_t num_hops);

}  // namespace ttnn::ccl
