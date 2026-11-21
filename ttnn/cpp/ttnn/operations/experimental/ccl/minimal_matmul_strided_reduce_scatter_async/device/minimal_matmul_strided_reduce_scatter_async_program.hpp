// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_strided_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include <tt-metalium/runtime_args_data.hpp>
#include <array>
#include <optional>
#include <utility>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncProgramFactory {
    // build_ring_strided_reduce_scatter_async_program_artifacts pushes mux, reader, writer, reduce first.
    static constexpr uint32_t kReaderKernelIdx = 1;
    static constexpr uint32_t kWriterKernelIdx = 2;
    static constexpr uint32_t kReduceKernelIdx = 3;
    // minimal_matmul_factory_helper_common appends these next. Ternary is fused in the reduce-scatter, not here.
    static constexpr uint32_t kMatmulIn0SenderKernelIdx = 4;
    static constexpr uint32_t kMatmulIn0ReceiverKernelIdx = 5;
    static constexpr uint32_t kMatmulIn1SenderKernelIdx = 6;
    static constexpr uint32_t kMatmulIn1ReceiverKernelIdx = 7;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

namespace detail {

// The op's types, for the descriptor adapter that MinimalMatmulStridedReduceScatterAsyncMeshWorkloadFactory wraps.
struct MinimalMatmulStridedReduceScatterAsyncAdapterOperation {
    using operation_attributes_t = MinimalMatmulStridedReduceScatterAsyncParams;
    using tensor_args_t = MinimalMatmulStridedReduceScatterAsyncInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
};

}  // namespace detail

// The descriptor workload plus, per cached program, the slots of the resolved buffer bindings and of the caller-owned
// progress and credit counters: a cache hit rewrites a slot only when its address changes, instead of re-applying
// every binding and walking every reader and matmul core of every per-device program.
struct MinimalMatmulStridedReduceScatterAsyncMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::MinimalMatmulStridedReduceScatterAsyncAdapterOperation>::
        DescriptorMeshWorkloadAdapter<MinimalMatmulStridedReduceScatterAsyncProgramFactory>;

    enum Counter : uint32_t { kProgress, kCredit, kNumCounters };
    struct CounterSlots {
        bool recorded = false;
        std::array<std::vector<std::pair<tt::tt_metal::RuntimeArgsData*, uint32_t>>, kNumCounters> slots;
        std::array<std::optional<uint32_t>, kNumCounters> written;
    };
    struct BindingSlots {
        bool recorded = false;
        std::vector<tt::tt_metal::RuntimeArgsData*> rt_args;
        std::vector<std::optional<uint32_t>> rt_written;
        std::vector<std::optional<uint32_t>> cb_written;
    };
    struct shared_variables_t {
        typename descriptor_adapter_t::shared_variables_t descriptor;
        CounterSlots counters;
        BindingSlots bindings;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
