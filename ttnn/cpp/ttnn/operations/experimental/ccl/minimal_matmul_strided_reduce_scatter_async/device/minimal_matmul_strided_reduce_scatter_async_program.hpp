// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_strided_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include <tt-metalium/runtime_args_data.hpp>
#include <array>
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

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate);
};

namespace detail {

// The op's types, for instantiating the descriptor adapter that
// MinimalMatmulStridedReduceScatterAsyncMeshWorkloadFactory wraps (the op itself names the wrapping factory in its
// program_factory_t).
struct MinimalMatmulStridedReduceScatterAsyncDescriptorAdapterOperation {
    using operation_attributes_t = MinimalMatmulStridedReduceScatterAsyncParams;
    using tensor_args_t = MinimalMatmulStridedReduceScatterAsyncInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
};

}  // namespace detail

// The descriptor workload plus a per-program cache of the runtime-arg slots that the cache-hit override rewrites, so a
// hit writes only the fields whose value changed instead of walking every core of five kernels on every device.
struct MinimalMatmulStridedReduceScatterAsyncMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::MinimalMatmulStridedReduceScatterAsyncDescriptorAdapterOperation>::
        DescriptorMeshWorkloadAdapter<MinimalMatmulStridedReduceScatterAsyncProgramFactory>;

    // Every value the cache-hit override writes: tensor / counter addresses, semaphores, the fused scalar.
    enum Field : uint32_t {
        kMatmulOutput,
        kRsIntermediate,
        kRsOutput,
        kInput,
        kWeight,
        kBias,
        kOptionalInput,
        kAddcmulA,
        kAddcmulB,
        kProgress,
        kCredit,
        kDirectionSemaphore0,
        kDirectionSemaphore1,
        kBatchSemaphore,
        kBarrierSemaphore,
        kReduceScalar,
        kNumFields,
    };
    using Slot = std::pair<tt::tt_metal::RuntimeArgsData*, uint32_t>;
    struct FastOverride {
        bool built = false;
        std::array<uint32_t, 8> structure{};  // the layout flags the slots were recorded under
        std::array<std::vector<Slot>, kNumFields> slots;
        std::array<uint32_t, kNumFields> written{};
        std::array<bool, kNumFields> valid{};
    };
    struct shared_variables_t {
        typename descriptor_adapter_t::shared_variables_t descriptor;
        FastOverride fast;
        // Same idea for the resolved buffer bindings: per collected buffer, its runtime-arg slots and the address
        // last written there; a hit rewrites only the slots of buffers whose address changed.
        std::vector<std::vector<Slot>> binding_slots;
        std::vector<uint64_t> bound_addresses;
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
