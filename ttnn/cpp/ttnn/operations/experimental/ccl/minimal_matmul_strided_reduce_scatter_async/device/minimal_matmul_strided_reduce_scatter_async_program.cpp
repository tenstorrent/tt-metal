// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <cstring>
#include <limits>
#include <span>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.hpp"

// Include RS types
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_ring_program_factory.hpp"

using namespace tt::constants;

// Import the RS program artifacts type
using ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterProgramArtifacts;

namespace ttnn::experimental::prim {

void MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor,
    const std::optional<ttnn::MeshCoordinate>&) {
    // Also rewrites tensor addresses: aliased addcmul inputs empty resolve_bindings, and this workload does not
    // rebuild. Caller-owned progress and credit counters are refreshed here too; workload-owned counters stay.
    constexpr uint32_t kReaderInputArg = 0;
    constexpr uint32_t kReaderIntermediateArg = 1;
    constexpr uint32_t kReaderSemaphoreArg = 2;
    constexpr uint32_t kReaderDirectionArg = 3;
    constexpr uint32_t kWriterIntermediateArg = 0;
    constexpr uint32_t kWriterOutputArg = 1;
    constexpr uint32_t kWriterDirectionSemaphoreArg = 4;
    constexpr uint32_t kWriterBatchSemaphoreArg = 5;
    constexpr uint32_t kWriterBarrierSemaphoreArg = 7;
    constexpr uint32_t kWriterDirectionArg = 8;
    constexpr uint32_t kReduceScalarArg = 3;
    constexpr uint32_t kNumDirections = 2;
    constexpr uint32_t kIn0InputArg = 0;
    constexpr uint32_t kIn0BiasArg = 1;
    constexpr uint32_t kIn0OptionalInputArg = 2;
    constexpr uint32_t kIn0OutputArg = 14;
    constexpr uint32_t kIn1WeightArg = 0;
    constexpr uint32_t kIn1BiasArg = 1;
    constexpr uint32_t kIn1OutputArg = 13;
    constexpr uint32_t kReaderArgPrefix = 6;
    constexpr uint32_t kWindowedProgressFromEnd = 4;
    constexpr uint32_t kWindowedCreditFromEnd = 2;

    TT_FATAL(output_tensor.size() == 3, "minimal_matmul_strided_reduce_scatter returns 3 tensors");
    TT_FATAL(output_tensor[0].is_allocated(), "matmul output must have a buffer");
    TT_FATAL(output_tensor[1].is_allocated(), "reduce-scatter intermediate must have a buffer");
    TT_FATAL(output_tensor[2].is_allocated(), "reduce-scatter output must have a buffer");
    TT_FATAL(tensor_args.input_tensor.is_allocated(), "matmul input must have a buffer");
    TT_FATAL(tensor_args.weight_tensor.is_allocated(), "matmul weight must have a buffer");
    const uint32_t mm_output_addr = static_cast<uint32_t>(output_tensor[0].buffer()->address());
    const uint32_t rs_intermediate_addr = static_cast<uint32_t>(output_tensor[1].buffer()->address());
    const uint32_t rs_output_addr = static_cast<uint32_t>(output_tensor[2].buffer()->address());
    const uint32_t input_addr = static_cast<uint32_t>(tensor_args.input_tensor.buffer()->address());
    const uint32_t weight_addr = static_cast<uint32_t>(tensor_args.weight_tensor.buffer()->address());
    uint32_t bias_addr = 0;
    if (tensor_args.bias.has_value()) {
        TT_FATAL(tensor_args.bias->is_allocated(), "matmul bias must have a buffer");
        bias_addr = static_cast<uint32_t>(tensor_args.bias->buffer()->address());
    }
    uint32_t optional_input_addr = 0;
    if (tensor_args.mm_optional_input_tensor.has_value()) {
        TT_FATAL(tensor_args.mm_optional_input_tensor->is_allocated(), "optional matmul input must have a buffer");
        optional_input_addr = static_cast<uint32_t>(tensor_args.mm_optional_input_tensor->buffer()->address());
    }
    const bool fuse_addcmul =
        tensor_args.addcmul_input_tensor1.has_value() && tensor_args.addcmul_input_tensor2.has_value();
    uint32_t addcmul_a_addr = 0;
    uint32_t addcmul_b_addr = 0;
    if (fuse_addcmul) {
        TT_FATAL(tensor_args.addcmul_input_tensor1->is_allocated(), "addcmul input A must have a buffer");
        TT_FATAL(tensor_args.addcmul_input_tensor2->is_allocated(), "addcmul input B must have a buffer");
        addcmul_a_addr = static_cast<uint32_t>(tensor_args.addcmul_input_tensor1->buffer()->address());
        addcmul_b_addr = static_cast<uint32_t>(tensor_args.addcmul_input_tensor2->buffer()->address());
    }
    const bool windowed = operation_attributes.mm_window_blocks.has_value();
    const bool caller_progress = tensor_args.mm_progress_counters.has_value();
    const bool caller_credit = windowed && tensor_args.mm_credit_counters.has_value();
    uint32_t progress_addr = 0;
    if (caller_progress) {
        TT_FATAL(tensor_args.mm_progress_counters->is_allocated(), "mm_progress_counters must have a buffer");
        progress_addr = static_cast<uint32_t>(tensor_args.mm_progress_counters->buffer()->address());
    }
    uint32_t credit_addr = 0;
    if (caller_credit) {
        TT_FATAL(tensor_args.mm_credit_counters->is_allocated(), "mm_credit_counters must have a buffer");
        credit_addr = static_cast<uint32_t>(tensor_args.mm_credit_counters->buffer()->address());
    }
    uint32_t reader_progress_arg = kReaderArgPrefix;
    if ((caller_progress || caller_credit) && output_tensor[1].is_sharded()) {
        std::vector<uint32_t> shard_args;
        shard_builder::extend_sharding_run_time_args(output_tensor[1], shard_args);
        reader_progress_arg += static_cast<uint32_t>(shard_args.size());
    }
    const uint32_t reader_credit_arg = reader_progress_arg + 1;

    TT_FATAL(
        operation_attributes.semaphore.size() > kNumDirections,
        "strided reduce scatter expects one semaphore per direction plus the batch semaphore");
    const auto batch_semaphore = static_cast<uint32_t>(operation_attributes.semaphore.at(kNumDirections).address());
    const auto barrier_semaphore = operation_attributes.barrier_semaphore.has_value()
                                       ? static_cast<uint32_t>(operation_attributes.barrier_semaphore->address())
                                       : 0u;

    auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx);
    for (auto& column : reader_runtime_args) {
        for (auto& args : column) {
            if (args.size() <= kReaderDirectionArg) {
                continue;
            }
            args[kReaderInputArg] = mm_output_addr;
            args[kReaderIntermediateArg] = rs_intermediate_addr;
            args[kReaderSemaphoreArg] =
                static_cast<uint32_t>(operation_attributes.semaphore.at(args[kReaderDirectionArg]).address());
            if (fuse_addcmul) {
                args[args.size() - 2] = addcmul_a_addr;
                args[args.size() - 1] = addcmul_b_addr;
            }
            if (caller_progress) {
                TT_FATAL(args.size() > reader_progress_arg, "reduce-scatter reader is missing the progress counter");
                args[reader_progress_arg] = progress_addr;
            }
            if (caller_credit) {
                TT_FATAL(args.size() > reader_credit_arg, "reduce-scatter reader is missing the credit counter");
                args[reader_credit_arg] = credit_addr;
            }
        }
    }

    auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx);
    for (auto& column : writer_runtime_args) {
        for (auto& args : column) {
            if (args.size() <= kWriterDirectionArg) {
                continue;
            }
            args[kWriterIntermediateArg] = rs_intermediate_addr;
            args[kWriterOutputArg] = rs_output_addr;
            args[kWriterDirectionSemaphoreArg] =
                static_cast<uint32_t>(operation_attributes.semaphore.at(args[kWriterDirectionArg]).address());
            args[kWriterBatchSemaphoreArg] = batch_semaphore;
            if (operation_attributes.barrier_semaphore.has_value()) {
                args[kWriterBarrierSemaphoreArg] = barrier_semaphore;
            }
        }
    }

    TT_FATAL(
        operation_attributes.matmul_struct.config.has_value(),
        "minimal_matmul_strided_reduce_scatter requires a matmul config");
    const auto grid = operation_attributes.matmul_struct.config->compute_with_storage_grid_size;
    auto& in0_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn0SenderKernelIdx);
    auto& in0_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn0ReceiverKernelIdx);
    auto& in1_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn1SenderKernelIdx);
    auto& in1_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kMatmulIn1ReceiverKernelIdx);
    const auto refresh_matmul_counters = [&](auto& args) {
        if (caller_progress) {
            TT_FATAL(
                !windowed || args.size() >= kWindowedProgressFromEnd,
                "matmul runtime args are missing the progress counter");
            const uint32_t progress_arg = windowed ? static_cast<uint32_t>(args.size() - kWindowedProgressFromEnd)
                                                   : static_cast<uint32_t>(args.size() - 1);
            args[progress_arg] = progress_addr;
        }
        if (caller_credit) {
            TT_FATAL(args.size() >= kWindowedCreditFromEnd, "matmul runtime args are missing the credit counter");
            args[args.size() - kWindowedCreditFromEnd] = credit_addr;
        }
    };
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            auto& in0_grid = x == 0 ? in0_sender_runtime_args : in0_receiver_runtime_args;
            if (x < in0_grid.size() && y < in0_grid[x].size() && in0_grid[x][y].size() > kIn0OutputArg) {
                auto& in0_args = in0_grid[x][y];
                in0_args[kIn0InputArg] = input_addr;
                in0_args[kIn0BiasArg] = bias_addr;
                in0_args[kIn0OptionalInputArg] = optional_input_addr;
                in0_args[kIn0OutputArg] = mm_output_addr;
                refresh_matmul_counters(in0_args);
            }
            auto& in1_grid = y == 0 ? in1_sender_runtime_args : in1_receiver_runtime_args;
            if (x < in1_grid.size() && y < in1_grid[x].size() && in1_grid[x][y].size() > kIn1OutputArg) {
                auto& in1_args = in1_grid[x][y];
                in1_args[kIn1WeightArg] = weight_addr;
                in1_args[kIn1BiasArg] = bias_addr;
                in1_args[kIn1OutputArg] = mm_output_addr;
                refresh_matmul_counters(in1_args);
            }
        }
    }

    if (operation_attributes.fused_ternary_scalar.has_value()) {
        float scalar_f = operation_attributes.fused_ternary_scalar.value();
        uint32_t scalar_u32 = 0;
        std::memcpy(&scalar_u32, &scalar_f, sizeof(uint32_t));
        auto& reduce_runtime_args = tt::tt_metal::GetRuntimeArgs(program, kReduceKernelIdx);
        for (auto& column : reduce_runtime_args) {
            for (auto& args : column) {
                if (args.size() > kReduceScalarArg) {
                    args[kReduceScalarArg] = scalar_u32;
                }
            }
        }
    }
}

using MmrsMeshWorkloadFactory = MinimalMatmulStridedReduceScatterAsyncMeshWorkloadFactory;

MmrsMeshWorkloadFactory::cached_mesh_workload_t MmrsMeshWorkloadFactory::create_mesh_workload(
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto descriptor_cached = descriptor_adapter_t::create_mesh_workload(
        operation_attributes, tensor_coords, tensor_args, tensor_return_value);
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (auto& [range, program] : descriptor_cached.workload.get_programs()) {
        shared_variables[range].descriptor = std::move(descriptor_cached.shared_variables.at(range));
    }
    return {std::move(descriptor_cached.workload), std::move(shared_variables)};
}

namespace {

// Same writes as MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments (same slots, same
// values), but the slots of each field are recorded once per cached program and a hit rewrites only the fields whose
// value changed. RuntimeArgsData objects live in each kernel's fixed per-core table and dispatch retargets them in
// place, so the recorded object pointers stay valid.
void fast_override(
    tt::tt_metal::Program& program,
    MmrsMeshWorkloadFactory::FastOverride& fast,
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensor) {
    using F = MmrsMeshWorkloadFactory;
    using PF = MinimalMatmulStridedReduceScatterAsyncProgramFactory;
    // Slot layout: must match the constants of PF::override_runtime_arguments.
    constexpr uint32_t kReaderInputArg = 0;
    constexpr uint32_t kReaderIntermediateArg = 1;
    constexpr uint32_t kReaderSemaphoreArg = 2;
    constexpr uint32_t kReaderDirectionArg = 3;
    constexpr uint32_t kWriterIntermediateArg = 0;
    constexpr uint32_t kWriterOutputArg = 1;
    constexpr uint32_t kWriterDirectionSemaphoreArg = 4;
    constexpr uint32_t kWriterBatchSemaphoreArg = 5;
    constexpr uint32_t kWriterBarrierSemaphoreArg = 7;
    constexpr uint32_t kWriterDirectionArg = 8;
    constexpr uint32_t kReduceScalarArg = 3;
    constexpr uint32_t kNumDirections = 2;
    constexpr uint32_t kIn0InputArg = 0;
    constexpr uint32_t kIn0BiasArg = 1;
    constexpr uint32_t kIn0OptionalInputArg = 2;
    constexpr uint32_t kIn0OutputArg = 14;
    constexpr uint32_t kIn1WeightArg = 0;
    constexpr uint32_t kIn1BiasArg = 1;
    constexpr uint32_t kIn1OutputArg = 13;
    constexpr uint32_t kReaderArgPrefix = 6;
    constexpr uint32_t kWindowedProgressFromEnd = 4;
    constexpr uint32_t kWindowedCreditFromEnd = 2;

    TT_FATAL(output_tensor.size() == 3, "minimal_matmul_strided_reduce_scatter returns 3 tensors");
    TT_FATAL(output_tensor[0].is_allocated(), "matmul output must have a buffer");
    TT_FATAL(output_tensor[1].is_allocated(), "reduce-scatter intermediate must have a buffer");
    TT_FATAL(output_tensor[2].is_allocated(), "reduce-scatter output must have a buffer");
    TT_FATAL(tensor_args.input_tensor.is_allocated(), "matmul input must have a buffer");
    TT_FATAL(tensor_args.weight_tensor.is_allocated(), "matmul weight must have a buffer");
    TT_FATAL(
        operation_attributes.semaphore.size() > kNumDirections,
        "strided reduce scatter expects one semaphore per direction plus the batch semaphore");
    TT_FATAL(
        operation_attributes.matmul_struct.config.has_value(),
        "minimal_matmul_strided_reduce_scatter requires a matmul config");

    std::array<uint32_t, F::kNumFields> value{};
    value[F::kMatmulOutput] = static_cast<uint32_t>(output_tensor[0].buffer()->address());
    value[F::kRsIntermediate] = static_cast<uint32_t>(output_tensor[1].buffer()->address());
    value[F::kRsOutput] = static_cast<uint32_t>(output_tensor[2].buffer()->address());
    value[F::kInput] = static_cast<uint32_t>(tensor_args.input_tensor.buffer()->address());
    value[F::kWeight] = static_cast<uint32_t>(tensor_args.weight_tensor.buffer()->address());
    if (tensor_args.bias.has_value()) {
        TT_FATAL(tensor_args.bias->is_allocated(), "matmul bias must have a buffer");
        value[F::kBias] = static_cast<uint32_t>(tensor_args.bias->buffer()->address());
    }
    if (tensor_args.mm_optional_input_tensor.has_value()) {
        TT_FATAL(tensor_args.mm_optional_input_tensor->is_allocated(), "optional matmul input must have a buffer");
        value[F::kOptionalInput] = static_cast<uint32_t>(tensor_args.mm_optional_input_tensor->buffer()->address());
    }
    const bool fuse_addcmul =
        tensor_args.addcmul_input_tensor1.has_value() && tensor_args.addcmul_input_tensor2.has_value();
    if (fuse_addcmul) {
        TT_FATAL(tensor_args.addcmul_input_tensor1->is_allocated(), "addcmul input A must have a buffer");
        TT_FATAL(tensor_args.addcmul_input_tensor2->is_allocated(), "addcmul input B must have a buffer");
        value[F::kAddcmulA] = static_cast<uint32_t>(tensor_args.addcmul_input_tensor1->buffer()->address());
        value[F::kAddcmulB] = static_cast<uint32_t>(tensor_args.addcmul_input_tensor2->buffer()->address());
    }
    const bool windowed = operation_attributes.mm_window_blocks.has_value();
    const bool caller_progress = tensor_args.mm_progress_counters.has_value();
    const bool caller_credit = windowed && tensor_args.mm_credit_counters.has_value();
    if (caller_progress) {
        TT_FATAL(tensor_args.mm_progress_counters->is_allocated(), "mm_progress_counters must have a buffer");
        value[F::kProgress] = static_cast<uint32_t>(tensor_args.mm_progress_counters->buffer()->address());
    }
    if (caller_credit) {
        TT_FATAL(tensor_args.mm_credit_counters->is_allocated(), "mm_credit_counters must have a buffer");
        value[F::kCredit] = static_cast<uint32_t>(tensor_args.mm_credit_counters->buffer()->address());
    }
    value[F::kDirectionSemaphore0] = static_cast<uint32_t>(operation_attributes.semaphore.at(0).address());
    value[F::kDirectionSemaphore1] = static_cast<uint32_t>(operation_attributes.semaphore.at(1).address());
    value[F::kBatchSemaphore] = static_cast<uint32_t>(operation_attributes.semaphore.at(kNumDirections).address());
    const bool has_barrier = operation_attributes.barrier_semaphore.has_value();
    if (has_barrier) {
        value[F::kBarrierSemaphore] = static_cast<uint32_t>(operation_attributes.barrier_semaphore->address());
    }
    const bool has_scalar = operation_attributes.fused_ternary_scalar.has_value();
    if (has_scalar) {
        const float scalar_f = operation_attributes.fused_ternary_scalar.value();
        std::memcpy(&value[F::kReduceScalar], &scalar_f, sizeof(uint32_t));
    }

    const auto grid = operation_attributes.matmul_struct.config->compute_with_storage_grid_size;
    uint32_t reader_progress_arg = kReaderArgPrefix;
    if ((caller_progress || caller_credit) && output_tensor[1].is_sharded()) {
        std::vector<uint32_t> shard_args;
        shard_builder::extend_sharding_run_time_args(output_tensor[1], shard_args);
        reader_progress_arg += static_cast<uint32_t>(shard_args.size());
    }
    const uint32_t reader_credit_arg = reader_progress_arg + 1;
    const std::array<uint32_t, 8> structure = {
        fuse_addcmul,
        caller_progress,
        caller_credit,
        windowed,
        reader_progress_arg,
        has_barrier,
        has_scalar,
        static_cast<uint32_t>(grid.x * 4096 + grid.y)};

    if (!fast.built || fast.structure != structure) {
        for (auto& s : fast.slots) {
            s.clear();
        }
        const auto add = [&](F::Field f, tt::tt_metal::RuntimeArgsData& args, uint32_t idx) {
            fast.slots[f].emplace_back(&args, idx);
        };
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kReaderKernelIdx);
        for (auto& column : reader_runtime_args) {
            for (auto& args : column) {
                if (args.size() <= kReaderDirectionArg) {
                    continue;
                }
                const uint32_t direction = args[kReaderDirectionArg];
                TT_FATAL(direction < kNumDirections, "reduce-scatter reader direction {} out of range", direction);
                add(F::kMatmulOutput, args, kReaderInputArg);
                add(F::kRsIntermediate, args, kReaderIntermediateArg);
                add(direction == 0 ? F::kDirectionSemaphore0 : F::kDirectionSemaphore1, args, kReaderSemaphoreArg);
                if (fuse_addcmul) {
                    add(F::kAddcmulA, args, static_cast<uint32_t>(args.size() - 2));
                    add(F::kAddcmulB, args, static_cast<uint32_t>(args.size() - 1));
                }
                if (caller_progress) {
                    TT_FATAL(
                        args.size() > reader_progress_arg, "reduce-scatter reader is missing the progress counter");
                    add(F::kProgress, args, reader_progress_arg);
                }
                if (caller_credit) {
                    TT_FATAL(args.size() > reader_credit_arg, "reduce-scatter reader is missing the credit counter");
                    add(F::kCredit, args, reader_credit_arg);
                }
            }
        }
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kWriterKernelIdx);
        for (auto& column : writer_runtime_args) {
            for (auto& args : column) {
                if (args.size() <= kWriterDirectionArg) {
                    continue;
                }
                const uint32_t direction = args[kWriterDirectionArg];
                TT_FATAL(direction < kNumDirections, "reduce-scatter writer direction {} out of range", direction);
                add(F::kRsIntermediate, args, kWriterIntermediateArg);
                add(F::kRsOutput, args, kWriterOutputArg);
                add(direction == 0 ? F::kDirectionSemaphore0 : F::kDirectionSemaphore1,
                    args,
                    kWriterDirectionSemaphoreArg);
                add(F::kBatchSemaphore, args, kWriterBatchSemaphoreArg);
                if (has_barrier) {
                    add(F::kBarrierSemaphore, args, kWriterBarrierSemaphoreArg);
                }
            }
        }
        auto& in0_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kMatmulIn0SenderKernelIdx);
        auto& in0_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kMatmulIn0ReceiverKernelIdx);
        auto& in1_sender_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kMatmulIn1SenderKernelIdx);
        auto& in1_receiver_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kMatmulIn1ReceiverKernelIdx);
        const auto add_matmul_counters = [&](tt::tt_metal::RuntimeArgsData& args) {
            if (caller_progress) {
                TT_FATAL(
                    !windowed || args.size() >= kWindowedProgressFromEnd,
                    "matmul runtime args are missing the progress counter");
                add(F::kProgress,
                    args,
                    windowed ? static_cast<uint32_t>(args.size() - kWindowedProgressFromEnd)
                             : static_cast<uint32_t>(args.size() - 1));
            }
            if (caller_credit) {
                TT_FATAL(args.size() >= kWindowedCreditFromEnd, "matmul runtime args are missing the credit counter");
                add(F::kCredit, args, static_cast<uint32_t>(args.size() - kWindowedCreditFromEnd));
            }
        };
        for (uint32_t x = 0; x < grid.x; ++x) {
            for (uint32_t y = 0; y < grid.y; ++y) {
                auto& in0_grid = x == 0 ? in0_sender_runtime_args : in0_receiver_runtime_args;
                if (x < in0_grid.size() && y < in0_grid[x].size() && in0_grid[x][y].size() > kIn0OutputArg) {
                    auto& in0_args = in0_grid[x][y];
                    add(F::kInput, in0_args, kIn0InputArg);
                    add(F::kBias, in0_args, kIn0BiasArg);
                    add(F::kOptionalInput, in0_args, kIn0OptionalInputArg);
                    add(F::kMatmulOutput, in0_args, kIn0OutputArg);
                    add_matmul_counters(in0_args);
                }
                auto& in1_grid = y == 0 ? in1_sender_runtime_args : in1_receiver_runtime_args;
                if (x < in1_grid.size() && y < in1_grid[x].size() && in1_grid[x][y].size() > kIn1OutputArg) {
                    auto& in1_args = in1_grid[x][y];
                    add(F::kWeight, in1_args, kIn1WeightArg);
                    add(F::kBias, in1_args, kIn1BiasArg);
                    add(F::kMatmulOutput, in1_args, kIn1OutputArg);
                    add_matmul_counters(in1_args);
                }
            }
        }
        if (has_scalar) {
            auto& reduce_runtime_args = tt::tt_metal::GetRuntimeArgs(program, PF::kReduceKernelIdx);
            for (auto& column : reduce_runtime_args) {
                for (auto& args : column) {
                    if (args.size() > kReduceScalarArg) {
                        add(F::kReduceScalar, args, kReduceScalarArg);
                    }
                }
            }
        }
        fast.structure = structure;
        fast.valid.fill(false);
        fast.built = true;
    }

    for (uint32_t f = 0; f < F::kNumFields; ++f) {
        if (fast.valid[f] && fast.written[f] == value[f]) {
            continue;
        }
        for (const auto& [args, idx] : fast.slots[f]) {
            (*args)[idx] = value[f];
        }
        fast.written[f] = value[f];
        fast.valid[f] = true;
    }
}

// tt::tt_metal::apply_resolved_bindings restricted to buffers whose address changed since the last hit of this program
// (same values, same slots); CB bindings take the generic path.
void apply_changed_bindings(
    tt::tt_metal::Program& program,
    MmrsMeshWorkloadFactory::shared_variables_t& shared,
    std::span<tt::tt_metal::Buffer* const> buffers) {
    const auto& bindings = shared.descriptor.resolved_bindings;
    if (!bindings.cbs.empty()) {
        tt::tt_metal::apply_resolved_bindings(program, bindings, buffers);
        return;
    }
    if (shared.binding_slots.size() != buffers.size()) {
        shared.binding_slots.assign(buffers.size(), {});
        for (const auto& b : bindings.rt_args) {
            TT_FATAL(b.tensor_buffer_idx < buffers.size(), "binding buffer index {} out of range", b.tensor_buffer_idx);
            auto* data = b.is_common ? &tt::tt_metal::GetCommonRuntimeArgs(program, b.kernel_idx)
                                     : &tt::tt_metal::GetRuntimeArgs(program, b.kernel_idx, b.core);
            shared.binding_slots[b.tensor_buffer_idx].emplace_back(data, b.arg_idx);
        }
        shared.bound_addresses.assign(buffers.size(), std::numeric_limits<uint64_t>::max());
    }
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto& slots = shared.binding_slots[i];
        if (slots.empty()) {
            continue;
        }
        const uint64_t address = buffers[i]->address();
        if (address == shared.bound_addresses[i]) {
            continue;
        }
        for (const auto& [data, arg_idx] : slots) {
            (*data)[arg_idx] = static_cast<uint32_t>(address);
        }
        shared.bound_addresses[i] = address;
    }
}

}  // namespace

void MmrsMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    if (cached_workload.shared_variables.empty()) {
        return;
    }
    // Every program's descriptor copy holds the same workload resources, so the buffers are collected once.
    const auto collected = descriptor_adapter_t::collect_tensor_buffers(
        tensor_args,
        tensor_return_value,
        cached_workload.shared_variables.begin()->second.descriptor.workload_descriptor);
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared = cached_workload.shared_variables.at(coordinate_range);
        // What the WorkloadDescriptor adapter does on a hit: re-apply the resolved buffer bindings, then the
        // factory's override.
        if (!operation_attributes.changed_args_only) {
            if (!shared.descriptor.resolved_bindings.empty()) {
                tt::tt_metal::apply_resolved_bindings(program, shared.descriptor.resolved_bindings, collected.buffers);
            }
            MinimalMatmulStridedReduceScatterAsyncProgramFactory::override_runtime_arguments(
                program,
                operation_attributes,
                tensor_args,
                tensor_return_value,
                std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
            continue;
        }
        if (!shared.descriptor.resolved_bindings.empty()) {
            apply_changed_bindings(program, shared, collected.buffers);
        }
        fast_override(program, shared.fast, operation_attributes, tensor_args, tensor_return_value);
    }
}

struct FusedProgram {
    tt::tt_metal::ProgramDescriptor descriptor;
    StridedReduceScatterProgramArtifacts rs_artifacts;
};

FusedProgram minimal_matmul_strided_reduce_scatter_async_program(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    Tensor& rs_intermediate_tensor,
    Tensor& rs_output_tensor,

    /* Reduce Scatter Params */
    const MeshCoordinate& target_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord reduce_scatter_core_grid_offset,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    std::optional<uint32_t> mm_window_blocks,
    const std::optional<const Tensor>& mm_credit_counters,

    /* Matmul Params */
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    ttnn::experimental::prim::MinimalMatmulConfig config,
    DeviceComputeKernelConfig compute_kernel_config,

    /* Fused addcmul params */
    const std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor1 = std::nullopt,
    const std::optional<const Tensor>& addcmul_input_tensor2 = std::nullopt,

    /* Caller-owned per-MM-core progress counter scratch shared across MMRS programs. */
    const std::optional<const Tensor>& mm_progress_counters = std::nullopt,
    /* Fused concat (concat-free): second in0 source (suffix half of K; input_tensor is the prefix). */
    const std::optional<const Tensor>& mm_optional_input_tensor = std::nullopt) {
    tt::tt_metal::ProgramDescriptor program{};

    // Derive matmul geometry parameters for the RS factory.
    // The matmul factory normally transposes its core grid when M > N, but
    // transposition is disabled when fusing with SRS (the RS iteration structure
    // requires mm_N_block_wt <= slice_Wt, which can be violated when transposing).
    // So we always use the non-transposed layout: M parallelized on y, N on x.
    uint32_t mm_cores_y_val = config.compute_with_storage_grid_size.y;
    uint32_t mm_block_ht_val = config.M_block_size;
    uint32_t mm_block_wt_val = config.N_block_size;

    // Compute mm_N_full_block_wt: total N tiles per core (N parallelized on x)
    uint32_t N_tiles = weight_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_cores_x = config.compute_with_storage_grid_size.x;
    uint32_t padded_N_tiles = tt::round_up(N_tiles, num_cores_x);
    uint32_t mm_N_full_block_wt_val = padded_N_tiles / num_cores_x;

    // With a rolling window the matmul output tensor is only mm_window_blocks M blocks tall, so the
    // RS needs the true height told to it separately. Take it from the activations, which are always
    // full height.
    const std::optional<uint32_t> mm_logical_Ht_val =
        mm_window_blocks.has_value() ? std::optional<uint32_t>(input_tensor.padded_shape()[-2] / TILE_HEIGHT)
                                     : std::nullopt;

    // =========================================================================
    // STEP 1: Create the Reduce Scatter program FIRST
    //
    // The RS factory creates a semaphore on the RS reader cores and records
    // their NOC coordinates. This info is captured in srs_fused_op_signaler,
    // which is then passed to the matmul factory in step 2.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler =
        ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler();
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> empty_rs_fused_op_signaler = std::nullopt;

    auto rs_shared_variables = ::ttnn::build_ring_strided_reduce_scatter_async_program_artifacts(
        program,
        matmul_output_tensor,    // RS input = MM output
        rs_intermediate_tensor,  // RS intermediate (scratch)
        target_device_coord,
        forward_coord,
        backward_coord,
        rs_output_tensor,  // RS output
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        empty_rs_fused_op_signaler,  // RS -> next op signaling (not used)
        srs_fused_op_signaler,       // MM -> RS signaling (populated by RS factory)
        num_workers_per_link,
        num_buffers_per_channel,
        reduce_scatter_core_grid_offset,
        mm_cores_y_val,
        mm_block_ht_val,
        mm_block_wt_val,
        mm_N_full_block_wt_val,
        chunk_width_in_mm_blocks,
        mm_window_blocks,
        mm_logical_Ht_val,
        mm_credit_counters,
        // Phase 2: fuse addcmul at the RS final write step (not in MM kernel)
        fused_ternary_scalar,
        addcmul_input_tensor1,
        addcmul_input_tensor2,
        mm_progress_counters);

    // =========================================================================
    // STEP 2: Create the Matmul program SECOND
    //
    // The matmul factory receives the populated srs_fused_op_signaler, which
    // contains the RS reader cores' NOC coordinates and semaphore ID. The
    // matmul kernels use this to signal the RS when output blocks are ready.
    // =========================================================================
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_mm_fused_op_signaler;

    std::vector<Tensor> mm_output_tensors = {matmul_output_tensor};
    auto mm_shared_variables = ttnn::experimental::prim::minimal_matmul_factory_helper_common(
        program,
        input_tensor,   // MM input (activations)
        weight_tensor,  // MM weights
        bias,
        fused_activation,
        config,
        mm_output_tensors,  // MM output (= RS input)
        compute_kernel_config,
        empty_mm_fused_op_signaler,  // No AG -> MM signaling
        1,                           // N_chunks = 1
        std::nullopt,                // ternary fused in RS, not MM
        std::nullopt,
        std::nullopt,
        srs_fused_op_signaler,  // MM -> RS signaling (populated from step 1)
        false,                  // fuse_swiglu
        mm_optional_input_tensor);

    (void)mm_shared_variables;
    return {std::move(program), std::move(rs_shared_variables)};
}

tt::tt_metal::WorkloadDescriptor MinimalMatmulStridedReduceScatterAsyncProgramFactory::create_workload_descriptor(
    const MinimalMatmulStridedReduceScatterAsyncParams& operation_attributes,
    const MinimalMatmulStridedReduceScatterAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& attributes = operation_attributes;
    auto& output_tensor = tensor_return_value;
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& mesh_coordinate : tensor_coords.coords()) {
        uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);

        // output_tensor[0] = MM output (= RS input)
        // output_tensor[1] = RS intermediate
        // output_tensor[2] = RS output
        auto built = minimal_matmul_strided_reduce_scatter_async_program(
            tensor_args.input_tensor,   // MM input (activations)
            tensor_args.weight_tensor,  // MM weights
            output_tensor[0],           // MM output = RS input
            output_tensor[1],           // RS intermediate
            output_tensor[2],           // RS output

            /* Reduce Scatter Params */
            mesh_coordinate,
            forward_coord,
            backward_coord,
            attributes.dim,
            attributes.num_links,
            attributes.ring_size,
            device_index,
            attributes.topology,
            attributes.semaphore,
            attributes.barrier_semaphore,
            attributes.using_persistent_buffers,
            attributes.sub_device_id,
            attributes.num_workers_per_link,
            attributes.num_buffers_per_channel,
            attributes.reduce_scatter_core_grid_offset,
            attributes.chunk_width_in_mm_blocks,
            attributes.mm_window_blocks,
            tensor_args.mm_credit_counters,

            /* Matmul Params */
            tensor_args.bias,
            attributes.matmul_struct.fused_activation,
            attributes.matmul_struct.config.value(),
            attributes.matmul_struct.compute_kernel_config,

            /* Fused addcmul params */
            attributes.fused_ternary_scalar,
            tensor_args.addcmul_input_tensor1,
            tensor_args.addcmul_input_tensor2,

            /* Shared MM->RS progress counter scratch */
            tensor_args.mm_progress_counters,
            /* Fused concat: second in0 source */
            tensor_args.mm_optional_input_tensor);

        auto park = [&](const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& mesh_buffer) {
            if (!mesh_buffer) {
                return;
            }
            workload.buffers.push_back(tt::tt_metal::WorkloadBuffer{
                .owner = mesh_buffer,
                // Same reference buffer the ring builder bound, including on a remote coordinate.
                .buffer = mesh_buffer->get_reference_buffer(),
            });
        };
        park(built.rs_artifacts.mm_progress_counters_buffer);
        park(built.rs_artifacts.rs_credit_counters_buffer);
        workload.programs.push_back(tt::tt_metal::WorkloadDescriptor::PerCoordProgram{
            .range = ttnn::MeshCoordinateRange(mesh_coordinate),
            .descriptor = std::move(built.descriptor),
        });
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
