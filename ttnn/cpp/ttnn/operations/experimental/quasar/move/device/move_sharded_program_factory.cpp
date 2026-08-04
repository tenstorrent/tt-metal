// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation_types.hpp"
#include "move_sharded_program_factory.hpp"

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Metal 2.0 resource names (ProgramSpec scope).
const m2::KernelSpecName READER{"reader"};
const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName OUTPUT{"output"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts MoveShardedProgramFactory::create_program_artifacts(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    const auto shard_spec = input.shard_spec().value();
    const auto shard_shape = shard_spec.shape;
    const auto shard_grid = shard_spec.grid;
    const auto& input_shape = input.logical_shape();
    const DataType input_dtype = input.dtype();
    const Layout input_layout = input.layout();
    TT_FATAL(
        input_layout == output.layout() && input_dtype == output.dtype() &&
            shard_shape == output.shard_spec().value().shape && input_shape == output.logical_shape(),
        "Error");

    // total_size_bytes is derived from the tensor spec (aligned size per bank), not from any
    // storage address, so it is safe to send to the kernel as a runtime arg.
    const uint32_t total_size_bytes = input.buffer()->aligned_size_per_bank();

    TT_FATAL(
        input.buffer()->alignment() == output.buffer()->alignment(),
        "Expected input buffer alignment ({} B) and output buffer alignment ({} B) to be equal",
        input.buffer()->alignment(),
        output.buffer()->alignment());

    //
    // -------- Build the ProgramSpec --------
    //
    // NOTE on the cache-hit hazard this port resolves:
    // The legacy factory computed move_chunk_size_bytes = (output_addr - input_addr) on the host
    // and emitted it as a plain runtime-arg scalar, relying on the slow cache-hit path re-running
    // create_descriptor() to recompute that delta from freshly-allocated buffer addresses.  The
    // Metal 2.0 factory concept does NOT re-run the factory on a cache hit (it only refreshes
    // tensor bindings), so a same-spec/different-storage hit would have read a STALE delta and
    // produced silently-wrong numerics.  We eliminate the host-computed delta entirely: the kernel
    // reads the input/output resident-shard base addresses from local TensorAccessors (over the
    // INPUT/OUTPUT tensor bindings, refreshed on cache hit) and recomputes the chunk size in-kernel
    // as (dst_base - src_base).  Only total_size_bytes (spec-derived, storage-independent) is
    // delivered as a runtime arg.

    m2::ProgramSpec spec;
    spec.name = "move_sharded";

    // Tensor parameters (src / dst).  Their addresses reach the kernel through the typed binding
    // channel (refreshed on cache hit); the kernel recovers each resident shard's local L1 base via
    // a TensorAccessor over these (no borrowed self-loop DFBs, which Metal 2.0 forbids on DM kernels).
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // Preserve the legacy processor/NOC selection (RISCV_1 / NOC_1) via an explicit Gen1Config.
    m2::DataMovementHardwareConfig reader_hw;
    if (input.device()->arch() == tt::ARCH::QUASAR) {
        reader_hw = m2::DataMovementGen2Config{};
    } else {
        reader_hw = m2::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    }
    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/"
            "reader_unary_local_l1_copy_backwards.cpp",
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .hw_config = std::move(reader_hw),
    };

    reader.runtime_arg_schema.runtime_arg_names = m2::Group<std::string>{"total_size_bytes"};

    spec.kernels.push_back(reader);

    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "wu",
        .kernels = {READER},
        .target_nodes = shard_grid,
    });

    //
    // -------- Build the ProgramRunArgs --------
    //

    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs reader_run_args;
    reader_run_args.kernel = READER;

    const auto cores = corerange_to_cores(shard_grid, std::nullopt, true);
    for (const auto& core : cores) {
        reader_run_args.runtime_arg_values["total_size_bytes"][core] = total_size_bytes;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));

    run_args.tensor_args.emplace(INPUT, input_mt);
    run_args.tensor_args.emplace(OUTPUT, output_mt);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
