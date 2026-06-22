// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshard/device/reshard_program_factory_same_width.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// (Names are prefixed to avoid Unity-build collisions with the sibling reshard factories.)
constexpr const char* kSWReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/dataflow/reshard_same_width_reader.cpp";
constexpr const char* kSWWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/dataflow/reshard_same_width_writer.cpp";

// Resource / parameter names referenced by the kernel sources (tensor::/dfb:: accessors).
constexpr const char* kSWRemoteTensorParam = "remote";
constexpr const char* kSWLocalTensorParam = "local_shard";
constexpr const char* kSWShardDfbName = "shard_cb";
constexpr const char* kSWScratchDfbName = "scratch_cb";

// Per-(kernel, node) collected runtime arguments before vararg padding.
struct SameWidthPerNodeArgs {
    NodeCoord node;
    uint32_t local_offset = 0;   // write_offset (reader) / read_offset (writer)
    uint32_t num_transfers = 0;  // num_reads (reader) / num_writes (writer)
    std::vector<uint32_t> tail;  // 3 per transfer: bank_id, offset, units_to_transfer
};

}  // namespace

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameWidthFactory<local_is_output>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;
    const auto& local_tensor = local_is_output ? output : input;
    const auto& remote_tensor = local_is_output ? input : output;

    auto* device = input.device();

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();

    auto remote_core_type = remote_tensor.buffer()->core_type();
    auto local_cores = get_optimal_worker_cores_for_sharded_tensor(local_tensor);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(local_cores));
    auto remote_cores = remote_tensor.buffer()->buffer_distribution_spec().value().cores_with_data();

    uint32_t unit_size = 0;
    uint32_t local_units_per_shard = 0;
    uint32_t remote_units_per_shard = 0;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());

    uint32_t num_units = local_tensor.buffer()->num_pages();
    if (local_tensor.layout() == Layout::TILE) {
        unit_size = tt::tile_size(data_format);
        local_units_per_shard = local_shard_spec.numel() / TILE_HW;
        remote_units_per_shard = remote_shard_spec.numel() / TILE_HW;
    } else {
        unit_size = static_cast<uint32_t>(local_shard_spec.shape[1] * local_tensor.element_size());
        local_units_per_shard = local_shard_spec.shape[0];
        remote_units_per_shard = remote_shard_spec.shape[0];
    }
    uint32_t local_unit_size_padded = tt::align(unit_size, local_tensor.buffer()->alignment());
    uint32_t remote_unit_size_padded = tt::align(unit_size, remote_tensor.buffer()->alignment());
    bool unaligned = false;
    if (remote_unit_size_padded != unit_size || local_unit_size_padded != unit_size) {
        unaligned = true;
    }
    const uint32_t total_size = local_units_per_shard * unit_size;

    // The scratch DFB / unaligned staging path only exists in the reader kernel source.
    const bool use_scratch = local_is_output && unaligned;

    const bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    auto* remote_buffer = remote_tensor.buffer();
    auto remote_buffer_type = remote_buffer->buffer_type();

    // ------------------------------------------------------------------
    // Per-core runtime argument generation (stateful walk over remote cores).
    // Mirrors the legacy ProgramDescriptor factory exactly; only the packing changes:
    //   - remote base addr RTA (legacy arg 0) is dropped (now tensor::remote)
    //   - local_offset/num_transfers become named RTAs
    //   - the per-transfer tail becomes positional varargs
    // ------------------------------------------------------------------
    std::vector<SameWidthPerNodeArgs> k0_args;  // reader endpoint
    std::vector<SameWidthPerNodeArgs> k1_args;  // writer endpoint
    k0_args.reserve(local_cores.size());
    k1_args.reserve(local_cores.size());
    uint32_t max_tail = 0;

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];

    uint32_t local_units_left = num_units;
    for (const auto& core : local_cores) {
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, 2u);
        uint32_t local_start_offset = 0;
        for (uint32_t ki = 0; ki < 2; ++ki) {
            SameWidthPerNodeArgs pa;
            pa.node = NodeCoord{core.x, core.y};
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                pa.local_offset = local_start_offset;
                local_start_offset += local_units_to_transfer * unit_size;
                while (local_units_to_transfer > 0) {
                    if (remote_core_units_rem == 0) {
                        remote_core_idx++;
                        remote_core_units_rem = remote_units_per_shard;
                        bank_id = device->allocator()->get_bank_ids_from_logical_core(
                            remote_buffer_type, remote_cores[remote_core_idx])[0];
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    pa.tail.push_back(bank_id);
                    pa.tail.push_back((remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded);
                    pa.tail.push_back(units_to_transfer);
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    pa.num_transfers++;
                }
            }
            max_tail = std::max(max_tail, static_cast<uint32_t>(pa.tail.size()));
            (ki == 0 ? k0_args : k1_args).push_back(std::move(pa));
        }
    }
    const uint32_t num_varargs = max_tail;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "reshard_same_width";

    const char* off_name = local_is_output ? "write_offset" : "read_offset";
    const char* count_name = local_is_output ? "num_reads" : "num_writes";
    const char* kernel_path = local_is_output ? kSWReaderKernelPath : kSWWriterKernelPath;

    KernelSpec::CompileTimeArgs compile_time_args = {
        {"interface_with_dram", static_cast<uint32_t>(interface_with_dram)},
        {"unit_size", unit_size},
    };
    if constexpr (local_is_output) {
        compile_time_args.emplace("remote_unit_size_padded", remote_unit_size_padded);
    }

    const auto make_worker = [&](const char* name, DataMovementRoleHint role, DFBEndpointType endpoint) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_path),
            .hw_config = DataMovementHardwareConfig{.role = role},
        };
        k.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = TensorParamName{kSWRemoteTensorParam}, .accessor_name = kSWRemoteTensorParam});
        k.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = TensorParamName{kSWLocalTensorParam}, .accessor_name = kSWLocalTensorParam});
        k.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFBSpecName{kSWShardDfbName},
            .accessor_name = kSWShardDfbName,
            .endpoint_type = endpoint,
        });
        if (use_scratch) {
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{kSWScratchDfbName},
                .accessor_name = kSWScratchDfbName,
                .endpoint_type = endpoint,
            });
        }
        k.compile_time_args = compile_time_args;
        if (use_scratch) {
            k.compiler_options.defines.emplace("UNALIGNED", "1");
        }
        k.runtime_arg_schema.runtime_arg_names = {off_name, count_name};
        k.advanced_options.num_runtime_varargs = num_varargs;
        return k;
    };

    KernelSpec k0 = make_worker("reader", DataMovementRoleHint::READER, DFBEndpointType::PRODUCER);
    KernelSpec k1 = make_worker("writer", DataMovementRoleHint::WRITER, DFBEndpointType::CONSUMER);

    DataflowBufferSpec shard_dfb{
        .unique_id = DFBSpecName{kSWShardDfbName},
        .entry_size = unit_size,
        .num_entries = local_units_per_shard,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kSWLocalTensorParam},
    };
    (void)total_size;  // == entry_size * num_entries; kept for parity with the legacy CB total_size.

    spec.kernels = {k0, k1};
    if (use_scratch) {
        DataflowBufferSpec scratch_dfb{
            .unique_id = DFBSpecName{kSWScratchDfbName},
            .entry_size = remote_unit_size_padded,
            .num_entries = remote_units_per_shard,
            .data_format_metadata = data_format,
        };
        spec.dataflow_buffers = {shard_dfb, scratch_dfb};
    } else {
        spec.dataflow_buffers = {shard_dfb};
    }
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kSWRemoteTensorParam}, .spec = remote_tensor.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kSWLocalTensorParam}, .spec = local_tensor.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "reshard_same_width_work_unit",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}},
        .target_nodes = all_cores,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    const auto build_kernel_run_args = [&](const char* name, const std::vector<SameWidthPerNodeArgs>& per_node) {
        KernelRunArgs run_args{.kernel = KernelSpecName{name}};
        for (const auto& pa : per_node) {
            run_args.runtime_arg_values.push_back(
                {pa.node, {{off_name, pa.local_offset}, {count_name, pa.num_transfers}}});
            AdvancedKernelRunArgs::Varargs varargs(num_varargs, 0u);
            std::copy(pa.tail.begin(), pa.tail.end(), varargs.begin());
            run_args.advanced_options.runtime_varargs.emplace(pa.node, std::move(varargs));
        }
        return run_args;
    };

    KernelRunArgs k0_run_args = build_kernel_run_args("reader", k0_args);
    KernelRunArgs k1_run_args = build_kernel_run_args("writer", k1_args);

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(k0_run_args), std::move(k1_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kSWRemoteTensorParam}, TensorArgument{remote_tensor.mesh_tensor()}},
        {TensorParamName{kSWLocalTensorParam}, TensorArgument{local_tensor.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

// Explicit template instantiations
template struct ReshardSameWidthFactory<true>;
template struct ReshardSameWidthFactory<false>;

}  // namespace ttnn::prim::qsr
