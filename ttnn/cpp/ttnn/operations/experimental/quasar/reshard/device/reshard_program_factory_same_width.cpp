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
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

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
                local_start_offset += local_units_to_transfer * local_unit_size_padded;
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

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"interface_with_dram", static_cast<uint32_t>(interface_with_dram)},
        {"unit_size", unit_size},
        {"local_unit_size_padded", local_unit_size_padded},
        {"remote_unit_size_padded", remote_unit_size_padded},
    };

    auto make_cta = [&](uint32_t is_reader) {
        KernelSpec::CompileTimeArgs cta = compile_time_args;
        if (use_scratch) {
            cta.emplace("is_reader", is_reader);
            cta.emplace("remote_units_per_shard", remote_units_per_shard);
        }
        return cta;
    };

    const auto make_worker = [&](const char* name,
                                 const DataMovementHardwareConfig& hw_config,
                                 DFBEndpointType endpoint,
                                 uint32_t is_reader) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_path),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kSWShardDfbName},
                .accessor_name = kSWShardDfbName,
                .endpoint_type = endpoint,
            }},
            .tensor_bindings =
                {TensorBinding{
                     .tensor_parameter_name = TensorParamName{kSWRemoteTensorParam},
                     .accessor_name = kSWRemoteTensorParam},
                 TensorBinding{
                     .tensor_parameter_name = TensorParamName{kSWLocalTensorParam},
                     .accessor_name = kSWLocalTensorParam}},
            .compile_time_args = make_cta(is_reader),
            .runtime_arg_schema = {.runtime_arg_names = {off_name, count_name}},
            .hw_config = hw_config,
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
        if (unaligned) {
            k.compiler_options.defines.emplace("UNALIGNED", "1");
        }
        if (use_scratch) {
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{kSWScratchDfbName},
                .accessor_name = kSWScratchDfbName,
                .endpoint_type = endpoint,
            });
        }
        return k;
    };

    KernelSpec k0 = make_worker(
        "reader",
        ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
        DFBEndpointType::PRODUCER,
        /*is_reader=*/1);
    KernelSpec k1 = make_worker(
        "writer",
        ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
        DFBEndpointType::CONSUMER,
        /*is_reader=*/0);

    // Borrowed DFB size is checked against TensorSpec packed bytes (no Buffer at spec time), so
    // clamp advertised bytes rather than the full padded shard. The DFB is only an address source.
    // If packed < one padded row, shrink entry_size so num_entries stays >= 1 (ProgramSpec rejects 0).
    const uint32_t shard_dfb_bytes = local_units_per_shard * local_unit_size_padded;
    const uint32_t local_packed_bytes =
        static_cast<uint32_t>(local_tensor.tensor_spec().compute_packed_buffer_size_bytes());
    uint32_t shard_dfb_entry_size = local_unit_size_padded;
    uint32_t shard_dfb_num_entries = std::min(shard_dfb_bytes, local_packed_bytes) / shard_dfb_entry_size;
    if (shard_dfb_num_entries == 0 && local_packed_bytes > 0) {
        shard_dfb_entry_size = local_packed_bytes;
        shard_dfb_num_entries = 1;
    }
    DataflowBufferSpec shard_dfb{
        .unique_id = DFBSpecName{kSWShardDfbName},
        .entry_size = shard_dfb_entry_size,
        .num_entries = shard_dfb_num_entries,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kSWLocalTensorParam},
    };

    spec.kernels = {k0, k1};
    if (use_scratch) {
        DataflowBufferSpec scratch_dfb{
            .unique_id = DFBSpecName{kSWScratchDfbName},
            .entry_size = remote_unit_size_padded,
            .num_entries = 2 * remote_units_per_shard,
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
        KernelRunArgs::RuntimeArgValues& run_args_rtas = run_args.runtime_arg_values;
        for (const auto& pa : per_node) {
            AddRuntimeArgsForNode(
                run_args_rtas,
                pa.node,
                {
                    {off_name, pa.local_offset},
                    {count_name, pa.num_transfers},
                });
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
