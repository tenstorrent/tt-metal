// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// (Names are prefixed to avoid unity-build collisions with the sibling reshard factories.)
constexpr const char* kSWReaderKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp";
constexpr const char* kSWWriterKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_writer.cpp";

// Resource / parameter names referenced by the kernel sources (tensor:: / dfb:: accessors).
constexpr const char* kSWRemoteTensorParam = "remote";
constexpr const char* kSWLocalTensorParam = "local";
constexpr const char* kSWShardDfbName = "shard";
constexpr const char* kSWScratchDfbName = "scratch";

constexpr const char* kSWReaderKernel = "reader";
constexpr const char* kSWWriterKernel = "writer";

// Per-(kernel, node) runtime arguments, collected before vararg padding.
struct SameWidthPerNodeArgs {
    NodeCoord node;
    uint32_t local_offset = 0;   // write_offset (reader source) / read_offset (writer source)
    uint32_t num_transfers = 0;  // num_reads (reader source) / num_writes (writer source)
    std::vector<uint32_t> tail;  // 3 words per transfer: bank_id, offset, units_to_transfer
};

}  // namespace

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameWidthFactory<local_is_output>::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;
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

    // The scratch DFB / unaligned staging path exists only in the reader kernel source: it bulk-reads
    // remote rows at their aligned stride into scratch, then re-strides them into the local buffer.
    // The writer path re-strides row-by-row directly (local source read via its L1 address), so it
    // needs no scratch.
    const bool use_scratch = unaligned && local_is_output;

    bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    auto* remote_buffer = remote_tensor.buffer();
    auto remote_buffer_type = remote_buffer->buffer_type();

    // ------------------------------------------------------------------
    // Per-core runtime argument generation (stateful walk over the remote cores).
    // Mirrors the legacy descriptor factory exactly; only the packing changes:
    //   - the remote base-address RTA (legacy slot 0) is gone (now tensor::remote)
    //   - local_offset / num_transfers become named RTAs
    //   - the per-transfer tail becomes positional varargs
    // The two kernel instances must be generated in this interleaved order, because the walk over
    // the remote cores (remote_core_idx / remote_core_units_rem) is stateful across both.
    // ------------------------------------------------------------------
    std::vector<SameWidthPerNodeArgs> reader_args;
    std::vector<SameWidthPerNodeArgs> writer_args;
    reader_args.reserve(local_cores.size());
    writer_args.reserve(local_cores.size());
    uint32_t max_tail = 0;

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];

    constexpr uint32_t num_kernels = 2;
    uint32_t local_units_left = num_units;
    for (const auto& core : local_cores) {
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, num_kernels);
        uint32_t local_start_offset = 0;
        for (uint32_t kernel_idx = 0; kernel_idx < num_kernels; ++kernel_idx) {
            SameWidthPerNodeArgs per_node;
            per_node.node = core;
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                per_node.local_offset = local_start_offset;
                // Advance by the padded (L1-aligned) stride so the second split kernel writes
                // to the correct row offset in the local buffer. Aligned path is unchanged
                // because local_unit_size_padded == unit_size there.
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
                    per_node.tail.push_back(bank_id);
                    per_node.tail.push_back((remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded);
                    per_node.tail.push_back(units_to_transfer);
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    per_node.num_transfers++;
                }
            }
            max_tail = std::max(max_tail, static_cast<uint32_t>(per_node.tail.size()));
            (kernel_idx == 0 ? reader_args : writer_args).push_back(std::move(per_node));
        }
    }
    // A single per-kernel vararg count covers every node, so each node's tail is zero-padded up to
    // the longest one. The kernel loops are bounded by the named transfer count, so padding words
    // are never read.
    const uint32_t num_varargs = max_tail;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "reshard_same_width";

    const char* offset_arg_name = local_is_output ? "write_offset" : "read_offset";
    const char* count_arg_name = local_is_output ? "num_reads" : "num_writes";
    const char* kernel_path = local_is_output ? kSWReaderKernelPath : kSWWriterKernelPath;

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"interface_with_dram", static_cast<uint32_t>(interface_with_dram)},
        {"unit_size", unit_size},
        {"local_unit_size_padded", local_unit_size_padded},
        {"remote_unit_size_padded", remote_unit_size_padded},
    };

    const auto make_worker = [&](const char* name, DataMovementHardwareConfig hw_config, DFBEndpointType endpoint) {
        KernelSpec kernel{
            .unique_id = KernelSpecName{name},
            .source = std::filesystem::path(kernel_path),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kSWShardDfbName},
                .accessor_name = kSWShardDfbName,
                .endpoint_type = endpoint,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{kSWRemoteTensorParam}, .accessor_name = kSWRemoteTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema = {.runtime_arg_names = {offset_arg_name, count_arg_name}},
            .hw_config = std::move(hw_config),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
        // The unaligned re-striding path is gated by a preprocessor flag rather than a compile-time
        // arg because the scratch DFB binding below is itself conditional: dfb::scratch must not
        // enter name lookup in the aligned build.
        if (unaligned) {
            kernel.compiler_options.defines.emplace("UNALIGNED", "1");
        }
        if (use_scratch) {
            kernel.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{kSWScratchDfbName},
                .accessor_name = kSWScratchDfbName,
                .endpoint_type = endpoint,
            });
        }
        return kernel;
    };

    // The two same-source instances split the local units of each core, so both raw-touch the
    // shard (and scratch) DFB. Two role-free touchers over one grid -> assign 1P + 1C.
    spec.kernels = {
        make_worker(
            kSWReaderKernel, ttnn::create_reader_datamovement_config(device->arch()), DFBEndpointType::PRODUCER),
        make_worker(
            kSWWriterKernel, ttnn::create_writer_datamovement_config(device->arch()), DFBEndpointType::CONSUMER),
    };

    // Local sharded DFB, built on the local buffer's borrowed memory so its backing L1 address is
    // refreshed from the tensor argument on every enqueue. The entry size is the L1-aligned per-row
    // stride, matching the local sharded buffer's layout; when aligned, local_unit_size_padded ==
    // unit_size.
    //
    // The DFB is only an address source (the kernel reaches the shard through get_write_ptr() /
    // get_read_ptr() + offset and never touches the FIFO), so the entry count does not bound its
    // accesses. Clamp it to the backing tensor's packed size: a padded shard shape — or, on the
    // unaligned path, the padded row stride itself — can push the shard-derived size past what
    // Metal 2.0's spec-time borrowed-DFB check allows (it compares against the TensorSpec's packed
    // size, having no Buffer at spec time to see the larger real per-bank allocation the legacy
    // dynamic CB was checked against). Behaviour is unchanged because nothing reads the size.
    const uint32_t shard_dfb_bytes = local_units_per_shard * local_unit_size_padded;
    const uint32_t local_packed_bytes =
        static_cast<uint32_t>(local_tensor.tensor_spec().compute_packed_buffer_size_bytes());
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = DFBSpecName{kSWShardDfbName},
        .entry_size = local_unit_size_padded,
        .num_entries = std::min(shard_dfb_bytes, local_packed_bytes) / local_unit_size_padded,
        .data_format_metadata = data_format,
        .borrowed_from = TensorParamName{kSWLocalTensorParam},
    }};
    if (use_scratch) {
        // Scratch DFB used only by the reader path (local_is_output): the entry size is the
        // remote-aligned stride, so the DFB spans remote rows at their padded stride.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFBSpecName{kSWScratchDfbName},
            .entry_size = remote_unit_size_padded,
            .num_entries = remote_units_per_shard,
            .data_format_metadata = data_format,
        });
    }

    // `local` is a borrowed-only TensorParameter: it backs the shard DFB and is never bound by a
    // kernel. The spec validator counts a borrowed_from reference as a use.
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kSWRemoteTensorParam}, .spec = remote_tensor.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kSWLocalTensorParam}, .spec = local_tensor.tensor_spec()},
    };

    spec.work_units = {WorkUnitSpec{
        .name = "reshard_same_width_work_unit",
        .kernels = {KernelSpecName{kSWReaderKernel}, KernelSpecName{kSWWriterKernel}},
        .target_nodes = all_cores,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    const auto build_kernel_run_args = [&](const char* name, const std::vector<SameWidthPerNodeArgs>& per_node_args) {
        KernelRunArgs run_args{.kernel = KernelSpecName{name}};
        for (const auto& per_node : per_node_args) {
            AddRuntimeArgsForNode(
                run_args.runtime_arg_values,
                per_node.node,
                {{offset_arg_name, per_node.local_offset}, {count_arg_name, per_node.num_transfers}});
            AdvancedKernelRunArgs::Varargs varargs(num_varargs, 0u);
            std::copy(per_node.tail.begin(), per_node.tail.end(), varargs.begin());
            run_args.advanced_options.runtime_varargs.emplace(per_node.node, std::move(varargs));
        }
        return run_args;
    };

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        build_kernel_run_args(kSWReaderKernel, reader_args),
        build_kernel_run_args(kSWWriterKernel, writer_args),
    };
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

}  // namespace ttnn::prim
