// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

template <bool local_is_output>
ttnn::device_operation::ProgramArtifacts ReshardSameWidthFactory<local_is_output>::create_program_spec(
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
    bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    auto* remote_buffer = remote_tensor.buffer();
    auto remote_buffer_type = remote_buffer->buffer_type();

    // Names. The local sharded buffer backs a borrowed-memory DFB; the remote tensor flows through the
    // typed TensorAccessor channel (Case-2 bridge). The unaligned reader path uses a local scratch DFB.
    const TensorParamName kLocal{"local"};
    const TensorParamName kRemote{"remote"};
    const DFBSpecName kShardCb{"shard_cb"};
    const DFBSpecName kScratchCb{"scratch_cb"};
    const KernelSpecName kReader{"reader"};
    const KernelSpecName kWriter{"writer"};

    // Both KernelDescriptors use the SAME source (selected by local_is_output). Only the reader source has
    // the unaligned scratch path. The CT define for the bank type differs by name across the two sources.
    const std::filesystem::path kKernelSource =
        local_is_output ? std::filesystem::path(
                              "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
                              "reshard_same_width_reader_m2.cpp")
                        : std::filesystem::path(
                              "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
                              "reshard_same_width_writer_m2.cpp");
    const std::string kDramCtaName = local_is_output ? "read_from_dram" : "write_to_dram";
    const std::string kNumName = local_is_output ? "num_reads" : "num_writes";
    const std::string kOffsetName = local_is_output ? "write_offset" : "read_offset";
    // The scratch DFB only exists on the (reader) unaligned path.
    const bool use_scratch = local_is_output && unaligned;

    ProgramSpec spec;
    spec.name = "reshard_same_width";

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kLocal, .spec = local_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kRemote, .spec = remote_tensor.tensor_spec()});

    // Local sharded CB bound to the local buffer: borrowed-memory DFB.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = kShardCb,
        .entry_size = unit_size,
        .num_entries = local_units_per_shard,
        .data_format_metadata = data_format,
        .borrowed_from = kLocal,
    });
    if (use_scratch) {
        // Local (non-borrowed) scratch DFB used by the unaligned read path. Base-pointer access only.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = kScratchCb,
            .entry_size = unit_size,
            .num_entries = remote_units_per_shard * remote_unit_size_padded / unit_size,
            .data_format_metadata = data_format,
        });
    }

    // Both DFBs are accessed by BOTH kernels (the two kernel instances run the SAME source selected by
    // local_is_output), so neither can be a self-loop (the local-DFB invariant forbids two same-role
    // bindings — or a multi-kernel self-loop — sharing one WorkUnitSpec). Each DFB is modelled as a
    // single producer/consumer pair across the two kernels: reader=PRODUCER, writer=CONSUMER.
    //  - shard_cb is borrowed memory and a pure address source (both instances access it only by base
    //    pointer; for a borrowed DFB the role is just a placement label, no real FIFO sync, and read
    //    and write pointers resolve to the same borrowed base).
    //  - scratch_cb (unaligned path only) is a single shared local scratch region used by both
    //    instances (each writes then reads it by base pointer) — exactly the legacy single shared c_1
    //    scratch CB. Binding it as one producer/consumer pair keeps that single shared region.
    // For both, behaviour is unchanged.
    auto bind_endpoints = [&](KernelSpec& k, DFBEndpointType endpoint_type) {
        k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = kShardCb, .accessor_name = "shard_cb", .endpoint_type = endpoint_type},
        };
        if (use_scratch) {
            k.dfb_bindings.push_back(
                DFBBinding{.dfb_spec_name = kScratchCb, .accessor_name = "scratch_cb", .endpoint_type = endpoint_type});
        }
    };

    auto make_kernel = [&](const KernelSpecName& unique_id, DataMovementRoleHint role, DFBEndpointType endpoint_type) {
        KernelSpec k;
        k.unique_id = unique_id;
        k.source = kKernelSource;
        k.compile_time_args = {
            {kDramCtaName, static_cast<uint32_t>(interface_with_dram)},
            {"unaligned", static_cast<uint32_t>(unaligned)},
            {"unit_size", unit_size},
            {"local_unit_size_padded", local_unit_size_padded},
            {"remote_unit_size_padded", remote_unit_size_padded},
        };
        if (use_scratch) {
            k.compiler_options.defines = {{"UNALIGNED", "1"}};
        }
        k.runtime_arg_schema.runtime_arg_names = {kOffsetName, kNumName};
        bind_endpoints(k, endpoint_type);
        k.tensor_bindings = {TensorBinding{.tensor_parameter_name = kRemote, .accessor_name = "remote"}};
        k.hw_config = DataMovementHardwareConfig{.role = role};
        return k;
    };

    KernelSpec reader = make_kernel(kReader, DataMovementRoleHint::READER, DFBEndpointType::PRODUCER);
    KernelSpec writer = make_kernel(kWriter, DataMovementRoleHint::WRITER, DFBEndpointType::CONSUMER);

    // ---- Build per-core, per-kernel runtime args (legacy work split, verbatim). ----
    // Legacy packed RTA layout per kernel: [remote_addr(dropped), start_offset, num_transfers, then
    // 3*num_transfers tail]. The remote base address is now supplied by the Case-2 TensorBinding.
    // We collect, per core, the (start_offset, tail) for kernel_0 (reader) and kernel_1 (writer).
    struct PerKernelArgs {
        uint32_t start_offset = 0;
        uint32_t num_transfers = 0;
        std::vector<uint32_t> tail;  // [bank_id, offset, units] * num_transfers
    };
    std::vector<std::array<PerKernelArgs, 2>> per_core_args(local_cores.size());

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    uint32_t local_units_left = num_units;
    uint32_t core_pos = 0;
    for (const auto& core : local_cores) {
        (void)core;
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, 2u);
        uint32_t local_start_offset = 0;
        for (uint32_t kidx = 0; kidx < 2; ++kidx) {
            PerKernelArgs& ka = per_core_args[core_pos][kidx];
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                uint32_t num_transfers = 0;
                ka.start_offset = local_start_offset;
                local_start_offset += local_units_to_transfer * unit_size;
                while (local_units_to_transfer > 0) {
                    if (remote_core_units_rem == 0) {
                        remote_core_idx++;
                        remote_core_units_rem = remote_units_per_shard;
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    auto bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    ka.tail.push_back(static_cast<uint32_t>(bank_id));
                    ka.tail.push_back((remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded);
                    ka.tail.push_back(units_to_transfer);
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    num_transfers++;
                }
                ka.num_transfers = num_transfers;
            }
        }
        ++core_pos;
    }

    // Max vararg tail length across cores (per kernel index).
    uint32_t max_tail = 0;
    for (const auto& pk : per_core_args) {
        max_tail = std::max<uint32_t>(max_tail, static_cast<uint32_t>(pk[0].tail.size()));
        max_tail = std::max<uint32_t>(max_tail, static_cast<uint32_t>(pk[1].tail.size()));
    }
    reader.advanced_options.num_runtime_varargs = max_tail;
    writer.advanced_options.num_runtime_varargs = max_tail;

    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    spec.work_units.push_back(WorkUnitSpec{
        .name = "wu",
        .kernels = {kReader, kWriter},
        .target_nodes = all_cores,
    });

    // ---- Run args. ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run;
    reader_run.kernel = kReader;
    KernelRunArgs writer_run;
    writer_run.kernel = kWriter;
    reader_run.runtime_arg_values.reserve(local_cores.size());
    writer_run.runtime_arg_values.reserve(local_cores.size());

    for (uint32_t i = 0; i < local_cores.size(); ++i) {
        const NodeCoord node{local_cores[i]};
        const PerKernelArgs& r = per_core_args[i][0];
        const PerKernelArgs& w = per_core_args[i][1];

        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node, .args = {{kOffsetName, r.start_offset}, {kNumName, r.num_transfers}}});
        writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node, .args = {{kOffsetName, w.start_offset}, {kNumName, w.num_transfers}}});

        AdvancedKernelRunArgs::Varargs reader_tail = r.tail;
        AdvancedKernelRunArgs::Varargs writer_tail = w.tail;
        reader_tail.resize(max_tail, 0u);
        writer_tail.resize(max_tail, 0u);
        reader_run.advanced_options.runtime_varargs[node] = std::move(reader_tail);
        writer_run.advanced_options.runtime_varargs[node] = std::move(writer_tail);
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    run_args.tensor_args = {
        {kLocal, local_tensor.mesh_tensor()},
        {kRemote, remote_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Explicit template instantiations
template struct ReshardSameWidthFactory<true>;
template struct ReshardSameWidthFactory<false>;

}  // namespace ttnn::prim
