// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_pipe_spec_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "dram_prefetcher_program_factory.hpp"

namespace ttnn::prim {

using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::PrefetcherPipeArgument;
using tt::tt_metal::experimental::PrefetcherPipeParameter;
using tt::tt_metal::experimental::PrefetcherPipeParamName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

ttnn::PrefetcherPipeList dram_prefetcher_reader_pipes(
    const ttnn::PrefetcherPipeList& prefetcher_pipes, const uint32_t num_readers) {
    std::vector<CoreRange> sender_ranges;
    sender_ranges.reserve(prefetcher_pipes.size());
    for (const auto& pipe : ttnn::prefetcher_pipe_refs(prefetcher_pipes)) {
        sender_ranges.emplace_back(pipe.get().sender_core());
    }
    const std::vector<CoreCoord> senders =
        corerange_to_cores(CoreRangeSet(std::move(sender_ranges)), std::nullopt, /*row_wise=*/true);
    TT_FATAL(
        senders.size() >= num_readers,
        "dram_prefetcher needs a PrefetcherPipe with its own sender core for each of its {} readers, but the {} "
        "prefetcher_pipes have only {} distinct sender cores",
        num_readers,
        prefetcher_pipes.size(),
        senders.size());

    ttnn::PrefetcherPipeList reader_pipes;
    reader_pipes.reserve(num_readers);
    for (uint32_t i = 0; i < num_readers; ++i) {
        const auto it = std::find_if(prefetcher_pipes.begin(), prefetcher_pipes.end(), [&](const auto& pipe) {
            return pipe->sender_core() == senders[i];
        });
        reader_pipes.push_back(*it);
    }
    return reader_pipes;
}

ttnn::device_operation::ProgramArtifacts DramPrefetcherPipeSpecFactory::create_program_artifacts(
    const DramPrefetcherParams& operation_attributes,
    const DramPrefetcherInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    const Tensor& tensor_addrs = input_tensors.back();
    const std::vector<Tensor> weight_tensors(input_tensors.begin(), input_tensors.end() - 1);
    const uint32_t num_tensors = weight_tensors.size();
    const uint32_t num_layers = operation_attributes.num_layers;

    const uint32_t num_readers = weight_tensors[0].shard_spec()->grid.num_cores();
    const ttnn::PrefetcherPipeList reader_pipes =
        dram_prefetcher_reader_pipes(operation_attributes.prefetcher_pipes, num_readers);
    const uint32_t num_receivers_per_reader = reader_pipes.front()->receiver_cores().num_cores();
    const DramPrefetcherGeometry geometry = compute_dram_prefetcher_geometry(weight_tensors, num_receivers_per_reader);

    std::vector<CoreCoord> reader_cores;
    std::vector<CoreRange> reader_ranges;
    reader_cores.reserve(num_readers);
    reader_ranges.reserve(num_readers);
    for (const auto& pipe : reader_pipes) {
        reader_cores.push_back(pipe->sender_core());
        reader_ranges.emplace_back(pipe->sender_core());
    }
    const CoreRangeSet reader_core_range(std::move(reader_ranges));

    const DFBSpecName STAGING_DFB{"staging"};
    const DFBSpecName ADDRS_DFB{"addrs"};
    const DFBSpecName SYNC_DFB{"sync"};
    const TensorParamName TENSOR_ADDRS{"tensor_addrs"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    // The reader's staging ring. Program-local, unlike the GlobalCircularBuffer path's (which lies on the
    // GCB's sender L1): a DFB can only borrow a tensor's memory, not a pipe's ring.
    const uint32_t staging_num_entries = kDramPrefetcherStagingBlocks * geometry.max_block_num_tiles;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = STAGING_DFB,
            .entry_size = geometry.max_tile_size,
            .num_entries = staging_num_entries,
        },
        // The weight addresses, read in place from the address tensor's shard on each reader.
        DataflowBufferSpec{
            .unique_id = ADDRS_DFB,
            .entry_size = tensor_addrs.buffer()->aligned_page_size(),
            .num_entries = 1,
            .borrowed_from = TENSOR_ADDRS,
        },
        // The writer's exit signal to the reader (see the kernels comment below). Only the push carries
        // meaning, so one entry of the smallest size.
        DataflowBufferSpec{
            .unique_id = SYNC_DFB,
            .entry_size = kDramPrefetcherSyncEntryBytes,
            .num_entries = 1,
        },
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      PrefetcherPipes
    ////////////////////////////////////////////////////////////////////////////
    // One parameter per reader, all behind the writer's one accessor: each reader core hosts exactly one
    // pipe's sender. The entry size is the first tensor's per-receiver block; the writer re-sizes the
    // pipe for each later tensor.
    Group<PrefetcherPipeParameter> prefetcher_pipe_parameters;
    Group<PrefetcherPipeParamName> prefetcher_pipe_names;
    for (uint32_t i = 0; i < num_readers; ++i) {
        const PrefetcherPipeParamName name{fmt::format("out_pipe_{}", i)};
        prefetcher_pipe_names.push_back(name);
        prefetcher_pipe_parameters.push_back(PrefetcherPipeParameter{
            .unique_id = name,
            .receivers = reader_pipes[i]->receiver_cores(),
            .ring_size = reader_pipes[i]->ring_size(),
            .entry_size = geometry.tensors.front().block_size_per_receiver,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    // Both kernels use NOC 0, as on the GlobalCircularBuffer path, in dynamic-NOC mode (a ProgramSpec
    // rejects two dedicated-NOC kernels pinned to one NOC). In dynamic-NOC mode the two RISCs share one
    // set of per-core NoC counters, which the reader re-syncs from the hardware at exit because
    // performance mode skips the per-read counter updates. That re-sync must not race the writer's
    // outstanding traffic, so the writer releases the reader through the sync DFB once its pipe is quiet.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STAGING_DFB,
                    .accessor_name = "staging",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                // Only the reader touches the addresses, so it holds both endpoints.
                DFBBinding{
                    .dfb_spec_name = ADDRS_DFB, .accessor_name = "addrs", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = ADDRS_DFB, .accessor_name = "addrs", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SYNC_DFB, .accessor_name = "sync", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .compile_time_args =
            {
                {"num_layers", num_layers},
                {"num_tensors", num_tensors},
                {"num_blocks", geometry.num_blocks},
                {"staging_size_bytes", kDramPrefetcherStagingBlocks * geometry.max_block_size},
                {"num_staging_blocks", kDramPrefetcherStagingBlocks},
                {"max_block_num_tiles", geometry.max_block_num_tiles},
                {"max_block_size", geometry.max_block_size},
                {"skip_ptr_update", static_cast<uint32_t>(operation_attributes.enable_performance_mode)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"bank_id", "vc"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::NOC_0,
                        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                    },
            },
    };
    // Per tensor: DRAM page size, then pages per block.
    reader.advanced_options.num_common_runtime_varargs = 2 * num_tensors;

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_pipe_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STAGING_DFB,
                    .accessor_name = "staging",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SYNC_DFB, .accessor_name = "sync", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {
                {"num_layers", num_layers},
                {"num_tensors", num_tensors},
                {"num_blocks", geometry.num_blocks},
                {"max_block_num_tiles", geometry.max_block_num_tiles},
            },
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::NOC_0,
                        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
                    },
            },
    };
    // Per tensor: pipe entry size, block height in tile rows, then one receiver's row as writes of
    // coalesced_page_size x coalesced_num_pages.
    writer.advanced_options.num_common_runtime_varargs = 4 * num_tensors;
    // The writer runs only on the reader cores, which host the pipes' senders and none of their receivers,
    // so it is every pipe's sender.
    writer.advanced_options.prefetcher_pipe_bindings = {
        {.pipe_parameter_names = prefetcher_pipe_names, .accessor_name = "out"}};

    ////////////////////////////////////////////////////////////////////////////
    //                      Run args
    ////////////////////////////////////////////////////////////////////////////
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0; i < num_readers; ++i) {
        // Reader i reads DRAM bank i.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            reader_cores[i],
            {{"bank_id", i}, {"vc", dram_prefetcher_reader_vc(reader_cores, i)}});
    }
    auto& reader_varargs = reader_run_args.advanced_options.common_runtime_varargs;
    auto& writer_varargs = writer_run_args.advanced_options.common_runtime_varargs;
    for (const auto& t : geometry.tensors) {
        reader_varargs.push_back(t.page_size);
    }
    for (const auto& t : geometry.tensors) {
        reader_varargs.push_back(t.block_num_pages);
    }
    for (const auto& t : geometry.tensors) {
        writer_varargs.push_back(t.block_size_per_receiver);
    }
    for (const auto& t : geometry.tensors) {
        writer_varargs.push_back(t.block_height_in_tiles);
    }
    for (const auto& t : geometry.tensors) {
        writer_varargs.push_back(t.coalesced_page_size);
    }
    for (const auto& t : geometry.tensors) {
        writer_varargs.push_back(t.coalesced_num_pages);
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    // The weights are reached through the addresses the address tensor holds, as on the GlobalCircularBuffer
    // path, so the address tensor is the one tensor bound.
    run_args.tensor_args = {{TENSOR_ADDRS, tensor_addrs.mesh_tensor()}};
    for (uint32_t i = 0; i < num_readers; ++i) {
        run_args.advanced_options.prefetcher_pipe_args.emplace(
            prefetcher_pipe_names[i], PrefetcherPipeArgument{*reader_pipes[i]});
    }

    ProgramSpec spec{
        .name = "dram_prefetcher_prefetcher_pipe",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {TensorParameter{.unique_id = TENSOR_ADDRS, .spec = tensor_addrs.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "readers", .kernels = {READER, WRITER}, .target_nodes = reader_core_range}},
        .advanced_options = {.prefetcher_pipe_parameters = std::move(prefetcher_pipe_parameters)},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
