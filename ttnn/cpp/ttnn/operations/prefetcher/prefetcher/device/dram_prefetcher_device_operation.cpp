// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <optional>

namespace ttnn::prim {

DramPrefetcherOperation::program_factory_t DramPrefetcherOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    if (!args.prefetcher_pipes.empty()) {
        return DramPrefetcherPipeSpecFactory{};
    }
    return DramPrefetcherProgramFactory{};
}

namespace {

// The GlobalCircularBuffer target: a worker-sender GCB whose first num_readers senders each feed the same
// number of receivers. Returns that receiver count.
uint32_t validate_global_cb_target(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb, const uint32_t num_readers) {
    TT_FATAL(
        tt::tt_metal::experimental::sender_core_type(global_cb) != tt::tt_metal::experimental::SenderCoreType::Dram,
        "ttnn.dram_prefetcher does not support DRAM-sender GlobalCircularBuffers. Use "
        "ttnn.experimental.start_tensor_prefetcher / ttnn.experimental.stop_tensor_prefetcher instead.");

    // Check that global_cb sender_receiver_core_mapping has same number of receivers for each sender core
    const auto& sender_receiver_core_mapping = global_cb.sender_receiver_core_mapping();
    for (uint32_t i = 0; i < num_readers; ++i) {
        const auto& [sender_core, receiver_core_range] = sender_receiver_core_mapping[i];
        TT_FATAL(
            receiver_core_range.size() == sender_receiver_core_mapping.begin()->second.size(),
            "Global circular buffer must have same number of receivers for each sender core");
    }
    return sender_receiver_core_mapping[0].second.num_cores();
}

// The PrefetcherPipe target: worker-sender pipes, at least one per reader, the ones the readers drive all
// with the same receiver count and ring size. Returns that receiver count.
uint32_t validate_prefetcher_pipes_target(
    const ttnn::PrefetcherPipeList& prefetcher_pipes, const uint32_t num_readers) {
    using tt::tt_metal::experimental::SenderCoreType;
    const auto pipes = ttnn::prefetcher_pipe_refs(prefetcher_pipes);
    for (size_t p = 0; p < pipes.size(); ++p) {
        const tt::tt_metal::experimental::PrefetcherPipe& pipe = pipes[p];
        TT_FATAL(
            pipe.sender_core_type() == SenderCoreType::Worker,
            "ttnn.dram_prefetcher drives worker-sender PrefetcherPipes, but pipe {} (sender {}) has a {} sender. "
            "DRAM-sender pipes belong to the Tensor prefetcher: use ttnn.experimental.start_tensor_prefetcher / "
            "ttnn.experimental.queue_tensor_prefetcher_request instead.",
            p,
            pipe.sender_core().str(),
            pipe.sender_core_type());
    }
    TT_FATAL(
        pipes.size() >= num_readers,
        "ttnn.dram_prefetcher needs one PrefetcherPipe per reader core: the weights are sharded over {} DRAM banks, "
        "so {} reader cores, but only {} prefetcher_pipes were given",
        num_readers,
        num_readers,
        pipes.size());

    const auto reader_pipes = dram_prefetcher_reader_pipes(prefetcher_pipes, num_readers);
    const tt::tt_metal::experimental::PrefetcherPipe& first_pipe = *reader_pipes.front();
    for (uint32_t i = 0; i < num_readers; ++i) {
        const tt::tt_metal::experimental::PrefetcherPipe& pipe = *reader_pipes[i];
        TT_FATAL(
            pipe.receiver_cores().num_cores() == first_pipe.receiver_cores().num_cores(),
            "ttnn.dram_prefetcher needs the same number of receivers on every reader's PrefetcherPipe, but reader {} "
            "(sender {}) has {} and reader 0 (sender {}) has {}",
            i,
            pipe.sender_core().str(),
            pipe.receiver_cores().num_cores(),
            first_pipe.sender_core().str(),
            first_pipe.receiver_cores().num_cores());
        TT_FATAL(
            pipe.ring_size() == first_pipe.ring_size(),
            "ttnn.dram_prefetcher needs one ring size across its PrefetcherPipes, but reader {} (sender {}) has {} B "
            "and reader 0 (sender {}) has {} B",
            i,
            pipe.sender_core().str(),
            pipe.ring_size(),
            first_pipe.sender_core().str(),
            first_pipe.ring_size());
    }
    return first_pipe.receiver_cores().num_cores();
}

}  // namespace

void DramPrefetcherOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "Must have at least one input tensor");
    TT_FATAL(args.num_layers > 0, "Prefetcher must run for at least 1 layer");
    TT_FATAL(
        args.global_cb.has_value() != !args.prefetcher_pipes.empty(),
        "ttnn.dram_prefetcher needs exactly one delivery target, global_cb or prefetcher_pipes, but got "
        "global_cb={} and {} prefetcher_pipes",
        args.global_cb.has_value() ? "set" : "None",
        args.prefetcher_pipes.size());

    const ttnn::Tensor& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs

    uint32_t num_readers = input_tensors[0].shard_spec()->grid.num_cores();
    const uint32_t num_receivers_per_sender =
        args.global_cb.has_value() ? validate_global_cb_target(*args.global_cb, num_readers)
                                   : validate_prefetcher_pipes_target(args.prefetcher_pipes, num_readers);

    TT_FATAL(num_readers > 0, "Number of reader cores must be greater than zero");
    TT_FATAL(num_receivers_per_sender > 0, "Number of receiver cores per sender must be greater than zero");

    for (size_t i = 0; i < input_tensors.size() - 1; ++i) {
        const auto& tensor = input_tensors[i];
        // Check that all tensors are on the same device
        TT_FATAL(tensor.device() == input_tensors[0].device(), "All tensors must be on the same device");
        TT_FATAL(tensor.layout() == Layout::TILE, "All tensors must be tilized");
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensors must be width sharded");
        TT_FATAL(tensor.memory_config().buffer_type() == BufferType::DRAM, "Input tensors must be in DRAM");

        // Check that all tensors' N (per shard) is divisible by number of cores in global CB receiver
        TT_FATAL(
            tensor.buffer()->shard_spec().shape()[1] % num_receivers_per_sender == 0,
            "All tensors' padded shard size (in last dim) {} must be divisible by the number of receiver cores per "
            "sender {}.",
            tensor.buffer()->shard_spec().shape()[1],
            num_receivers_per_sender);

        tt::DataFormat tensor_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        TT_FATAL(
            tensor_data_format == tt::DataFormat::Bfp4_b || tensor_data_format == tt::DataFormat::Bfp8_b ||
                tensor_data_format == tt::DataFormat::Float16_b,
            "Input tensors must be of type Bfp4_b, Bfp8_b, or Float16_b");
    }

    if (!args.prefetcher_pipes.empty()) {
        // Each receiver is delivered one block per pipe entry, so a ring must hold one block of every tensor.
        // It need not hold a whole number of them: the pipe skips the gap at the wrap.
        const uint32_t ring_size =
            dram_prefetcher_reader_pipes(args.prefetcher_pipes, num_readers).front()->ring_size();
        const std::vector<Tensor> weight_tensors(input_tensors.begin(), input_tensors.end() - 1);
        const auto geometry = compute_dram_prefetcher_geometry(weight_tensors, num_receivers_per_sender);
        for (size_t t = 0; t < geometry.tensors.size(); ++t) {
            TT_FATAL(
                geometry.tensors[t].block_size_per_receiver <= ring_size,
                "ttnn.dram_prefetcher delivers each receiver one block of tensor {} per PrefetcherPipe entry, {} B "
                "({} tiles of {} B over {} receivers), but the pipes' ring is only {} B. Use a larger ring or more "
                "receivers per pipe.",
                t,
                geometry.tensors[t].block_size_per_receiver,
                geometry.tensors[t].block_num_tiles,
                geometry.tensors[t].tile_size,
                num_receivers_per_sender,
                ring_size);
        }
    }

    TT_FATAL(
        tensor_addrs.device() == input_tensors[0].device(),
        "tensors_addrs must be on the same device as the input tensors");
    TT_FATAL(tensor_addrs.layout() == Layout::ROW_MAJOR, "Tensor containing addresses must be row major");
    TT_FATAL(
        tensor_addrs.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Tensor containing addresses must be height sharded");
    TT_FATAL(tensor_addrs.memory_config().buffer_type() == BufferType::L1, "Tensor containing addresses must be in L1");

    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.dtype());
    TT_FATAL(tensor_addrs_data_format == tt::DataFormat::UInt32, "Tensor containing addresses must be of type UInt32");
}

tt::tt_metal::TensorSpec DramPrefetcherOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{32, 32},
        tt::tt_metal::TensorLayout(
            tensor_args.input_tensors[0].dtype(),
            tt::tt_metal::PageConfig(tensor_args.input_tensors[0].layout()),
            MemoryConfig{}));
}

DramPrefetcherOperation::tensor_return_value_t DramPrefetcherOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensors[0].device());
}

ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    const uint32_t num_layers,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const bool enable_performance_mode,
    const ttnn::PrefetcherPipeList& prefetcher_pipes) {
    auto operation_attributes = DramPrefetcherParams{
        .num_layers = num_layers,
        .enable_performance_mode = enable_performance_mode,
        .global_cb = global_cb,
        .prefetcher_pipes = prefetcher_pipes,
    };
    auto tensor_args = DramPrefetcherInputs{.input_tensors = tensors};

    return ttnn::device_operation::launch<DramPrefetcherOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
