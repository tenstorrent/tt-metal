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

// A target built for the Tensor prefetcher (DRAM senders) is redirected there before anything else is checked.
void validate_worker_senders(const DramPrefetcherParams& args) {
    using tt::tt_metal::experimental::SenderCoreType;
    if (args.global_cb.has_value()) {
        TT_FATAL(
            tt::tt_metal::experimental::sender_core_type(*args.global_cb) != SenderCoreType::Dram,
            "ttnn.dram_prefetcher does not support DRAM-sender GlobalCircularBuffers. Use "
            "ttnn.experimental.start_tensor_prefetcher / ttnn.experimental.stop_tensor_prefetcher instead.");
        return;
    }
    const auto pipes = ttnn::prefetcher_pipe_refs(args.prefetcher_pipes);
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
}

// What the target must hold. The GlobalCircularBuffer holds every receiver's share of the largest tensor, and
// on its senders the reader's staging. A PrefetcherPipe ring holds one entry, one receiver's slice of one block,
// of every tensor.
void validate_target_capacity(
    const DramPrefetcherParams& args, const DramPrefetcherGeometry& geometry, const uint32_t num_receivers_per_reader) {
    if (args.global_cb.has_value()) {
        const uint32_t global_cb_size = args.global_cb->size();
        const uint32_t max_tensor_size = geometry.max_block_size / num_receivers_per_reader * geometry.num_blocks;
        TT_FATAL(
            max_tensor_size <= global_cb_size,
            "largest tensor {} must fit in global cb {}",
            max_tensor_size,
            global_cb_size);
        const uint32_t reader_cb_size = geometry.max_block_size * kDramPrefetcherStagingBlocks;
        TT_FATAL(
            reader_cb_size <= global_cb_size,
            "reader_cb_size {} must not be larger than global cb {}",
            reader_cb_size,
            global_cb_size);
        return;
    }
    // The ring need not hold a whole number of entries: the pipe skips the gap at the wrap. The readers' pipes
    // share the writer's one accessor, which ProgramSpec holds to one ring size.
    const uint32_t ring_size = args.prefetcher_pipes.front()->ring_size();
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
            num_receivers_per_reader,
            ring_size);
    }
}

}  // namespace

void DramPrefetcherOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "Must have at least one input tensor");
    TT_FATAL(args.num_layers > 0, "Prefetcher must run for at least 1 layer");
    const bool has_global_cb = args.global_cb.has_value();
    const bool has_pipes = !args.prefetcher_pipes.empty();
    TT_FATAL(
        has_global_cb != has_pipes,
        "ttnn.dram_prefetcher needs exactly one delivery target, global_cb or prefetcher_pipes, but got "
        "global_cb={} and {} prefetcher_pipes",
        has_global_cb ? "set" : "None",
        args.prefetcher_pipes.size());
    validate_worker_senders(args);

    const ttnn::Tensor& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs

    uint32_t num_readers = input_tensors[0].shard_spec()->grid.num_cores();
    TT_FATAL(num_readers > 0, "Number of reader cores must be greater than zero");
    const DramPrefetcherReaders readers = dram_prefetcher_readers(args, num_readers);
    const uint32_t num_receivers_per_sender = readers.num_receivers_per_reader;
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

        // Each receiver is sent the same whole number of tile columns of every block row.
        const uint32_t shard_width_in_tiles =
            tensor.buffer()->shard_spec().shape()[1] / tensor.tensor_spec().tile().get_tile_shape()[1];
        TT_FATAL(
            shard_width_in_tiles % num_receivers_per_sender == 0,
            "Tensor {}'s shard is {} tiles wide, which does not split evenly over the {} receiver cores per sender. "
            "Each receiver must get the same whole number of tile columns.",
            i,
            shard_width_in_tiles,
            num_receivers_per_sender);

        tt::DataFormat tensor_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        TT_FATAL(
            tensor_data_format == tt::DataFormat::Bfp4_b || tensor_data_format == tt::DataFormat::Bfp8_b ||
                tensor_data_format == tt::DataFormat::Float16_b,
            "Input tensors must be of type Bfp4_b, Bfp8_b, or Float16_b");
    }

    const ttsl::Span<const Tensor> weight_tensors(input_tensors.data(), input_tensors.size() - 1);
    validate_target_capacity(
        args, compute_dram_prefetcher_geometry(weight_tensors, num_receivers_per_sender), num_receivers_per_sender);

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

    // Each reader reads the weights' addresses in place from its own shard, every layer's in its first row.
    TT_FATAL(tensor_addrs.shard_spec().has_value(), "Tensor containing addresses must have a 2D shard spec");
    const tt::tt_metal::ShardSpec& addrs_shard_spec = *tensor_addrs.shard_spec();
    const uint32_t num_tensors = input_tensors.size() - 1;
    TT_FATAL(
        addrs_shard_spec.shape[1] >= args.num_layers * num_tensors,
        "Each row of tensor_addrs must hold the address of every tensor of every layer, {} layers x {} tensors, but "
        "its shard is only {} wide",
        args.num_layers,
        num_tensors,
        addrs_shard_spec.shape[1]);
    for (const CoreCoord& core : readers.cores) {
        TT_FATAL(
            addrs_shard_spec.grid.contains(core),
            "tensor_addrs must be sharded over the reader cores, each of which reads the addresses from its own "
            "shard, but reader core {} is outside its shard grid {}",
            core.str(),
            addrs_shard_spec.grid.str());
    }
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
