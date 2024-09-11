// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/experimental/paged_cache/device/paged_update_cache_program_factory.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

using namespace tt::constants;
using namespace tt;

bool enable_fp32_dest(const tt_metal::Device * device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config, const tt::DataFormat& input_cb_data_format) {
    bool fp32_dest_acc_en;
    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            fp32_dest_acc_en = input_cb_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_THROW("arch not supported");
        }

    }, compute_kernel_config);

    return fp32_dest_acc_en;
}

operation::ProgramWithCallbacks paged_update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, std::optional<const Tensor> update_idxs_tensor, std::optional<const Tensor> page_table, const std::vector<uint32_t> update_idxs, const uint32_t batch_offset, ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    Program program{};

    tt_metal::Device *device = input_tensor.device();


    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.get_dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest(device, compute_kernel_config, input_cb_data_format);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_tensor_tile_size = 0;
    uint32_t index_buffer_addr = 0;
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
    if (use_index_tensor) {
        index_buffer_addr = use_index_tensor ? update_idxs_tensor.value().buffer()->address() : 0;
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().get_dtype());
        index_tensor_tile_size = tt_metal::detail::TileSize(index_data_format);
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();

        log2_page_size = std::log2(index_stick_size);
        TT_FATAL(1 << log2_page_size == index_stick_size, "Error");
    }

    // Pagetable-specific parameters
    bool is_paged_cache = page_table.has_value();
    uint32_t batch_size = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        batch_size = page_table_tensor.get_legacy_shape()[0];
        block_size = cache_tensor.get_legacy_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.get_legacy_shape()[1];
        page_table_stick_size = page_table_tensor.get_legacy_shape()[-1] * page_table_tensor.element_size();
        log2_page_table_stick_size = std::log2(page_table_stick_size);
        TT_FATAL(1 << log2_page_table_stick_size == page_table_stick_size, "Error");

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.get_dtype());

        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    uint32_t Wt = cache_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.get_legacy_shape()[-1] * sizeof(float) : cache_tensor.get_legacy_shape()[-1] * 2; // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor.volume() / TILE_HW;
    uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.get_legacy_shape()[0];
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    uint32_t B = input_tensor.get_legacy_shape()[1];
    uint32_t num_heads = cache_tensor.get_legacy_shape()[1];

    log_debug("cache_cb_data_format: {}", cache_cb_data_format);
    log_debug("input_cb_data_format: {}", input_cb_data_format);
    log_debug("interm_cb_data_format: {}", interm_cb_data_format);
    log_debug("Wbytes: {}", Wbytes);
    log_debug("Wt: {}", Wt);


    std::optional<ShardSpec> shard_spec = input_tensor.shard_spec();
    bool row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet all_cores = shard_spec.value().grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    auto bbox = all_cores.bounding_box();
    uint32_t num_cores_x = bbox.end_coord.x + 1;
    uint32_t num_cores_y = bbox.end_coord.y + 1;

    auto in1_buffer_address = shard_spec.has_value() ? input_tensor.buffer() : nullptr;

    uint32_t num_cache_tiles = 2 * Wt; // double buffered
    uint32_t num_interm_tiles = 2 * Wt; // double buffered
    uint32_t num_output_tiles = B * Wt;

    const tt::CB src0_cb_index = CB::c_in0;
    const tt::CB src1_cb_index = CB::c_in1;
    const tt::CB cb_index_id = CB::c_in2;
    const tt::CB cb_pagetable_id = CB::c_in3;
    const tt::CB intermed0_cb_index = CB::c_intermed0;
    const tt::CB intermed1_cb_index = CB::c_intermed1;
    const tt::CB intermed2_cb_index = CB::c_intermed2;
    const tt::CB output_cb_index = CB::c_out0;

    create_cb(src0_cb_index, program, all_cores, cache_single_tile_size, num_cache_tiles, cache_cb_data_format);
    auto [_, cb_src1] = create_cb(src1_cb_index, program, all_cores, input_single_tile_size, num_input_tiles, input_cb_data_format, in1_buffer_address);
    create_cb({intermed0_cb_index, intermed1_cb_index}, program, all_cores, interm_single_tile_size, num_interm_tiles, interm_cb_data_format);
    create_cb(intermed2_cb_index, program, all_cores, interm_single_tile_size, num_interm_tiles, interm_cb_data_format);
    create_cb(output_cb_index, program, all_cores, cache_single_tile_size, num_output_tiles, cache_cb_data_format);

    if (use_index_tensor) {
        create_cb(cb_index_id, program, all_cores, index_tensor_tile_size, 1, index_data_format);
    }

    if (is_paged_cache) {
        create_cb(cb_pagetable_id, program, all_cores, page_table_stick_size, 1, page_table_data_format);
    }

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index,
        // Index tensor args
        (std::uint32_t) use_index_tensor,
        (std::uint32_t) index_is_dram,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        log2_page_size,
        index_stick_size,
        // page_table args
        (std::uint32_t) is_paged_cache,
        (std::uint32_t) num_heads,
        (std::uint32_t) block_size,
        (std::uint32_t) block_size_t,
        (std::uint32_t) max_blocks_per_seq,
        log2_page_table_stick_size,
        page_table_stick_size,
        (std::uint32_t) page_table_is_dram,
        cb_pagetable_id,

    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) output_cb_index,
        (std::uint32_t) intermed0_cb_index,
        (std::uint32_t) intermed1_cb_index,
        (std::uint32_t) intermed2_cb_index,
        // Index tensor args
        (std::uint32_t) use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        Wbytes,
        // page_table args
        (std::uint32_t) is_paged_cache,
        (std::uint32_t) num_heads,
        (std::uint32_t) block_size,
        (std::uint32_t) block_size_t,
        (std::uint32_t) max_blocks_per_seq,
        cb_pagetable_id,
    };

    std::vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        output_cb_index,
        Wt,
    };

    auto unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp",
        all_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args}
    );

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; ++i) {
        const CoreCoord &core = cores.at(i);
        const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);
        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        // Offset to write into untilized cache
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                index_buffer_addr,
                i,
                is_paged_cache ? page_table.value().buffer()->address() : 0,
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
            }
        );
    }

    auto override_runtime_arguments_callback = [
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        cores,
        Wbytes,
        Wt,
        cb_src1,
        cache_batch_num_tiles,
        use_index_tensor,
        is_paged_cache
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const std::vector<uint32_t> update_idxs = static_cast<const PagedUpdateCacheDeviceOperation*>(operation)->update_idxs;

        auto src_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = input_tensors.at(0).buffer();

        auto index_tensor_addr = use_index_tensor ? optional_input_tensors.at(0).value().buffer()->address() : 0;
        auto page_table_tensor_addr = is_paged_cache ? optional_input_tensors.at(1).value().buffer()->address() : 0;

        if (input_tensors.at(1).is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer);
        }

        auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

        for (uint32_t i = 0, num_tiles_read = 0; i < cores.size(); ++i){
            const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);
            // Cache tile info
            const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
            const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
            // Offset to write into untilized cache
            uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

            const CoreCoord &core = cores.at(i);

            {
                auto &runtime_args = reader_args_by_core.at(core.x).at(core.y);
                runtime_args[0] = dst_buffer->address();
                runtime_args[1] = cache_start_id;
                runtime_args[2] = index_tensor_addr;
                runtime_args[4] = page_table_tensor_addr;
            }

            {
                auto &runtime_args = writer_args_by_core.at(core.x).at(core.y);
                runtime_args[0] = dst_buffer->address();
                runtime_args[1] = cache_start_id;
                runtime_args[2] = tile_update_offset_B;
            }
        }

    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // ttnn::operations::experimental::paged_cache::detail
