// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>
#include <cmath>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SamplingProgramFactory::create_descriptor(
    const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor) {
    using namespace tt::tt_metal;

    const auto& input_values_tensor = tensor_args.input_values.mesh_tensor();
    const auto& input_indices_tensor = tensor_args.input_indices.mesh_tensor();
    const auto& k = tensor_args.k.mesh_tensor();
    const auto& p = tensor_args.p.mesh_tensor();
    const auto& temp = tensor_args.temp.mesh_tensor();

    const auto& seed = operation_attributes.seed;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    uint32_t random_seed = 0;

    auto* device = &input_values_tensor.mutable_device();

    // The bitonic top-k LLK carries sort indices through the dest register, and the index
    // load/store width is tied to fp32_dest_acc_en (INT32 when enabled, LO16 otherwise). WH/BH
    // support the cheaper 16-bit (UInt16) path with fp32 dest accumulation disabled (unchanged
    // behaviour) so that WH/BH does not suffer any restrictions on the dest register. Every other
    // architecture (e.g. Quasar, which additionally lacks UInt16/UInt32 tile (DFB) metadata
    // support) uses 32-bit (Int32) index intermediates with fp32 dest accumulation enabled. This
    // is gated on !(WH || BH) so new architectures default to the safe 32-bit path.
    const bool use_32bit_index = !(device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.dtype());
    tt::DataFormat index_cb_data_format = use_32bit_index ? tt::DataFormat::Int32 : tt::DataFormat::UInt16;
    // On the 32-bit path (e.g. Quasar), validation already requires k to be INT32 (UInt32 DFB
    // metadata is unsupported there), so the dtype-derived format is correct as-is.
    tt::DataFormat k_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(k.dtype());
    tt::DataFormat p_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(p.dtype());
    tt::DataFormat temp_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(temp.dtype());

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    const auto& output_mesh = output_tensor.mesh_tensor();

    auto input_shape = input_values_tensor.logical_shape();
    const uint32_t tile_height = input_values_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_values_tensor.tensor_spec().tile().get_width();
    // `num_users` is the logical user count (rows in dim 2) in the range [1, 32]; validation
    // guarantees N == C == 1, so it is just input_shape[2]. The data still occupies a single padded
    // row-tile (Ht == 1) and only `num_users` cores run. Decoupling num_cores from Ht*tile_height is
    // what lets <32 users work: the old `(.../tile_height)` would integer-divide to Ht == 0 (and
    // num_cores == 0) for fewer than tile_height users.
    const uint32_t num_users = input_shape[2];
    uint32_t Ht = (num_users + tile_height - 1) / tile_height;  // == 1 for 1..32 users
    uint32_t Wt = input_shape[3] / tile_width;
    auto num_cores = num_users;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    if (sub_core_grids.has_value()) {
        core_grid = sub_core_grids.value();
    }
    auto cores = corerange_to_cores(core_grid, num_cores, true);

    // `sub_core_grids` may be over-provisioned (more cores than users); only the first `num_cores`
    // cores actually run. Confine CB allocation and the reader kernel to exactly those cores so we
    // don't place a reader (with unset runtime args) on the unused cores.
    if (core_grid.num_cores() != num_cores) {
        std::vector<CoreRange> active_core_ranges;
        active_core_ranges.reserve(cores.size());
        for (const auto& core : cores) {
            active_core_ranges.emplace_back(core);
        }
        core_grid = CoreRangeSet(std::move(active_core_ranges));
    }

    validate_reduce_op_program_grid(
        "Sampling",
        core_grid,
        compute_with_storage_grid_size,
        sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr,
        true,
        {});

    if (seed.has_value()) {
        random_seed = seed.value();
    }

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    ProgramDescriptor desc;

    // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    uint32_t input_values_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_values_cb_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    uint32_t index_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * index_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Reduce scaler CBs — separate because MAX and SUM use different tile fill layouts
    tt::DataFormat scalar_df =
        (input_values_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scale_tiles = 1;
    uint32_t scalar_tile_size = tile_size(scalar_df);

    uint32_t scaler_max_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scaler_max_cb_index),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    uint32_t scaler_sum_cb_index = tt::CBIndex::c_17;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scaler_sum_cb_index),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    uint32_t topk_mask_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(topk_mask_cb_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    // Compute kernel CBs
    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_transposed_cb_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt * index_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_transposed_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Output sampling values
    uint32_t values_cb_index = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(values_cb_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    uint32_t cb_local_vals_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_local_vals_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    // Output local indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * index_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_ind_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    uint32_t num_out_tiles = Ht;
    uint32_t cb_cur_max_index = tt::CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_cur_max_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    uint32_t cb_cur_sum_index = tt::CBIndex::c_10;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_out_tiles * input_values_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_cur_sum_index),
            .data_format = input_values_cb_data_format,
            .page_size = input_values_tile_size,
        }}},
    });

    // RM CBs for sampling

    // random number
    const uint32_t rand_tile_size = tile_size(tt::DataFormat::Float16_b);
    constexpr uint32_t rand_tile_index = tt::CBIndex::c_11;
    desc.cbs.push_back(CBDescriptor{
        .total_size = rand_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(rand_tile_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = rand_tile_size,
        }}},
    });

    // final indices
    uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();  // 4 for int32
    uint32_t aligned_final_indices_rm_unit_size = Wt * tile_width * final_indices_rm_unit_size;
    uint32_t final_indices_rm_cb_index = tt::CBIndex::c_12;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Ht * tile_height * aligned_final_indices_rm_unit_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(final_indices_rm_cb_index),
            .data_format = input_indices_cb_data_format,
            .page_size = aligned_final_indices_rm_unit_size,
        }}},
    });

    // Output sampling indices
    uint32_t output_unit_size = output_mesh.element_size();
    uint32_t aligned_out0_unit_size = Ht * tile_height * output_unit_size;
    uint32_t output_cb_index = tt::CBIndex::c_13;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_out0_unit_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = index_cb_data_format,
            .page_size = aligned_out0_unit_size,
        }}},
    });

    // Add k and p circular buffers
    const uint32_t uint32_bytes = 4;
    const uint32_t bf16_bytes = 2;
    uint32_t k_cb_index = tt::CBIndex::c_14;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t k_chunk_size = num_cores * uint32_bytes;
    desc.cbs.push_back(CBDescriptor{
        .total_size = k_chunk_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(k_cb_index),
            .data_format = k_cb_data_format,
            .page_size = k_chunk_size,
        }}},
    });

    uint32_t p_cb_index = tt::CBIndex::c_15;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t p_chunk_size = num_cores * bf16_bytes;
    desc.cbs.push_back(CBDescriptor{
        .total_size = p_chunk_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(p_cb_index),
            .data_format = p_cb_data_format,
            .page_size = p_chunk_size,
        }}},
    });

    // Add temp circular buffer
    uint32_t temp_cb_index = tt::CBIndex::c_16;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t temp_chunk_size = num_cores * bf16_bytes;
    desc.cbs.push_back(CBDescriptor{
        .total_size = temp_chunk_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(temp_cb_index),
            .data_format = temp_cb_data_format,
            .page_size = temp_chunk_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_values_cb_index,
        final_indices_rm_cb_index,
        index_cb_index,
        Ht,
        Wt,
        aligned_final_indices_rm_unit_size,
        tile_height,
        static_cast<uint32_t>(use_32bit_index),
        num_users};
    tt::tt_metal::TensorAccessorArgs(input_values_tensor).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_indices_tensor).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};
    // The reader kernel is created once on the entire core_grid; set the same runtime args on every core
    for (const auto& core : cores) {
        reader_desc.emplace_runtime_args(
            core,
            {
                input_values_tensor,
                input_indices_tensor,
            });
    }
    desc.kernels.push_back(std::move(reader_desc));

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        CoreRangeSet single_core{CoreRange(core, core)};

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(output_mesh).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(temp).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(k).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(p).append_to(writer_compile_time_args);
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {
                output_cb_index,
                topk_mask_cb_index,
                scaler_max_cb_index,
                scaler_sum_cb_index,
                final_indices_rm_cb_index,
                cb_local_vals_index,
                output_ind_cb_index,
                aligned_final_indices_rm_unit_size,
                aligned_out0_unit_size,
                rand_tile_index,
                k_cb_index,
                p_cb_index,
                temp_cb_index,
                i,
                tile_width,
                num_cores,
                static_cast<uint32_t>(use_32bit_index),
                num_users,
            });

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = single_core;
        writer_desc.compile_time_args = writer_compile_time_args;
        writer_desc.config = WriterConfigDescriptor{};
        writer_desc.emplace_runtime_args(core, {output_mesh, temp, k, p});
        desc.kernels.push_back(std::move(writer_desc));

        std::vector<uint32_t> compute_args = {
            input_values_cb_index,
            index_cb_index,
            input_transposed_cb_index,
            index_transposed_cb_index,
            values_cb_index,
            output_ind_cb_index,
            topk_mask_cb_index,
            scaler_max_cb_index,
            scaler_sum_cb_index,
            cb_cur_max_index,
            cb_cur_sum_index,
            Ht,
            Wt,
            static_cast<uint32_t>(std::log2(Wt)),
            rand_tile_index,
            random_seed,
            cb_local_vals_index,
            temp_cb_index,
            tile_width};

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = single_core;
        compute_desc.compile_time_args = compute_args;
        // 32-bit (Int32) sort indices require fp32 dest accumulation so the top-k LLK loads/stores
        // indices in INT32 mode; the 16-bit (UInt16) path uses LO16 mode with fp32 dest acc off.
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = use_32bit_index,
        };
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
