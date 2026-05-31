// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>
#include <set>

#include "ttnn/tensor/tensor.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ConcatProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    ProgramDescriptor desc;
    IDevice* device = output.device();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    constexpr bool rm_orientation = false;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    const uint32_t common_align_len = std::max(input_tensors[0].buffer()->alignment(), output.buffer()->alignment());
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
        num_output_pages = output.physical_volume() / TILE_HW;
        single_page_size = tt::tile_size(cb_data_format);
    }

    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores;
    uint32_t num_tiles_per_core_group_1;
    uint32_t num_tiles_per_core_group_2;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
    std::vector<CoreCoord> cores_list;

    if (sub_core_grids.has_value() && !output.is_sharded()) {
        // Use sub_core_grids for interleaved output
        uint32_t ncores = sub_core_grids->num_cores();
        TT_FATAL(ncores != 0, "number of cores cannot be 0");

        // Find the maximum number of cores that evenly divides num_output_pages
        for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
            if (num_output_pages % core_id == 0) {
                ncores = core_id;
                break;
            }
            ncores--;
        }
        TT_FATAL(
            (num_output_pages % ncores == 0),
            "{} num of pages are not split uniformly across {} num of cores",
            num_output_pages,
            ncores);

        cores_list = corerange_to_cores(sub_core_grids.value(), ncores, rm_orientation);
        all_cores =
            num_cores_to_corerangeset_in_subcoregrids(cores_list[0], ncores, sub_core_grids.value(), rm_orientation);
        if (ncores == 1) {
            all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores_list[0]));
        }
        num_cores = ncores;
        num_tiles_per_core_group_1 = num_output_pages / ncores;
        num_tiles_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    } else {
        // Use full compute grid
        const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_tiles_per_core_group_1_result,
             num_tiles_per_core_group_2_result] =
                split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_tiles_per_core_group_1 = num_tiles_per_core_group_1_result;
        num_tiles_per_core_group_2 = num_tiles_per_core_group_2_result;
    }

    const uint32_t num_input_tensors = input_tensors.size();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    // Depth=2 is a prefetch optimization; fall back to depth=1 when it would overflow L1.
    const uint32_t l1_budget =
        (device->l1_size_per_core() / 2) - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_capacity =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    TT_FATAL(
        single_page_size <= l1_capacity,
        "ttnn.concat: required CB page size ({} B) exceeds per-core L1 capacity ({} B); "
        "op cannot fit on this device.",
        single_page_size,
        l1_capacity);
    uint32_t num_input_pages = 2;
    if (num_input_pages * single_page_size > l1_budget) {
        num_input_pages = 1;
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages * single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_page_size,
        }}},
    });

    const uint32_t num_dims = output.padded_shape().rank();

    std::vector<uint32_t> src_addr(num_input_tensors);
    std::vector<uint32_t> num_pages_per_block(num_input_tensors);
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);
    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.padded_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && dim < num_dims - 1) {
            num_accum_pages /= output.padded_shape()[-1];
        }
    } else {
        if (dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    uint32_t num_output_pages_per_block = 0;

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            page_size_per_tensor[i] = buffer->page_size();
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = input_tensors[i].padded_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages;
                num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            page_size_per_tensor[i] = buffer->page_size();
            uint32_t dim_pages = input_tensors[i].padded_shape()[dim] / scale_factor;
            num_pages_per_block[i] = num_accum_pages * dim_pages;
            num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }
    std::vector<uint32_t> common_reader_kernel_args = {0, 0, 0};
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), src_addr.cbegin(), src_addr.cend());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_pages_per_block.cbegin(), num_pages_per_block.cend());

    // Reader compile-time args
    // Data is 32 byte aligned
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {src0_cb_index, num_input_tensors};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), page_size_per_tensor.cbegin(), page_size_per_tensor.cend());
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args);
    }

    KernelDescriptor::Defines concat_defines;
    if (rm_layout && dim == num_dims - 1) {
        concat_defines.emplace_back("WIDTH_CONCAT", "1");
    }

    KernelDescriptor::CompileTimeArgs writer_compile_time_args;
    if (rm_layout) {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index, dst_buffer->page_size()};
    } else {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    }
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Pin kernel core_ranges to the full compute grid (or the full
    // sub_core_grids when supplied), regardless of how many cores
    // `split_work_to_cores` allocated for the work-split. Runtime args
    // are still only emplaced on the work-split subset `cores`, so
    // per-core work is unchanged.
    //
    // Why: the program-cache slow path
    // (`apply_descriptor_runtime_args` in
    // mesh_device_operation_adapter.hpp) re-runs `create_descriptor`
    // on cache hit and applies the fresh descriptor's runtime_args
    // against the cached program's compiled kernel placement. If the
    // cached call had fewer `num_output_pages` (and thus a smaller
    // `all_cores`) than a subsequent call that hash-collides into the
    // same cache entry, the new call's runtime_args contain core
    // coordinates the cached kernel was never placed on, which fires
    //   TT_FATAL: Cannot get runtime args for kernel ... not placed on core X-Y
    // Making the kernel placement invariant (full grid every time)
    // means the cached program always covers any possible runtime
    // args core, removing this mismatch class entirely.
    const auto cores = (sub_core_grids.has_value() && !output.is_sharded())
                           ? cores_list
                           : grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    const CoreRangeSet kernel_core_ranges = (sub_core_grids.has_value() && !output.is_sharded())
                                                ? sub_core_grids.value()
                                                : CoreRangeSet(CoreRange(
                                                      {0, 0},
                                                      {device->compute_with_storage_grid_size().x - 1,
                                                       device->compute_with_storage_grid_size().y - 1}));

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = rm_layout ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                                            "reader_concat_stick_layout_interleaved_start_id.cpp"
                                          : "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                                            "reader_concat_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = kernel_core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(concat_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        rm_layout
            ? "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = kernel_core_ranges;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    const uint32_t g1_num_cores = core_group_1.num_cores();
    for (uint32_t i = 0, num_pages_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_pages_per_core =
            (i < g1_num_cores) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
        const uint32_t block_id = num_pages_written / num_output_pages_per_block;
        uint32_t id_within_block = num_pages_written % num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            page_id_per_tensor[j] = block_id * num_pages_per_block[j];
            if (id_within_block == 0) {
                continue;
            }
            if (id_within_block >= num_pages_per_block[j]) {
                page_id_per_tensor[j] += num_pages_per_block[j];
                id_within_block -= num_pages_per_block[j];
                curr_tensor = j + 1;
            } else {
                page_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] = num_pages_per_core;
        reader_kernel_args[1] = curr_tensor;
        reader_kernel_args[2] = curr_tensor_id;
        reader_kernel_args.insert(reader_kernel_args.end(), page_id_per_tensor.begin(), page_id_per_tensor.end());

        std::vector<uint32_t> writer_kernel_args;
        if (rm_layout) {
            writer_kernel_args = {
                dst_buffer->address(), output.buffer()->page_size(), num_pages_per_core, num_pages_written};
        } else {
            writer_kernel_args = {dst_buffer->address(), num_pages_per_core, num_pages_written};
        }

        reader_desc.runtime_args.emplace_back(core, std::move(reader_kernel_args));
        writer_desc.runtime_args.emplace_back(core, std::move(writer_kernel_args));
        num_pages_written += num_pages_per_core;
    }

    // Pinning the kernel to the full grid (kernel_core_ranges above)
    // means the cached program reserves rt-args storage for every
    // core in the full grid, sized by the slot we set on the very
    // first call. If a later call hits the cache with a *larger*
    // work-split (more entries in `cores`), apply_descriptor_runtime_args
    // tries to write rt-args at slots that were never allocated on
    // that core, surfacing as
    //   TT_FATAL: Index N is larger than runtime args size 0
    // Emplace a same-sized zero rt-args vector on every full-grid
    // core not in `cores`. num_tiles=0 short-circuits the kernel's
    // main loop, and the other zero-valued indices are read but never
    // dereferenced because the loop never runs.
    if (!sub_core_grids.has_value() || output.is_sharded()) {
        std::set<CoreCoord> cores_with_work(cores.begin(), cores.end());
        const size_t reader_rt_size = 3 + 2 * num_input_tensors + num_input_tensors;
        const size_t writer_rt_size = rm_layout ? 4 : 3;
        for (uint32_t x = 0; x < num_cores_x; ++x) {
            for (uint32_t y = 0; y < num_cores_y; ++y) {
                CoreCoord c{x, y};
                if (cores_with_work.count(c) == 0) {
                    reader_desc.runtime_args.emplace_back(c, std::vector<uint32_t>(reader_rt_size, 0));
                    writer_desc.runtime_args.emplace_back(c, std::vector<uint32_t>(writer_rt_size, 0));
                }
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
