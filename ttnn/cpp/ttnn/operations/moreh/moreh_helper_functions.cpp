// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_helper_functions.hpp"

#include <magic_enum/magic_enum.hpp>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>

#include "tt-metalium/hal.hpp"

namespace ttnn {
namespace operations {

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t offset_x,
    uint32_t offset_y) {
    std::set<CoreRange> new_all_cores_set;
    std::set<CoreRange> new_core_group_1_set;
    std::set<CoreRange> new_core_group_2_set;

    for (auto core : all_cores.ranges()) {
        new_all_cores_set.insert(CoreRange(
            {core.start_coord.x + offset_x, core.start_coord.y + offset_y},
            {core.end_coord.x + offset_x, core.end_coord.y + offset_y}));
    }

    for (auto core : core_group_1.ranges()) {
        new_core_group_1_set.insert(CoreRange(
            {core.start_coord.x + offset_x, core.start_coord.y + offset_y},
            {core.end_coord.x + offset_x, core.end_coord.y + offset_y}));
    }

    for (auto core : core_group_2.ranges()) {
        new_core_group_2_set.insert(CoreRange(
            {core.start_coord.x + offset_x, core.start_coord.y + offset_y},
            {core.end_coord.x + offset_x, core.end_coord.y + offset_y}));
    }

    CoreRangeSet new_all_cores(new_all_cores_set);
    CoreRangeSet new_core_group_1(new_core_group_1_set);
    CoreRangeSet new_core_group_2(new_core_group_2_set);

    return std::make_tuple(new_all_cores, new_core_group_1, new_core_group_2);
}

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores_wt_core_range(
    CoreRange core_range, uint32_t units_to_divide) {
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    CoreCoord grid_size = {core_w, core_h};
    auto
        [num_cores,
         all_cores_t,
         core_group_1_t,
         core_group_2_t,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = tt_metal::split_work_to_cores(grid_size, units_to_divide);

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    auto [all_cores, core_group_1, core_group_2] =
        add_core_offset(all_cores_t, core_group_1_t, core_group_2_t, core_x_offset, core_y_offset);

    {
        auto iter = core_group_1.ranges();
        for_each(
            iter.begin(), iter.end(), [](CoreRange core) { log_debug(LogTest, "Use core_group_1 {}", core.str()); });
    }
    log_debug(LogTest, "num_tiles_per_core_group_1 {}", num_tiles_per_core_group_1);

    {
        auto iter = core_group_2.ranges();
        for_each(
            iter.begin(), iter.end(), [](CoreRange core) { log_debug(LogTest, "Use core_group_2 {}", core.str()); });
    }
    log_debug(LogTest, "num_tiles_per_core_group_2 {}", num_tiles_per_core_group_2);

    return std::make_tuple(
        num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);
}

[[maybe_unused]] KernelHandle CreateReadKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    std::map<std::string, std::string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(hal::get_arch()),
            .compile_args = compile_args,
            .defines = std::move(defines)});
}

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::vector<uint32_t>& compile_args,
    std::map<std::string, std::string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(hal::get_arch()),
            .compile_args = compile_args,
            .defines = std::move(defines)});
}

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program& program,
    const std::string& file_name,
    const std::vector<ComputeKernelArg>& args,
    const std::map<std::string, std::string>& defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    const std::vector<UnpackToDestMode>& unpack_to_dest_mode) {
    std::vector<KernelHandle> compute_kernel_ids{};
    KernelHandle compute_kernel_id{};
    for (auto arg : args) {
        compute_kernel_id = CreateComputeKernel(
            program, file_name, arg, defines, math_fidelity, fp32_dest_acc_en, math_approx_mode, unpack_to_dest_mode);
        compute_kernel_ids.push_back(compute_kernel_id);
    }
    return compute_kernel_ids;
}

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program& program,
    const std::string& file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    std::vector<UnpackToDestMode> unpack_to_dest_mode) {
    KernelHandle compute_kernel_id{0};
    if (arg.num_tile_per_core_group > 0) {
        compute_kernel_id = CreateKernel(
            program,
            file_name,
            arg.core_spec,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
                .math_approx_mode = math_approx_mode,
                .compile_args = arg.compile_args,
                .defines = std::move(defines)});
    }
    return compute_kernel_id;
}

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program& program,
    const std::string& file_name,
    const std::vector<ComputeKernelArg>& args,
    const ComputeKernelConfig& config) {
    std::vector<KernelHandle> compute_kernel_ids{};
    KernelHandle compute_kernel_id{};
    for (auto arg : args) {
        compute_kernel_id = CreateComputeKernel(program, file_name, arg, config);
        compute_kernel_ids.push_back(compute_kernel_id);
    }
    return compute_kernel_ids;
}

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program& program, const std::string& file_name, ComputeKernelArg arg, const ComputeKernelConfig& config) {
    KernelHandle compute_kernel_id{0};
    if (arg.num_tile_per_core_group > 0) {
        compute_kernel_id = CreateKernel(
            program,
            file_name,
            arg.core_spec,
            tt_metal::ComputeConfig{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = config.fp32_dest_acc_en,
                .unpack_to_dest_mode = config.unpack_to_dest_mode,
                .math_approx_mode = config.math_approx_mode,
                .compile_args = arg.compile_args,
                .defines = config.defines});
    }
    return compute_kernel_id;
}

[[maybe_unused]] std::vector<CBHandle> CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_range,
    tt::DataFormat data_format,
    const std::vector<CircularBufferArg>& args) {
    std::vector<CBHandle> cb_ids{};
    CBHandle cb_id{};
    for (const auto& arg : args) {
        cb_id = CreateCircularBuffer(program, core_range, data_format, arg);
        cb_ids.push_back(cb_id);
    }
    return cb_ids;
}

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_range,
    tt::DataFormat data_format,
    const CircularBufferArg& arg) {
    CBHandle cb_id{0};
    if (arg.num_tiles > 0) {
        auto _buffer_index = arg.buffer_index;
        auto _num_tiles = arg.num_tiles;
        auto _data_format = (arg.data_format != tt::DataFormat::Invalid) ? arg.data_format : data_format;
        auto _core_range = (arg.core_range != std::nullopt) ? arg.core_range : core_range;

        tt_metal::CircularBufferConfig cb_config =
            tt_metal::CircularBufferConfig(
                _num_tiles * tt_metal::detail::TileSize(_data_format), {{_buffer_index, _data_format}})
                .set_page_size(_buffer_index, tt_metal::detail::TileSize(_data_format));

        cb_id = tt_metal::CreateCircularBuffer(program, _core_range.value(), cb_config);
    }
    return cb_id;
}

void check_tensor(
    const Tensor& tensor,
    const std::string& op_name,
    const std::string& tensor_name,
    const std::initializer_list<DataType>& data_types,
    Layout layout,
    bool check_dtype,
    bool check_layout) {
    if (check_layout) {
        TT_FATAL(
            tensor.layout() == layout,
            "{} {} only supports {} layout.",
            op_name,
            tensor_name,
            magic_enum::enum_name(layout));
    }
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} {} need to be on device!", op_name, tensor_name);
    TT_FATAL(tensor.buffer() != nullptr, "{} {} need to be allocated in buffers on device!", op_name, tensor_name);

    if (check_dtype) {
        bool dtype_supported = false;
        for (const auto& data_type : data_types) {
            if (tensor.dtype() == data_type) {
                dtype_supported = true;
                break;
            }
        }
        if (!dtype_supported) {
            std::string dtype_string = "[";
            bool is_first = true;
            for (const auto& data_type : data_types) {
                if (!is_first) {
                    dtype_string += ", ";
                }
                dtype_string += fmt::format("{}", magic_enum::enum_name(data_type));
                is_first = false;
            }
            dtype_string += "]";

            TT_FATAL(
                dtype_supported, "{} {} only supports specific data types. {} ", op_name, tensor_name, dtype_string);
        }
    }
}

void check_tensor(
    std::optional<Tensor> tensor,
    const std::string& op_name,
    const std::string& tensor_name,
    const std::initializer_list<DataType>& data_types,
    Layout layout,
    bool check_dtype,
    bool check_layout) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name, tensor_name, data_types, layout, check_dtype, check_layout);
}

bool is_hw_dim(uint32_t dim, uint32_t rank) { return (dim >= rank - 2); }

uint32_t compute_inner(const ttnn::Shape& shape, uint32_t dim) {
    uint32_t num_inner = 1;
    auto rank = shape.rank();

    for (uint32_t i = rank - dim; i < rank; i++) {
        auto size = shape[i];
        if (is_hw_dim(i, rank)) {
            size = tt::div_up(size, constants::TILE_WIDTH);
        }
        num_inner *= size;
    }

    return num_inner;
}

uint32_t compute_outer(const ttnn::Shape& shape, uint32_t dim) {
    uint32_t num_outer = 1;
    auto rank = shape.rank();

    for (uint32_t i = 0; i < rank - dim; i++) {
        auto size = shape[i];
        if (is_hw_dim(i, rank)) {
            size = tt::div_up(size, constants::TILE_WIDTH);
        }
        num_outer *= size;
    }
    return num_outer;
}

void expand_to_max_dim(ttnn::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
        dim[i] = shape[idx];
    }
}

void validate_input_with_dim(const Tensor& input, const int64_t& dim) {
    const auto& input_shape = input.padded_shape();
    const auto input_rank = input_shape.rank();
    log_debug(LogOp, "{}:{} input_rank {}", __func__, __LINE__, input_rank);
    TT_FATAL(
        (dim >= 0 && dim <= tt::tt_metal::MAX_NUM_DIMENSIONS),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL((dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

void validate_output_with_keepdim(const Tensor& input, const Tensor& output, const int64_t& dim, const bool& keepdim) {
    auto input_shape = input.padded_shape();
    auto input_shape_wo_padding = input.logical_shape();
    const auto input_rank = input_shape_wo_padding.rank();
    auto padded_dim = dim + input_shape.rank() - input_shape_wo_padding.rank();

    const auto& output_shape = output.padded_shape();
    const auto& output_shape_wo_padding = output.logical_shape();
    const auto output_rank = output_shape_wo_padding.rank();

    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    log_debug(LogOp, "{}:{} keepdim {} dim {}", __func__, __LINE__, keepdim, dim);
    log_debug(LogOp, "{}:{} input_shape {} wo_padding {}", __func__, __LINE__, input_shape, input_shape_wo_padding);
    log_debug(LogOp, "{}:{} output_shape {} wo_paddoutg {}", __func__, __LINE__, output_shape, output_shape_wo_padding);

    if (keepdim) {
        bool ranks_are_equal = (input_rank == output_rank);
        input_shape[padded_dim] = (is_tile_dim) ? (TILE_HEIGHT) : (1);
        input_shape_wo_padding[dim] = 1;

        if (!ranks_are_equal) {
            log_warning(
                LogOp,
                "{}:{} input_rank {} and output_rank {} are not the same in keepdim mode",
                __func__,
                __LINE__,
                input_rank,
                output_rank);
        }

        ttnn::SmallVector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> input_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        ttnn::SmallVector<uint32_t> output_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        expand_to_max_dim(input_dim, input_shape);
        expand_to_max_dim(output_dim, output_shape);
        expand_to_max_dim(input_dim_wo_padding, input_shape_wo_padding);
        expand_to_max_dim(output_dim_wo_padding, output_shape_wo_padding);

        for (int i = 0; i < input_shape.rank(); ++i) {
            TT_FATAL(input_dim[i] == output_dim[i], "Error");
        }
        for (int i = 0; i < input_shape_wo_padding.rank(); ++i) {
            TT_FATAL(input_dim_wo_padding[i] == output_dim_wo_padding[i], "Error");
        }
    } else {
        ttnn::SmallVector<uint32_t> expected_output_shape;
        for (int i = 0; i < output_shape.rank(); ++i) {
            if (i == padded_dim && !is_tile_dim) {
                expected_output_shape.push_back(1);
            }
            expected_output_shape.push_back(output_shape[i]);
        }
        ttnn::SmallVector<uint32_t> expected_output_shape_wo_padding;
        for (int i = 0; i < output_shape_wo_padding.rank(); ++i) {
            if (i == dim && !is_tile_dim) {
                expected_output_shape_wo_padding.push_back(1);
            }
            expected_output_shape_wo_padding.push_back(output_shape_wo_padding[i]);
        }

        log_debug(LogOp, "{}:{} expected_output_shape {}", __func__, __LINE__, expected_output_shape);
        log_debug(
            LogOp, "{}:{} expected_output_shape_wo_padding {}", __func__, __LINE__, expected_output_shape_wo_padding);
        for (int i = 0; i < expected_output_shape.size(); ++i) {
            TT_FATAL(i == padded_dim || input_shape[i] == expected_output_shape[i], "Error");
        }
        for (int i = 0; i < expected_output_shape_wo_padding.size(); ++i) {
            TT_FATAL(i == dim || input_shape_wo_padding[i] == expected_output_shape_wo_padding[i], "Error");
        }
    }
}

void initialize_dims_with_range(ttnn::SmallVector<int64_t>& dims, uint32_t input_rank) {
    dims.resize(input_rank);
    std::iota(dims.begin(), dims.end(), 0);
}

ttnn::SmallVector<int64_t> get_dim(
    const std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>>& dim, uint32_t input_rank) {
    ttnn::SmallVector<int64_t> dims;
    if (!dim.has_value()) {
        initialize_dims_with_range(dims, input_rank);
    } else if (std::holds_alternative<int64_t>(dim.value())) {
        auto d = std::get<int64_t>(dim.value());
        dims.push_back(d);
    } else {
        dims = std::get<ttnn::SmallVector<int64_t>>(dim.value());
        if (dims.empty()) {
            initialize_dims_with_range(dims, input_rank);
        }
    }
    return dims;
}

std::tuple<uint32_t, uint32_t, uint32_t> extract_spatial_dims(const ttnn::Shape& shape) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t W = shape[-1];
    uint32_t H = shape[-2];

    uint32_t other_dims_product = 1;
    for (auto i = 0; i < rank - 2; ++i) {
        other_dims_product *= shape[i];
    }

    return {W, H, other_dims_product};
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(
    const ttnn::Shape& shape, uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return {Wt, Ht, inner_tile_size, reduce_tile_size};
}

}  // namespace operations
}  // namespace ttnn
