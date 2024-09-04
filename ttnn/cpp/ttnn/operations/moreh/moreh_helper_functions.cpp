// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace operations {
namespace primary {

std::tuple<CoreRangeSet, CoreRangeSet, CoreRangeSet> add_core_offset(
    CoreRangeSet all_cores,
    CoreRangeSet core_group_1,
    CoreRangeSet core_group_2,
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

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> split_work_to_cores(
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
         num_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, units_to_divide);

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
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch()),
            .compile_args = compile_args,
            .defines = defines});
}

[[maybe_unused]] KernelHandle CreateWriteKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &compile_args,
    std::map<string, string> defines) {
    return tt_metal::CreateKernel(
        program,
        file_name,
        core_spec,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch()),
            .compile_args = compile_args,
            .defines = defines});
}

[[maybe_unused]] std::vector<KernelHandle> CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    std::vector<ComputeKernelArg> args,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool preserve_fp32_precision) {
    std::vector<KernelHandle> compute_kernel_ids{};
    KernelHandle compute_kernel_id{};
    for (auto arg : args) {
        compute_kernel_id =
            CreateComputeKernel(program, file_name, arg, defines, math_fidelity, fp32_dest_acc_en, math_approx_mode, preserve_fp32_precision);
        compute_kernel_ids.push_back(compute_kernel_id);
    }
    return compute_kernel_ids;
}

[[maybe_unused]] KernelHandle CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    ComputeKernelArg arg,
    std::map<std::string, std::string> defines,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool preserve_fp32_precision) {
    KernelHandle compute_kernel_id{0};
    if (arg.num_tile_per_core_group > 0) {
        compute_kernel_id = CreateKernel(
            program,
            file_name,
            arg.core_spec,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .preserve_fp32_precision = preserve_fp32_precision,
                .math_approx_mode = math_approx_mode,
                .compile_args = arg.compile_args,
                .defines = defines});
    }
    return compute_kernel_id;
}

[[maybe_unused]] std::vector<CBHandle> CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    std::vector<CircularBufferArg> args) {
    std::vector<CBHandle> cb_ids{};
    CBHandle cb_id{};
    for (auto arg : args) {
        cb_id = CreateCircularBuffer(program, core_range, data_format, arg);
        cb_ids.push_back(cb_id);
    }
    return cb_ids;
}

[[maybe_unused]] CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_range,
    tt::DataFormat data_format,
    CircularBufferArg arg) {
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
    const std::initializer_list<DataType> &data_types,
    Layout layout,
    bool check_dtype,
    bool check_layout) {
    if (check_layout) {
        TT_FATAL(tensor.get_layout() == layout, "{} {} only supports {} layout.", op_name, tensor_name, magic_enum::enum_name(layout));
    }
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} {} need to be on device!", op_name, tensor_name);
    TT_FATAL(tensor.buffer() != nullptr, "{} {} need to be allocated in buffers on device!", op_name, tensor_name);

    if (check_dtype) {
        bool dtype_supported = false;
        for (const auto& data_type : data_types) {
            if (tensor.get_dtype() == data_type) {
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

            TT_FATAL(dtype_supported, "{} {} only supports specific data types. {} ", op_name, tensor_name, dtype_string);
        }
    }
}

void check_tensor(
    std::optional<Tensor> tensor,
    const std::string& op_name,
    const std::string& tensor_name,
    const std::initializer_list<DataType> &data_types,
    Layout layout,
    bool check_dtype,
    bool check_layout) {
    if (!tensor.has_value()) {
        return;
    }
    check_tensor(tensor.value(), op_name, tensor_name, data_types, layout, check_dtype, check_layout);
}

bool is_hw_dim(uint32_t dim, uint32_t rank) {
    if (rank == 1 || rank == 2) {
        return true;
    }
    if (rank >= 3) {
        if (dim >= rank - 2) {
            return true;
        }
    }
    return false;
}

uint32_t compute_inner(Shape shape, uint32_t dim) {
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

uint32_t compute_outer(Shape shape, uint32_t dim) {
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

void expand_to_max_dim(std::vector<uint32_t> &dim, const Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
        dim[i] = shape[idx];
    }
}


}  // namespace primary
}  // namespace operations
}  // namespace tt
