// gelu_program_factory.cpp
#include <algorithm>

#include "gelu_backward_program_factory.hpp"
#include "gelu_backward_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "tt-metalium/host_api.hpp"

namespace {
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/gelu/device/kernels/dataflow/writer_gelu_interleaved_start_id.cpp";
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/gelu/device/kernels/dataflow/reader_gelu_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath = "ttnn/cpp/ttnn/operations/gelu/device/kernels/compute/gelu_kernel.cpp";

constexpr auto kDstBufferIdx = 0;
constexpr auto kSrcBufferIdx = 0;

constexpr auto kSrc0CbIndex = tt::CBIndex::c_0;
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;

constexpr uint32_t kNumInputTiles = 2;
constexpr uint32_t kNumOutputTiles = 2;
}  // namespace

namespace ttnn::operations::gelu_backward::program {

using namespace tt::constants;

// Helper struct for kernels
struct GeluKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

// Helper functions to create circular buffers and kernels
inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal;

    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
                                         .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

inline tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path) {
    using namespace tt::tt_metal;

    return CreateKernel(program, kernel_path, core_ranges, ReaderDataMovementConfig(compile_time_args));
}

inline tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path) {
    using namespace tt::tt_metal;

    return CreateKernel(program, kernel_path, core_ranges, WriterDataMovementConfig(compile_time_args));
}

inline tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path,
    bool math_approx_mode) {
    using namespace tt::tt_metal;

    return CreateKernel(
        program,
        kernel_path,
        core_ranges,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compile_time_args});
}

GeluBackwardProgramFactory::cached_program_t GeluBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    // Setup device, data formats, tile sizes, and compute split
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat data_fmt_in = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat data_fmt_out = datatype_to_dataformat_converter(output.get_dtype());

    uint32_t single_tile_size_in = tt::tt_metal::detail::TileSize(data_fmt_in);
    uint32_t single_tile_size_out = tt::tt_metal::detail::TileSize(data_fmt_out);

    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // Create and configure circular buffers
    auto cb_src0 =
        create_circular_buffer(program, all_cores, kSrc0CbIndex, data_fmt_in, single_tile_size_in, kNumInputTiles);

    auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_fmt_out, single_tile_size_out, kNumOutputTiles);

    // Create reader/writer kernels
    auto src_buffer = input.buffer();
    bool src_is_dram = (src_buffer->buffer_type() == BufferType::DRAM);

    std::vector<uint32_t> reader_compile_args = {static_cast<uint32_t>(src_is_dram)};

    auto dst_buffer = output.buffer();
    bool dst_is_dram = (dst_buffer->buffer_type() == BufferType::DRAM);

    std::vector<uint32_t> writer_compile_args = {
        static_cast<uint32_t>(kOutputCbIndex), static_cast<uint32_t>(dst_is_dram)};

    GeluKernels kernels;
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, kReaderKernelPath);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, kWriterKernelPath);

    // Create compute kernels for GELU
    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_size
    };

    bool math_approx_mode = false;

    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, kComputeKernelPath, math_approx_mode);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
        };

        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, kComputeKernelPath, math_approx_mode);
    }

    // Assign runtime args for each core
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many tiles this core will process
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        // Reader kernel: (src_addr, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(program, kernels.reader, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        // Writer kernel: (dst_addr, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(program, kernels.writer, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    // Return the configured program & relevant shared variables
    return cached_program_t{
        std::move(program),
        {/* gelu_reader_kernel_id  = */ kernels.reader,
         /* gelu_writer_kernel_id  = */ kernels.writer,
         /* gelu_kernel_group_1_id = */ kernels.compute_group_1,
         /* gelu_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void GeluBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    auto& shared_vars = cached_program.shared_variables;
    auto& gelu_reader_kernel = shared_vars.gelu_reader_kernel_id;
    auto& gelu_writer_kernel = shared_vars.gelu_writer_kernel_id;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto& input = tensor_args.input;
    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    // Only update buffer addresses
    auto& reader_runtime_args = GetRuntimeArgs(program, gelu_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, gelu_writer_kernel);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update the source address for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kSrcBufferIdx] = src_buffer->address();
        }
        // Update the destination address for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kDstBufferIdx] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::gelu_backward::program
