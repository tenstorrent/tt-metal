#include "example_device_operation.hpp"

namespace ttnn::operations::examples {
ExampleDeviceOperation::SingleCore::cached_program_t ExampleDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores = num_cores_y * num_cores_x;
    auto all_cores = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size);

    for (auto cb_index : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_3}) {
        tt::tt_metal::CreateCircularBuffer(program, all_cores, tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_index, cb_data_format}})
                .set_page_size(cb_index, single_tile_size));
    }

    for (auto cb_index : {tt::CBIndex::c_4, tt::CBIndex::c_5, tt::CBIndex::c_6}) {
        tt::tt_metal::CreateCircularBuffer(program, all_cores, tt::tt_metal::CircularBufferConfig(single_tile_size_output, {{cb_index, cb_data_format_output}})
                .set_page_size(cb_index, single_tile_size_output));
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/reader_unary.cpp",
        all_cores, tt::tt_metal::ReaderDataMovementConfig());


    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/writer_unary.cpp",
        all_cores, tt::tt_metal::WriterDataMovementConfig());

    bool math_approx_mode = false;
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/eltwise_sfpu.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = math_approx_mode,
            .compile_args = {num_tiles},
        });

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {num_tiles, num_tiles_written});
        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {num_tiles, num_tiles_written});
        num_tiles_written += num_tiles;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void ExampleDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
}

}  // namespace ttnn::operations::examples
