// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/bcast/device/bcast_device_operation.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt;
using namespace tt::tt_metal;
using namespace	tt::constants;



namespace ttnn::operations::data_movement {
operation::ProgramWithCallbacks bcast_multi_core_h(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math) {

    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

	uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = a.device();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Ht);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
	TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

	uint32_t src0_cb_index = 0;
	uint32_t num_input_tiles = 2;

	tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

	uint32_t src1_cb_index = 1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
		.set_page_size(src1_cb_index, src1_single_tile_size);
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
	auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);

	bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

	KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
		program,
		"ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_interleaved_input_rows_partitioned.cpp",
		all_device_cores,
		tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

	KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
		program,
		"ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/writer_unary_interleaved_input_cols_batched.cpp",
		all_device_cores,
		tt_metal::WriterDataMovementConfig(writer_compile_time_args));

	std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(BcastOpDim::H, bcast_math);
	auto bcast_kernel_id = tt_metal::CreateKernel(
		program,
		"ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h.cpp",
		all_device_cores,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_defines}
	);

	for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t Ht_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_2;
		} else {
			tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(15, 0));
			tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
			tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
			continue;
		}
		uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

		tt_metal::SetRuntimeArgs(
			program,
			binary_reader_kernel_id,
			core,
			{
				a.buffer()->address(), // 0
				0, // 1
				0, // 2
				num_tensor_tiles_per_core, // 3
				b.buffer()->address(), // 4
				0, // 5
				0, // 6
				num_btensor_tiles, // 7
				num_tensor_tiles_per_core, // 8
				NC, // 9
				Ht_per_core, // 10
				Wt, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
			}
		);

		tt_metal::SetRuntimeArgs(
			program,
			bcast_kernel_id,
			core,
			{
				NC, // B
				Ht_per_core, // Ht
				Wt  // Wt
			}
		);

		tt_metal::SetRuntimeArgs(
			program, unary_writer_kernel_id, core,
			{
				output.buffer()->address(),
				0,
				0,
				Ht_per_core,
				Wt,
				num_Wtiles_read,
				0,
				NC,
				Ht*Wt,
			}
		);

		num_Wtiles_read += Ht_per_core * Wt;
	}

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
			bcast_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
		uint32_t num_cores_x = compute_with_storage_grid_size.x;
		uint32_t num_cores_y = compute_with_storage_grid_size.y;

        auto src_dram_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

		const auto ashape = input_tensors.at(0).get_legacy_shape();
		const auto bshape = input_tensors.at(1).get_legacy_shape();
        uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
        uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
        uint32_t H = ashape[-2];
        uint32_t W = ashape[-1];
        uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
        uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
        uint32_t bH = bshape[-2];
        uint32_t bW = bshape[-1];
        uint32_t NC = N * C;
        uint32_t HW = H * W;

        uint32_t Wt = W/TILE_WIDTH;
		uint32_t Ht = H/TILE_HEIGHT;

		uint32_t num_tensor_tiles = NC*Ht*Wt;
		uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

		uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

		auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Ht);

		for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++){
			CoreCoord core = {i / num_cores_y, i % num_cores_y};
			uint32_t Ht_per_core;
			if (core_group_1.core_coord_in_core_ranges(core)) {
				Ht_per_core = Ht_per_core_group_1;
			} else if (core_group_2.core_coord_in_core_ranges(core)) {
				Ht_per_core = Ht_per_core_group_2;
			} else {
				tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(15, 0));
				tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
				tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
				continue;
			}
			uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

			tt_metal::SetRuntimeArgs(
				program,
				binary_reader_kernel_id,
				core,
				{
					src_dram_buffer_a->address(), // 0
					0, // 1
					0, // 2
					num_tensor_tiles_per_core, // 3
					src_dram_buffer_b->address(), // 4
					0, // 5
					0, // 6
					num_btensor_tiles, // 7
					num_tensor_tiles_per_core, // 8
					NC, // 9
					Ht_per_core, // 10
					Wt, // 11
					bnc1, // 12
					num_Wtiles_read, // 13
					Ht*Wt, // 14
				}
			);

			tt_metal::SetRuntimeArgs(
				program,
				bcast_kernel_id,
				core,
				{
					NC, // B
					Ht_per_core, // Ht
					Wt  // Wt
				}
			);

			tt_metal::SetRuntimeArgs(
				program, unary_writer_kernel_id, core,
				{
					dst_dram_buffer->address(),
					0,
					0,
					Ht_per_core,
					Wt,
					num_Wtiles_read,
					0,
					NC,
					Ht*Wt,
				}
			);

			num_Wtiles_read += Ht_per_core * Wt;
		}
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

} // ttnn::operations::data_movement
