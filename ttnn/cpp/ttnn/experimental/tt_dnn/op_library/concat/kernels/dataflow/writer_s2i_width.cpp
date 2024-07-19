// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "debug/dprint.h"

void kernel_main() {
	constexpr uint32_t num_tensors = get_compile_time_arg_val(0);
	constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

	const uint32_t dst_addr  = get_arg_val<uint32_t>(0);
	const uint32_t core_id = get_arg_val<uint32_t>(1);
	const uint32_t num_pages_per_core = get_arg_val<uint32_t>(2);
	const uint32_t stick_size = get_arg_val<uint32_t>(3);
	const uint32_t num_tensors_times_rows_per_shard = get_arg_val<uint32_t>(4);
	const uint32_t num_pages_per_tensor = get_arg_val<uint32_t>(5);

	uint32_t arg_index = 6;

	const InterleavedAddrGenFast<dst_is_dram> s = {
		.bank_base_address = dst_addr,
		.page_size = stick_size,
	};



	for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
		const uint32_t input_shard_cb = get_arg_val<uint32_t>(arg_index++);
		cb_wait_front(input_shard_cb, num_pages_per_tensor);
		uint32_t l1_read_addr = get_read_ptr(input_shard_cb);
		uint32_t page_id = 0;
		for(uint32_t page_id_input = 0; page_id_input < num_pages_per_tensor; page_id_input++) {
			uint32_t input_page_id = page_id +
				num_pages_per_core*core_id*num_tensors + tensor_id ;
			noc_async_write_tile(input_page_id, s, l1_read_addr);
			noc_async_write_barrier();
			l1_read_addr += stick_size;
			page_id+= num_tensors;
		}
		cb_pop_front(input_shard_cb, num_pages_per_tensor);
	}

}
