// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
void kernel_main() {
	constexpr uint32_t input_cb = get_compile_time_arg_val(0);
	constexpr uint32_t output_shard_cb = get_compile_time_arg_val(1);
	const uint32_t num_pages = get_arg_val<uint32_t>(0);
	const uint32_t total_data_size = get_arg_val<uint32_t>(1);

	uint32_t l1_write_addr = get_write_ptr(output_shard_cb);
	uint32_t tensor_id_offset = 0;
	cb_wait_front(input_cb, num_pages);
	uint32_t l1_read_addr = get_read_ptr(input_cb);
	noc_async_read(get_noc_addr(l1_read_addr), l1_write_addr, total_data_size);
	noc_async_read_barrier();
	cb_pop_front(input_cb, num_pages);
}
