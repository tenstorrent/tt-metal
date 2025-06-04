#include <stdint.h>

void kernel_main() {
    const uint32_t num_pages_per_core = get_arg_val<uint32_t>(0);
    const uint32_t page_start = get_arg_val<uint32_t>(1);
    const uint32_t metadata_buffer_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr bool rm_layout = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t ublock_size_pages = 1;

    // READ METADATA from buffer address
    volatile tt_l1_ptr uint32_t* metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_buffer_addr);

    // Read header
    const uint32_t num_tensors = metadata_ptr[0];
    const uint32_t num_output_pages_per_block = metadata_ptr[1];
    const uint32_t concat_dim = metadata_ptr[2];
    const uint32_t is_rm_layout = metadata_ptr[3];

    // Process pages assigned to this core
    const uint32_t page_end = page_start + num_pages_per_core;

    for (uint32_t page = page_start; page < page_end; ++page) {
        // Figure out which block and position within block
        uint32_t block_id = page / num_output_pages_per_block;
        uint32_t pos_in_block = page % num_output_pages_per_block;

        // Find tensor and page within tensor
        uint32_t tensor_id = 0;
        uint32_t pages_before_tensor = 0;

        for (uint32_t t = 0; t < num_tensors; ++t) {
            // Each tensor metadata starts at index 4 + t*5
            uint32_t tensor_metadata_start = 4 + t * 5;
            uint32_t pages_per_block = metadata_ptr[tensor_metadata_start + 3];

            if (pos_in_block < pages_before_tensor + pages_per_block) {
                tensor_id = t;
                break;
            }
            pages_before_tensor += pages_per_block;
        }

        // Read tensor metadata
        uint32_t tensor_metadata_start = 4 + tensor_id * 5;
        uint32_t src_addr = metadata_ptr[tensor_metadata_start + 0];
        uint32_t is_dram = metadata_ptr[tensor_metadata_start + 1];
        uint32_t page_size = metadata_ptr[tensor_metadata_start + 2];
        uint32_t pages_per_block = metadata_ptr[tensor_metadata_start + 3];

        // Calculate page within tensor
        uint32_t page_in_tensor = block_id * pages_per_block + (pos_in_block - pages_before_tensor);

        // Read page from the identified tensor
        cb_reserve_back(cb_id_in, ublock_size_pages);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);

        if (is_dram) {
            InterleavedAddrGen<true> addr_gen = {.bank_base_address = src_addr, .page_size = page_size};
            noc_async_read_page(page_in_tensor, addr_gen, l1_write_addr);
        } else {
            InterleavedAddrGen<false> addr_gen = {.bank_base_address = src_addr, .page_size = page_size};
            noc_async_read_page(page_in_tensor, addr_gen, l1_write_addr);
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_in, ublock_size_pages);
    }
}
