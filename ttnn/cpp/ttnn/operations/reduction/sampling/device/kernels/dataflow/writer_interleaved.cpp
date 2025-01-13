#include <random>

// // Function to generate a random number between 0 and 1
// double random_uniform() {
//     static std::random_device rd;
//     static std::mt19937 gen(rd());
//     static std::uniform_real_distribution<> dis(0.0, 1.0);
//     return dis(gen);
// }

// // Custom binary search to find the first index where value > target
// int binary_search_first_greater(const std::vector<double>& values, int start, int end, double target) {
//     int low = start;
//     int high = end - 1;  // end is exclusive
//     while (low <= high) {
//         int mid = low + (high - low) / 2;  // Prevents overflow
//         if (values[mid] > target) {
//             high = mid - 1;
//         } else {
//             low = mid + 1;
//         }
//     }
//     return low;  // The first index where value > target
// }

// // Combined binary search to find the first index where value > r within the range defined by `p`
// int combined_binary_search(const std::vector<double>& values, double p, double r) {
//     int low = 0;
//     int high = values.size() - 1;

//     while (low <= high) {
//         int mid = low + (high - low) / 2;

//         // Check if the current value exceeds `p` (cutoff condition)
//         if (values[mid] >= p) {
//             high = mid - 1;  // Restrict to the left of `p` range
//         } else if (values[mid] <= r) {
//             low = mid + 1;  // Search for values greater than `r`
//         } else {
//             // Found the first value satisfying value > r within the range defined by `p`
//             return mid;
//         }
//     }
//     return low;  // If no exact match, `low` points to the first valid index
// }

// // Function to sample from the top-k values
// int sample_k(const std::vector<double>& values, const std::vector<int>& indices, int k) {
//     double r = random_uniform();  // Random value between 0 and 1

//     // Perform binary search in values[:k]
//     int index = binary_search_first_greater(values, 0, k, r);
//     return indices[index];
// }

// // Function to sample from the top-p values
// int sample_p(const std::vector<double>& values, const std::vector<int>& indices, double p) {
//     double r = random_uniform();  // Random value between 0 and 1

//     // Perform the combined binary search
//     int index = combined_binary_search(values, p, r);
//     return indices[index];
// }

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // constexpr uint32_t cb_id_values = get_compile_time_arg_val(0);
    // constexpr uint32_t cb_id_im_indices = get_compile_time_arg_val(1);
    // constexpr uint32_t cb_id_final_indices = get_compile_time_arg_val(2);
    // constexpr uint32_t values_stick_size = get_compile_time_arg_val(6);
    // constexpr uint32_t im_indices_stick_size = get_compile_time_arg_val(7);
    // constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(8);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t k = get_compile_time_arg_val(2);
    uint32_t core_id = get_arg_val<uint32_t>(3);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(4);

    // constexpr uint32_t p = get_compile_time_arg_val(10);
    // constexpr uint32_t ids_per_batch = get_compile_time_arg_val(11);

    // Use cb as L1 scratch memory
    // uint32_t cb_id_values_addr = get_write_ptr(cb_id_values);
    // volatile tt_l1_ptr uint16_t* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_id_values_addr);

    // uint32_t cb_id_im_indices_addr = get_write_ptr(cb_id_im_indices);
    // volatile tt_l1_ptr uint32_t* im_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_id_im_indices_addr);

    // uint32_t cb_id_final_indices_addr = get_write_ptr(cb_id_final_indices);
    // volatile tt_l1_ptr uint32_t* final_indices =
    //     reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_id_final_indices_addr);

    uint32_t out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint32_t* index_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    // uint32_t start_id = core_id * ids_per_batch;
    // uint32_t end_id = start_id + ids_per_batch;

    // if (p == 0) {
    //     // Sample from the top-k values
    //     int index = sample_k(values, im_indices, start_id, end_id, k);
    //     index_out[core_id] = final_indices[index];
    // } else {
    //     // Sample from the top-p values
    //     int index = sample_p(values, im_indices, start_id, end_id, p);
    //     index_out[core_id] = final_indices[index];
    // }
    index_out[core_id] = core_id;

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    noc_async_write(out_addr, dst_noc_addr, out_stick_size);
    noc_async_write_barrier();
}
