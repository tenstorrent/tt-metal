#define MAX_OUTPUT_TENSORS 3
#define UINT32_ALIGNED_COUNT 4
struct alignas(32) OutputTensorInfo {
    uint32_t num_tensors;
    uint32_t page_size;
    uint32_t num_pages;
    uint32_t sender_config_l1_addr;
    uint32_t receiver_config_l1_addr;
    uint32_t base_addr[MAX_OUTPUT_TENSORS];

    // base_addr is already aligned to 8 * 4 = 32 bytes.
    uint32_t write_index[UINT32_ALIGNED_COUNT];
    uint32_t read_index[UINT32_ALIGNED_COUNT];
};
