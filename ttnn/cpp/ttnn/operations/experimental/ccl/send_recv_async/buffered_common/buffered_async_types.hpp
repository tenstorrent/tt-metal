#define MAX_OUTPUT_TENSORS 3

struct alignas(16) OutputTensorInfo {
    uint32_t num_tensors;
    uint32_t page_size;
    uint32_t num_pages;
    uint32_t write_index;
    uint32_t read_index;
    uint32_t base_addr[MAX_OUTPUT_TENSORS];
};
