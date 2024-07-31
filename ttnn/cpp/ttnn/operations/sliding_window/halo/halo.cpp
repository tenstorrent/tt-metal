#include "halo.hpp"
namespace ttnn::operations::sliding_window::halo
{
    static Tensor HaloOperation::operator()(
                uint8_t queue_id,
                const Tensor& input_tensor,
                const SlidingWindowConfig& config,
                uint32_t pad_val,
                bool remote_read,
                bool transpose_mcast,
                uint32_t reshard_num_cores_nhw,
                MemoryConfig output_memory_config)
    {
        return halo_op(
            input_tensor,
            config,
            pad_val,
            remote_read,
            transpose_mcast,
            reshard_num_cores_nhw,
            output_memory_config);

    }
};
