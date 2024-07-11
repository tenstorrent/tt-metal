// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/memory/memory.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"

namespace tt
{
    namespace tt_metal
    {
        Tensor read_tensor_from_L1(
            uint64_t addr,
            CoreCoord core,
            uint64_t size,
            DataType dtype,
            Device* device
        )
        {
            DeviceBuffer dev_buffer(new Buffer(device,size*2,size*2,BufferType::L1));
            dev_buffer->set_address(addr);
            return Tensor(DeviceStorage(dev_buffer),{size},dtype,Layout::ROW_MAJOR);
        }
        void print_tensor_info(Tensor &tensor)
        {
            auto storage = std::get<DeviceStorage>(tensor.get_storage());
            auto buffer = *storage.get_buffer();
            printf("Storage Addr :%d, Size : %d, Page_size :%d, Buffer Type :%d \n",buffer.address(),buffer.size(),buffer.page_size(),static_cast<int>(buffer.buffer_type()));
        }
    }
}
