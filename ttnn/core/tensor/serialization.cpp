#include "ttnn/tensor/serialization.hpp"

#include <ttnn/distributed/host_ccl.hpp>
#include "tt_stl/cleanup.hpp"

#include "tt-metalium/tensor/flatbuffer/tensor_flatbuffer.hpp"
#include <flatbuffers/flatbuffers.h>

namespace {

void safe_fwrite(const void* buffer, size_t size, size_t count, FILE* file) {
    TT_FATAL(fwrite(buffer, size, count, file) == count, "Failed to write tensor data: file write failed");
}

constexpr std::uint32_t kFlatbufferAlignment = alignof(std::uint64_t);

}  // namespace

namespace ttnn {

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor) {
    Tensor cpu_tensor = tensor.cpu();

    // Dump tensor to disk from (global) rank 0 host.
    // Note we use global context as opposed to context embedded to the host-side tensor, since the tensor may already
    // be fully host-local. In this latter case, host buffer context will consist of a single (local) host rank, and
    // each host will attempt to flush the serialized tensor file to disk.
    cpu_tensor = ttnn::distributed::host_ccl::all_gather(cpu_tensor);
    const auto& ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (ctx->rank() == tt::tt_metal::distributed::multihost::Rank(0)) {
        FILE* output_file = fopen(file_name.c_str(), "wb");
        TT_FATAL(output_file != nullptr, "Cannot open \"{}\"", file_name);
        auto cleanup = ttsl::make_cleanup([f = output_file]() {
            if (f && fclose(f) != 0) {
                log_warning(tt::LogAlways, "Failed to close file");
            }
        });

        std::vector<tt::tt_metal::HostBuffer> buffers;
        flatbuffers::FlatBufferBuilder builder;
        auto tensor_offset = tt::tt_metal::to_flatbuffer(cpu_tensor, builder, buffers);
        // To be able to read flatbuffer data with `mmap` safely, make sure the serialized flatbuffer is aligned to at
        // least 8 bytes, just like `header_size`. Individual `buffers` are aligned according to their element size,
        // which is already what we need for `mmap` to work.
        builder.Align(kFlatbufferAlignment);
        builder.Finish(tensor_offset);

        uint64_t header_size = builder.GetSize();
        safe_fwrite(&header_size, sizeof(header_size), 1, output_file);
        safe_fwrite(builder.GetBufferPointer(), header_size, 1, output_file);

        for (const auto& buffer : buffers) {
            auto buffer_view = buffer.view_bytes();
            safe_fwrite(buffer_view.data(), buffer_view.size(), 1, output_file);
        }
    }
    ctx->barrier();
}

}  // namespace ttnn
