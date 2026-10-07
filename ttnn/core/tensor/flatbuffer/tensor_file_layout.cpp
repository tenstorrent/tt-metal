// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/tensor_file_layout.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>

#include <fmt/format.h>

#include <tt_stl/assert.hpp>
#include <tt_stl/cleanup.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn {
namespace {

void safe_fwrite_bytes(const void* buffer, size_t bytes, FILE* file, std::string_view filename, std::string_view what) {
    TT_FATAL(bytes > 0, "Expected to write > 0 bytes to file");

    // Use byte-wise fwrite so we can detect partial writes
    const size_t written = fwrite(buffer, /*size=*/1, /*count=*/bytes, file);
    TT_FATAL(
        written == bytes,
        "Failed to write {} to \"{}\": wrote {}/{} bytes (ferror={}, errno={} \"{}\")",
        what,
        filename,
        written,
        bytes,
        ferror(file),
        errno,
        strerror(errno));
}

// Zero-fills `bytes` bytes. Aligning to `kTensorDataAlignment` never needs more than that many bytes of padding.
void write_padding(size_t bytes, FILE* file, std::string_view filename) {
    if (bytes == 0) {
        return;
    }
    static constexpr std::array<std::byte, kTensorDataAlignment> zeros{};
    TT_FATAL(bytes < zeros.size(), "Padding of {} bytes exceeds the {}-byte alignment", bytes, zeros.size());
    safe_fwrite_bytes(zeros.data(), bytes, file, filename, "padding");
}

void write_tensor_contents(
    FILE* file,
    std::string_view file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    ttsl::Span<const SerializedTensorBuffer> buffers) {
    // Pad the header so the data region starts on `kTensorDataAlignment`. The padding is counted in `header_size`,
    // so a reader that only knows about the 8-byte minimum still finds the data region where it expects it.
    const uint64_t flatbuffer_size = builder.GetSize();
    const uint64_t header_size = tt::align(sizeof(uint64_t) + flatbuffer_size, kTensorDataAlignment) - sizeof(uint64_t);

    safe_fwrite_bytes(&header_size, sizeof(header_size), file, file_name, "tensor header size");
    safe_fwrite_bytes(builder.GetBufferPointer(), flatbuffer_size, file, file_name, "tensor header");
    write_padding(header_size - flatbuffer_size, file, file_name);

    // Offset, relative to the start of the data region, one past the last buffer written so far.
    uint64_t data_region_end = 0;
    for (const auto& [buffer, offset] : buffers) {
        auto buffer_view = buffer.view_bytes();
        TT_FATAL(!buffer_view.empty(), "Unexpected empty buffer during tensor serialization");
        TT_FATAL(
            offset >= data_region_end,
            "Shard buffer offset {} overlaps the preceding buffer, which ends at {}",
            offset,
            data_region_end);
        write_padding(offset - data_region_end, file, file_name);
        safe_fwrite_bytes(buffer_view.data(), buffer_view.size(), file, file_name, "tensor data");
        data_region_end = offset + buffer_view.size();
    }
}

// Distinguishes temporary files written by different threads of one process.
uint64_t next_temp_file_id() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// Creates a new file in the directory of `path`, named `<path>.tmp.<pid>.<n>`, and returns its descriptor and name.
//
// The file is created exclusively: containers sharing one cache can share a pid, so the name alone is not unique. It
// takes the permissions of `path` when `path` exists, so replacing a restricted cache file does not widen access, and
// is otherwise created 0666 under the umask, as fopen would create `path`. A writer killed outright leaves this file
// behind; its name does not end in .tensorbin, so cache listings that glob for tensorbins never pick it up.
std::pair<int, std::string> create_sibling_file(const std::string& path) {
    mode_t mode = 0666;
    struct stat path_stat{};
    if (stat(path.c_str(), &path_stat) == 0) {
        mode = path_stat.st_mode & 07777;
    }
    while (true) {
        std::string sibling = fmt::format("{}.tmp.{}.{}", path, getpid(), next_temp_file_id());
        const int fd = open(sibling.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode);
        if (fd != -1) {
            return {fd, std::move(sibling)};
        }
        TT_FATAL(errno == EEXIST, "Cannot create \"{}\" for writing: errno={} \"{}\"", sibling, errno, strerror(errno));
    }
}

}  // namespace

void write_tensor_file(
    const std::string& file_name,
    const flatbuffers::FlatBufferBuilder& builder,
    ttsl::Span<const SerializedTensorBuffer> buffers) {
    // Resolve symlinks, so the rename below replaces the file a link points to rather than the link. A path that does
    // not resolve, such as a file that does not exist yet, is written as given.
    std::error_code resolve_error;
    std::filesystem::path target = std::filesystem::canonical(file_name, resolve_error);
    if (resolve_error) {
        target = file_name;
    }

    auto [fd, temp_name] = create_sibling_file(target.string());
    auto remove_temp_file = ttsl::make_cleanup([&temp_name = temp_name]() { unlink(temp_name.c_str()); });

    auto close_file = [](FILE* file) { fclose(file); };
    std::unique_ptr<FILE, decltype(close_file)> file(fdopen(fd, "wb"));
    if (file == nullptr) {
        const int fdopen_errno = errno;
        close(fd);
        TT_THROW("Cannot open \"{}\" for writing: errno={} \"{}\"", temp_name, fdopen_errno, strerror(fdopen_errno));
    }

    write_tensor_contents(file.get(), file_name, builder, buffers);
    // Close before publishing: a deferred write error (ENOSPC, a network file system) surfaces at fclose, and only a
    // fully written file may be renamed into place.
    TT_FATAL(fclose(file.release()) == 0, "Failed to write \"{}\": errno={} \"{}\"", temp_name, errno, strerror(errno));
    TT_FATAL(
        rename(temp_name.c_str(), target.c_str()) == 0,
        "Failed to rename \"{}\" to \"{}\": errno={} \"{}\"",
        temp_name,
        target.string(),
        errno,
        strerror(errno));
    std::move(remove_temp_file).cancel();
}

std::shared_ptr<void> map_tensor_file(int fd, size_t file_size, std::string_view file_name) {
    void* mapping = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        const int shared_errno = errno;
        mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        TT_FATAL(
            mapping != MAP_FAILED,
            "Failed to mmap file \"{}\": MAP_SHARED failed with \"{}\", MAP_PRIVATE with \"{}\"",
            file_name,
            strerror(shared_errno),
            strerror(errno));
        // Once per process: a model loads many tensor files from one filesystem, and they all fall back alike.
        static std::once_flag private_mapping_warned;
        std::call_once(private_mapping_warned, [&] {
            log_warning(
                tt::LogMetal,
                "Mapped \"{}\" MAP_PRIVATE because MAP_SHARED failed with \"{}\". Uploading a tensor loaded from a "
                "privately mapped file through pinned memory first copies every page of it into anonymous memory, "
                "which is slow and doubles its resident memory. Reported for the first such file only.",
                file_name,
                strerror(shared_errno));
        });
    }
    return std::shared_ptr<void>(mapping, [file_size](void* addr) { munmap(addr, file_size); });
}

}  // namespace ttnn
