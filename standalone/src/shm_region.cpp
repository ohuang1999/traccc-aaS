#include "shm_region.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace detray_shm {

void* map_region(const std::string& shm_name, std::size_t bytes, bool create) {
    const int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
    const int fd = ::shm_open(shm_name.c_str(), flags, 0600);
    if (fd < 0) {
        throw std::runtime_error("shm_open(" + shm_name +
                                 ") failed: " + std::strerror(errno));
    }

    if (create) {
        if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            ::close(fd);
            throw std::runtime_error(std::string("ftruncate failed: ") +
                                     std::strerror(errno));
        }
    } else {
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed");
        }
        bytes = static_cast<std::size_t>(st.st_size);
    }

    /* MAP_FIXED_NOREPLACE: fail loudly rather than silently clobbering an
     * existing mapping, which MAP_FIXED would do. A collision here is the
     * signal to move to offset-relocated views. */
    void* base = ::mmap(reinterpret_cast<void*>(FIXED_BASE), bytes,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
    ::close(fd);

    if (base == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap at FIXED_BASE failed: ") +
                                 std::strerror(errno));
    }
    if (reinterpret_cast<std::uintptr_t>(base) != FIXED_BASE) {
        ::munmap(base, bytes);
        throw std::runtime_error(
            "kernel placed the mapping elsewhere; FIXED_BASE is unusable");
    }
    return base;
}

void* map_region_shared(const std::string& shm_name) {
    static std::mutex mtx;
    static std::map<std::string, void*> mapped;

    const std::lock_guard<std::mutex> lock(mtx);
    const auto it = mapped.find(shm_name);
    if (it != mapped.end()) {
        return it->second;
    }
    void* base = map_region(shm_name, 0, /*create=*/false);
    mapped.emplace(shm_name, base);
    return base;
}

void unmap_region(void* base, std::size_t bytes) {
    if (base) {
        ::munmap(base, bytes);
    }
}

}  // namespace detray_shm
