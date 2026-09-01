/* A vecmem::memory_resource that hands out a pre-mapped region.
 *
 * This is the whole trick of the plan, in ~30 lines: detray allocates through
 * whatever vecmem resource it is given, so if that resource returns pointers
 * inside an mmap'd /dev/shm region, the detector's data lands in shared memory
 * with no serialization step anywhere.
 *
 * Deliberately dumb: it is a bump allocator and never frees. That is correct
 * here -- the detector is built once, read-only afterwards, and the region is
 * discarded whole. Wrapping it in vecmem::contiguous_memory_resource is what
 * keeps detray's many small allocations packed and in-region.
 */
#pragma once

#include <vecmem/memory/memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace detray_shm {

class shm_memory_resource final : public vecmem::memory_resource {
    public:
    shm_memory_resource(void* base, std::size_t capacity)
        : m_base(static_cast<std::byte*>(base)),
          m_capacity(capacity),
          m_used(0) {}

    /// Bytes actually handed out -- this is what the header records.
    std::size_t used() const { return m_used; }
    std::size_t capacity() const { return m_capacity; }

    private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        const std::size_t aligned = (m_used + alignment - 1) & ~(alignment - 1);
        if (aligned + bytes > m_capacity) {
            throw std::bad_alloc();
        }
        void* p = m_base + aligned;
        m_used = aligned + bytes;
        return p;
    }

    void do_deallocate(void*, std::size_t, std::size_t) override {
        // Intentionally empty: bump allocator, freed as a whole region.
    }

    bool do_is_equal(const vecmem::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::byte* m_base;
    std::size_t m_capacity;
    std::size_t m_used;
};

}  // namespace detray_shm
