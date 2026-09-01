/* Shared-memory region for a detray detector: layout + mapping helpers.
 *
 * Layout of /dev/shm/<name>:
 *
 *   [ 0 .. HEADER_BYTES )   header (below), fixed size, never grows
 *   [ HEADER_BYTES .. )     payload — every byte the detray detector owns
 *
 * The payload is produced by allocating the detector THROUGH a
 * vecmem::contiguous_memory_resource sitting on this region, so all of
 * detray's containers land inside it, contiguous, in one shot.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace detray_shm {

inline constexpr std::uint64_t MAGIC = 0x4431'4445'5452'4159ULL;  // "D1DETRAY"
inline constexpr std::uint32_t FORMAT_VERSION = 1;
inline constexpr std::size_t HEADER_BYTES = 4096;

/* Fixed virtual address both processes map at.
 *
 * v0 deliberately sidesteps pointer relocation: if producer and consumer map
 * the region at the SAME address, every pointer detray stored inside it is
 * valid in both. That isolates the question "can the detector be built inside
 * one contiguous region at all" from "can views be rebuilt at a new base",
 * which is the follow-up experiment. Chosen high and page-aligned to stay out
 * of the way of the heap, the stack and any mmap'd .so.
 */
inline constexpr std::uintptr_t FIXED_BASE = 0x0000'2000'0000'0000ULL;

/* The detector's view object, memcpy'd verbatim.
 *
 * detray builds view_type on demand from its member containers; it is a small
 * POD-ish bundle of pointers and sizes. With a fixed base it is directly
 * reusable in the consumer. VIEW_BYTES is generous; the producer static_asserts
 * the real size fits.
 */
inline constexpr std::size_t VIEW_BYTES = 1024;

struct header {
    std::uint64_t magic;
    std::uint32_t format_version;
    std::uint32_t header_bytes;

    std::uint64_t base_addr;      // where the producer mapped it
    std::uint64_t region_bytes;   // total file size
    std::uint64_t payload_bytes;  // bytes actually handed out by the allocator

    // ABI: any mismatch must refuse the load.
    std::uint32_t detray_major, detray_minor, detray_patch;
    std::uint32_t vecmem_major, vecmem_minor, vecmem_patch;
    std::uint32_t view_bytes;     // sizeof(const_view_type) as the producer saw it
    std::uint32_t ready;          // 0 while writing, 1 when complete

    // Cheap structural check — the consumer must reproduce these exactly.
    std::uint64_t n_volumes;
    std::uint64_t n_surfaces;
    std::uint64_t n_transforms;

    unsigned char view[VIEW_BYTES];
};

static_assert(sizeof(header) <= HEADER_BYTES, "header outgrew its slot");

/* Create (producer) or open (consumer) the region, mapped at FIXED_BASE.
 * Returns the base pointer, or throws. `bytes` is ignored when create=false.
 */
void* map_region(const std::string& shm_name, std::size_t bytes, bool create);
void  unmap_region(void* base, std::size_t bytes);

/* Open the region once per PROCESS, returning the same base on every call.
 *
 * Triton runs `instance_group { count: N }` as N model instances inside ONE
 * process, and FIXED_BASE can only be claimed once per address space -- the
 * second instance's mmap would be refused by MAP_FIXED_NOREPLACE. Since the
 * detector is read-only after construction, all instances can share one
 * mapping. Never unmapped: the region must outlive every instance holding a
 * view into it, and the process exit reclaims it anyway.
 */
void* map_region_shared(const std::string& shm_name);

}  // namespace detray_shm
