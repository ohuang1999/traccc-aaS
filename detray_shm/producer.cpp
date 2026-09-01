/* D3 producer -- build the ITk detray detector directly inside /dev/shm, for
 * the Triton backend to adopt.
 *
 * Stands in for Athena, which will eventually do this itself (phase D4). The
 * only difference from the backend's own initialize() is the memory resource:
 * instead of the default host resource, read_detector allocates through a
 * contiguous resource sitting on a shared-memory region, so the detector is
 * built in shared memory rather than copied there afterwards.
 *
 * shm_region.hpp is deliberately NOT a local copy -- it is taken from
 * ../../traccc-aaS/standalone/src/, the same header the backend compiles
 * against. Producer and consumer must agree on the header layout exactly; two
 * copies that drift would disagree silently.
 *
 *   ./d3_producer <geo_dir> [shm_name] [region_GB]
 */
#include "shm_memory_resource.hpp"
#include "shm_region.hpp"

#include <traccc/geometry/detector.hpp>
#include <traccc/geometry/host_detector.hpp>
#include <traccc/io/read_detector.hpp>

#include <detray/version.hpp>
#include <vecmem/memory/contiguous_memory_resource.hpp>
#include <vecmem/version.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

namespace {

using itk_host = traccc::itk_detector::host;

template <typename clock_t>
double ms_since(const clock_t& t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <geo_dir> [shm_name=/d3_itk_detector] [region_GB=2]\n";
        return 2;
    }
    const std::string geo_dir = argv[1];
    const std::string shm_name = (argc > 2) ? argv[2] : "/d3_itk_detector";
    const std::size_t region_gb = (argc > 3) ? std::stoul(argv[3]) : 2;
    const std::size_t region_bytes = region_gb * (1UL << 30);

    const std::string geometry_file = geo_dir + "/detray_detector_geometry.json";
    const std::string material_file = geo_dir + "/detray_detector_material_maps.json";
    const std::string grid_file     = geo_dir + "/detray_detector_surface_grids.json";

    std::cout << "region : " << shm_name << "  (" << region_gb << " GB)\n"
              << "geometry: " << geometry_file << "\n";

    void* base = detray_shm::map_region(shm_name, region_bytes, /*create=*/true);
    std::cout << "mapped at " << base << "\n";

    // Payload starts after the header slot.
    std::byte* payload = static_cast<std::byte*>(base) + detray_shm::HEADER_BYTES;
    const std::size_t payload_capacity = region_bytes - detray_shm::HEADER_BYTES;

    detray_shm::shm_memory_resource shm_mr{payload, payload_capacity};
    vecmem::contiguous_memory_resource cmr{shm_mr, payload_capacity};

    // ---- the one line that differs from the backend: `cmr`, not a host mr ----
    const auto t0 = std::chrono::high_resolution_clock::now();
    traccc::host_detector detector;
    traccc::io::read_detector(detector, cmr, geometry_file, material_file,
                              grid_file);
    const double parse_ms = ms_since(t0);
    std::cout << "read_detector: " << parse_ms << " ms  (this is the cost the "
                 "plan removes)\n";

    if (!detector.is<traccc::itk_detector>()) {
        std::cerr << "FATAL: not an ITk detector -- got "
                  << detector.type().name() << "\n";
        return 3;
    }
    const itk_host& det = detector.as<traccc::itk_detector>();

    const auto view = detray::get_data(det);
    using view_t = std::decay_t<decltype(view)>;
    static_assert(std::is_trivially_copyable_v<view_t>,
                  "view is not trivially copyable -- v0's memcpy shortcut is "
                  "invalid, go straight to offset-relocated views");
    static_assert(sizeof(view_t) <= detray_shm::VIEW_BYTES, "raise VIEW_BYTES");

    auto* hdr = static_cast<detray_shm::header*>(base);
    std::memset(hdr, 0, sizeof(detray_shm::header));
    hdr->magic          = detray_shm::MAGIC;
    hdr->format_version = detray_shm::FORMAT_VERSION;
    hdr->header_bytes   = static_cast<std::uint32_t>(detray_shm::HEADER_BYTES);
    hdr->base_addr      = reinterpret_cast<std::uint64_t>(base);
    hdr->region_bytes   = region_bytes;
    hdr->payload_bytes  = shm_mr.used();
    hdr->detray_major   = DETRAY_VERSION_MAJOR;
    hdr->detray_minor   = DETRAY_VERSION_MINOR;
    hdr->detray_patch   = DETRAY_VERSION_PATCH;
    hdr->vecmem_major   = VECMEM_VERSION_MAJOR;
    hdr->vecmem_minor   = VECMEM_VERSION_MINOR;
    hdr->vecmem_patch   = VECMEM_VERSION_PATCH;
    hdr->view_bytes     = static_cast<std::uint32_t>(sizeof(view_t));
    hdr->n_volumes      = det.volumes().size();
    hdr->n_surfaces     = det.surfaces().size();
    hdr->n_transforms   = det.transform_store().size();
    std::memcpy(hdr->view, &view, sizeof(view_t));

    // Publish last: a consumer that sees ready==1 sees a complete header.
    __atomic_store_n(&hdr->ready, 1u, __ATOMIC_RELEASE);

    std::cout << "\n=== produced ===\n"
              << "  payload      : " << (hdr->payload_bytes / (1024.0 * 1024.0))
              << " MB of " << region_gb << " GB\n"
              << "  volumes      : " << hdr->n_volumes << "\n"
              << "  surfaces     : " << hdr->n_surfaces << "\n"
              << "  transforms   : " << hdr->n_transforms << "\n"
              << "  view bytes   : " << hdr->view_bytes << "\n"
              << "  detray       : " << DETRAY_VERSION << "\n"
              << "  vecmem       : " << VECMEM_VERSION << "\n\n"
              << "region stays in /dev/shm after exit. Now start the server:\n"
              << "  SHM=" << shm_name << " ./run_server.sh\n";

    // NOTE: deliberately NOT unmapped/unlinked -- the region must outlive us.
    return 0;
}
