#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>
#include <chrono>
#include <unordered_map>

// CUDA include(s).
#include <cuda_runtime.h>

// local includes
#include "TracccEdmConversion.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/seeding/seed_parameter_estimation_algorithm.hpp"
#include "traccc/cuda/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/triplet_seeding_algorithm.hpp"
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#include "traccc/cuda/utils/stream_wrapper.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/geometry/detector_conditions_description.hpp"
#include "traccc/geometry/detector_design_description.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/utils/algorithm.hpp"

// magnetic field include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/io/read_magnetic_field.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/io/frontend/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/stream_wrapper.hpp>

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/geometry/detector_buffer.hpp"

// Detray-in-shared-memory (sandbox_detray_shm phase D3)
#include "shm_region.hpp"
#include <detray/core/detail/container_buffers.hpp>
#include <detray/version.hpp>
#include <vecmem/version.hpp>
#include <cstdlib>
#include <cstring>
#include "traccc/io/read_detector_description.hpp"

// algorithm options
#include "traccc/options/detector.hpp"

// Set the CUDA device to use, and hand the ID back for stream construction.
static int setCudaDevice(int deviceID)
{
    cudaError_t err = cudaSetDevice(deviceID);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device: \
                " + std::string(cudaGetErrorString(err)));
    }
    return deviceID;
}

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

std::chrono::high_resolution_clock::time_point start_total, end_total;
std::chrono::high_resolution_clock::time_point start_read, end_read;
std::chrono::high_resolution_clock::time_point start_copy_in, end_copy_in;
std::chrono::high_resolution_clock::time_point start_cluster, end_cluster;
std::chrono::high_resolution_clock::time_point start_spacepoint, end_spacepoint;
std::chrono::high_resolution_clock::time_point start_seeding, end_seeding;
std::chrono::high_resolution_clock::time_point start_params, end_params;
std::chrono::high_resolution_clock::time_point start_finding, end_finding;
std::chrono::high_resolution_clock::time_point start_fitting, end_fitting;
std::chrono::high_resolution_clock::time_point start_copy_out, end_copy_out;

struct cell_order {
    bool operator()(const traccc::io::csv::cell& lhs,
                    const traccc::io::csv::cell& rhs) const {
        if (lhs.channel1 != rhs.channel1) {
            return (lhs.channel1 < rhs.channel1);
        } else {
            return (lhs.channel0 < rhs.channel0);
        }
    }
};  // struct cell_order

struct TracccResults {
    traccc::edm::track_container<traccc::default_algebra>::host tracks_and_states;
    traccc::edm::measurement_collection::host measurements;
};

traccc::magnetic_field make_magnetic_field(std::string filename) {
    traccc::magnetic_field result;
    traccc::io::read_magnetic_field(result, filename, traccc::data_format::binary);
    return result;
}

// Type definitions
using host_detector_type = traccc::host_detector;
using device_detector_type = traccc::detector_buffer;
using scalar_type = traccc::default_detector::host::scalar_type;

using spacepoint_formation_algorithm =
    traccc::cuda::silicon_pixel_spacepoint_formation_algorithm;
using clustering_algorithm = traccc::cuda::clusterization_algorithm;
using seeding_algorithm = traccc::cuda::triplet_seeding_algorithm;
using track_params_estimation_algorithm =
    traccc::cuda::seed_parameter_estimation_algorithm;
using finding_algorithm =
    traccc::cuda::combinatorial_kalman_filter_algorithm;

template <typename scalar_t>
using unit = detray::unit<scalar_t>;

static traccc::seedfinder_config create_and_setup_finder_config() {
    traccc::seedfinder_config cfg;
    cfg.zMin = -3000.f * traccc::unit<float>::mm;
    cfg.zMax = 3000.f * traccc::unit<float>::mm;
    cfg.rMax = 320.f * traccc::unit<float>::mm;
    cfg.rMin = 33.f * traccc::unit<float>::mm;
    // 3 sigmas of max beam spot delta z for Run 3
    cfg.collisionRegionMax = 3 * 38 * traccc::unit<float>::mm;
    cfg.collisionRegionMin = -cfg.collisionRegionMax;
    // Based on Pixel barrel layers layout
    // Minimum R distance between 2 layers: 30
    // Max distance between N, N+2 layer (allowing one "hole"): 160
    // Plus some margin (2 mm)
    cfg.deltaRMin = 20.f * traccc::unit<float>::mm;
    cfg.deltaRMax = 100.f * traccc::unit<float>::mm;
    cfg.deltaZMax = 800.f * traccc::unit<float>::mm;
    cfg.minPt = 900.f * traccc::unit<float>::MeV;
    cfg.cotThetaMax = 27.2899f;
    cfg.impactMax = 2.f * traccc::unit<float>::mm;
    cfg.sigmaScattering = 3.0f;
    cfg.maxPtScattering = 10.f * traccc::unit<float>::GeV;
    cfg.radLengthPerSeed = 0.05f;
    cfg.maxSeedsPerSpM = 2;
    cfg.setup();
    return cfg;
}

static traccc::seedfilter_config create_and_setup_filter_config() {
    traccc::seedfilter_config cfg;
    cfg.good_spB_min_radius = 150.f * unit<float>::mm;
    cfg.good_spB_weight_increase = 400.f;
    cfg.good_spT_max_radius = 150.f * unit<float>::mm;
    cfg.good_spT_weight_increase = 200.f;
    cfg.good_spB_min_weight = 380.f;
    cfg.seed_min_weight = 200.f;
    cfg.spB_min_radius = 43.f * unit<float>::mm;
    cfg.compatSeedLimit = 1;
    return cfg;
}

// Assign field-by-field rather than using an aggregate initializer, so that
// fields added by future traccc releases keep their upstream defaults instead
// of tripping -Werror=missing-field-initializers.
static finding_algorithm::config_type create_and_setup_finding_config() {
    finding_algorithm::config_type cfg{};
    cfg.max_num_branches_per_seed = 3;
    cfg.max_num_branches_per_surface = 1;
    cfg.min_track_candidates_per_track = 7;
    cfg.max_track_candidates_per_track = 20;
    cfg.max_num_skipping_per_cand = 2;
    cfg.max_num_consecutive_skipped = 1;
    cfg.max_num_tracks_per_measurement = 1;
    cfg.min_step_length_for_next_surface = 0.5f * detray::unit<float>::mm;
    cfg.max_step_counts_for_next_surface = 100;
    cfg.chi2_max = 30.f;

    cfg.propagation.navigation.intersection.overstep_tolerance =
        -300.f * unit<float>::um;
    cfg.propagation.stepping.min_stepsize = 1e-4f * unit<float>::mm;
    cfg.propagation.stepping.rk_error_tol = 1e-4f * unit<float>::mm;
    cfg.propagation.stepping.step_constraint =
        std::numeric_limits<float>::max();
    cfg.propagation.stepping.path_limit = 5.f * unit<float>::m;
    cfg.propagation.stepping.max_rk_updates = 10000u;
    cfg.propagation.stepping.use_mean_loss = true;
    cfg.propagation.stepping.use_eloss_gradient = false;
    cfg.propagation.stepping.use_field_gradient = false;
    cfg.propagation.stepping.do_covariance_transport = true;

    cfg.initial_links_per_seed = 5;
    return cfg;
}

class TracccGpuStandalone
{
private:
    /// Device ID to use
    int m_device_id;
    /// Geometry directory path
    std::string m_geoDir;

    /// Logger 
    std::unique_ptr<const traccc::Logger> logger;
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
    /// Pinned host memory resource
    vecmem::cuda::host_memory_resource m_pinned_host_mr;
    /// Cached pinned host memory resource
    mutable vecmem::binary_page_memory_resource m_cached_pinned_host_mr;
    /// (vecmem) CUDA stream that owns the underlying cudaStream_t
    vecmem::cuda::stream_wrapper m_vecmem_stream;
    /// (traccc) non-owning view of the CUDA stream
    traccc::cuda::stream_wrapper m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource* m_device_mr;
    /// Device caching memory resource
    mutable vecmem::binary_page_memory_resource m_cached_device_mr;
    /// (Asynchronous) memory copy object
    mutable vecmem::cuda::async_copy m_copy;

    /// Athena to detray map
    std::map<int64_t, uint64_t> m_athena_to_detray_map;
    /// Detray to Athena map (reverse mapping)
    std::unordered_map<uint64_t, int64_t> m_detray_to_athena_map;
    /// detector description to geo id map
    std::unordered_map<traccc::geometry_id, unsigned int> m_geomIdMap;

    // program configuration 
    /// detector options
    traccc::opts::detector m_detector_opts;
    /// Configuration for clustering
    traccc::clustering_config m_clustering_config;
    /// Configuration for the seed finding
    traccc::seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    traccc::spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    traccc::seedfilter_config m_filter_config;

    /// further configuration
    /// Configuration for the track parameter estimation
    traccc::track_params_estimation_config m_track_params_estimation_config;
    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;

    /// host field object
    traccc::magnetic_field m_host_field;
    /// device field object
    traccc::magnetic_field m_field;
    /// const. field for the track finding and fitting
    traccc::vector3 m_field_vec;

    /// Detector design description (module segmentation)
    traccc::detector_design_description::host m_det_descr_storage;
    /// Detector conditions description (module -> design map, geometry IDs)
    traccc::detector_conditions_description::host m_det_cond_storage;
    /// Detector design description buffer
    traccc::detector_design_description::buffer m_device_det_descr;
    /// Detector conditions description buffer
    traccc::detector_conditions_description::buffer m_device_det_cond;
    /// Host detector
    traccc::host_detector m_detector;
    traccc::detector_buffer m_device_detector;

    /// Sub-algorithms used by this full-chain algorithm
    /// Clusterization algorithm
    clustering_algorithm m_clusterization;
    /// Measurement sorting algorithm
    traccc::cuda::measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation_algorithm m_track_parameter_estimation;

    /// Track finding algorithm
    finding_algorithm m_finding;

    // Helper function to read in cells
    std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>> read_all_cells(
        const std::vector<traccc::io::csv::cell> &cells);

    void read_cells(traccc::edm::silicon_cell_collection::host &out,
                const std::vector<traccc::io::csv::cell> &cells);

public:
    TracccGpuStandalone( 
        vecmem::host_memory_resource* host_mr,
        vecmem::cuda::device_memory_resource* device_mr,
        int deviceID = 0,
        const std::string& geoDir = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-01/itk-geo/") :
            m_device_id(deviceID), 
            m_geoDir(geoDir),
            logger(traccc::getDefaultLogger("TracccGpuStandalone", traccc::Logging::Level::INFO)),
            m_host_mr(*host_mr),
            m_pinned_host_mr(),
            m_cached_pinned_host_mr(m_pinned_host_mr),
            m_vecmem_stream(setCudaDevice(deviceID)),
            m_stream(m_vecmem_stream.stream()),
            m_device_mr(device_mr),
            m_cached_device_mr(*m_device_mr),
            m_copy(m_stream.cudaStream()),
            m_clustering_config{256, 16, 8, 256},
            m_finder_config(create_and_setup_finder_config()),
            m_grid_config(m_finder_config),
            m_filter_config(create_and_setup_filter_config()),
            m_finding_config(create_and_setup_finding_config()),
            m_host_field(make_magnetic_field(geoDir + "ITk_bfield.cvf")),
            m_field(traccc::cuda::make_magnetic_field(m_host_field)),
            m_field_vec({0.f, 0.f, m_finder_config.bFieldInZ}),
            m_det_descr_storage(m_host_mr),
            m_det_cond_storage(m_host_mr),
            m_device_det_descr(std::vector<unsigned int>{}, *m_device_mr,
                               &m_host_mr,
                               vecmem::data::buffer_type::resizable),
            m_device_det_cond(0, *m_device_mr),
            m_clusterization({m_cached_device_mr, &m_cached_pinned_host_mr}, m_copy,
                            m_stream, m_clustering_config),
            m_measurement_sorting({m_cached_device_mr, &m_cached_pinned_host_mr},
                                    m_copy, m_stream,
                                    logger->cloneWithSuffix("MeasSortingAlg")),
            m_spacepoint_formation({m_cached_device_mr, &m_cached_pinned_host_mr},
                                    m_copy, m_stream,
                                    logger->cloneWithSuffix("SpFormationAlg")),
            m_seeding(m_finder_config, m_grid_config, m_filter_config,
                        {m_cached_device_mr, &m_cached_pinned_host_mr}, m_copy,
                        m_stream, logger->cloneWithSuffix("SeedingAlg")),
            m_track_parameter_estimation(
                m_track_params_estimation_config,
                {m_cached_device_mr, &m_cached_pinned_host_mr}, m_copy, m_stream,
                logger->cloneWithSuffix("TrackParEstAlg")),
            m_finding(m_finding_config, {m_cached_device_mr, &m_cached_pinned_host_mr},
                        m_copy, m_stream, logger->cloneWithSuffix("TrackFindingAlg"))
    {
        // Tell the user what device is being used.
        int device = 0;
        CUDA_ERROR_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << "Using CUDA device: " << props.name << " [id: " << device
                << ", bus: " << props.pciBusID
                << ", device: " << props.pciDeviceID << "]" << std::endl;

        initialize();
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    std::vector<traccc::io::csv::cell> read_from_array(
        const int64_t* cell_positions,
        const float* cell_properties,
        size_t num_cells,
        bool athena_ids);

    // getters
    const std::map<int64_t, uint64_t>& getAthenaToDetrayMap() const {
        return m_athena_to_detray_map;
    }

    const std::unordered_map<uint64_t, int64_t>& getDetrayToAthenaMap() const {
        return m_detray_to_athena_map;
    }

    void initialize();

    /// Adopt the detector from a /dev/shm region instead of parsing JSON.
    /// Throws on any mismatch -- see the comment in the definition.
    void adopt_detector_from_shm(const std::string& shm_name);

    TracccResults run(std::vector<traccc::io::csv::cell> cells, bool show_stats = false);

};


void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = m_geoDir + "/detray_detector_geometry.json";
    m_detector_opts.digitization_file = m_geoDir + "/ITk_digitization_config.json";
    m_detector_opts.grid_file = m_geoDir + "/detray_detector_surface_grids.json";
    m_detector_opts.material_file = m_geoDir + "/detray_detector_material_maps.json";
    // ITk has no separate conditions file. The conditions reader only picks up
    // the optional "shift" key, which this file does not carry, so every module
    // ends up with a zero measurement translation.
    m_detector_opts.conditions_file = m_detector_opts.digitization_file;

    // Load Athena-to-Detray mapping
    std::string athenaTransformsPath = m_geoDir + "/athenaIdentifierToDetrayMap.txt";
    m_athena_to_detray_map = read_athena_to_detray_mapping(athenaTransformsPath);

    // Create reverse mapping from Detray to Athena
    m_detray_to_athena_map.reserve(m_athena_to_detray_map.size());
    for (const auto& [athena_id, detray_id] : m_athena_to_detray_map) {
        m_detray_to_athena_map[detray_id] = athena_id;
    }

    traccc::io::read_detector_description(
        m_det_descr_storage, m_det_cond_storage, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, m_detector_opts.conditions_file,
        traccc::data_format::json);

    // The design description holds jagged bin-edge arrays, so its device buffer
    // needs a per-element capacity and must be resizable.
    std::vector<unsigned int> descr_sizes(m_det_descr_storage.size());
    for (std::size_t i = 0; i < m_det_descr_storage.size(); ++i) {
        auto this_design = m_det_descr_storage.at(i);
        descr_sizes[i] = std::max(
            static_cast<unsigned int>(this_design.bin_edges_x().size()),
            static_cast<unsigned int>(this_design.bin_edges_y().size()));
    }
    m_device_det_descr = traccc::detector_design_description::buffer(
        descr_sizes, *m_device_mr, &m_host_mr,
        vecmem::data::buffer_type::resizable);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(vecmem::get_data(m_det_descr_storage), m_device_det_descr)->wait();

    m_device_det_cond = traccc::detector_conditions_description::buffer(
        static_cast<traccc::detector_conditions_description::buffer::size_type>(
            m_det_cond_storage.size()),
        *m_device_mr);
    m_copy.setup(m_device_det_cond)->wait();
    m_copy(vecmem::get_data(m_det_cond_storage), m_device_det_cond)->wait();
    m_stream.synchronize();

    // fill the module (conditions) index to geometry id map
    m_geomIdMap.clear();
    m_geomIdMap.reserve(m_det_cond_storage.geometry_id().size());
    for (unsigned int i = 0; i < m_det_cond_storage.geometry_id().size(); ++i) {
        m_geomIdMap[m_det_cond_storage.geometry_id()[i].value()] = i;
    }

    // Two geometry sources. Setting TRACCC_DETRAY_SHM adopts an already-built
    // detector from a shared-memory region; otherwise the original JSON path
    // runs unchanged. Both are kept so the two can be compared on identical
    // input -- a result with nothing to compare against proves nothing.
    if (const char* shm_name = std::getenv("TRACCC_DETRAY_SHM")) {
        adopt_detector_from_shm(shm_name);
    } else {
        traccc::io::read_detector(
            m_detector, m_host_mr, m_detector_opts.detector_file,
            m_detector_opts.material_file, m_detector_opts.grid_file);
        m_device_detector =
            traccc::buffer_from_host_detector(m_detector, *m_device_mr, m_copy);
    }
    m_stream.synchronize();

    return;
}

/* Adopt a detray detector built by another process inside /dev/shm.
 *
 * No parsing happens here. The producer built the detector THROUGH a memory
 * resource sitting on the region, so the payload is already laid out; all that
 * crosses is the view -- a small bundle of pointers and sizes -- which is valid
 * here only because both processes map at the identical detray_shm::FIXED_BASE.
 *
 * Every check below refuses the load. None is a warning: sharing bytes across a
 * version gap gives a parse error, but sharing a constructed OBJECT across one
 * gives silent memory corruption -- the layout differs, nothing throws, and the
 * tracks are quietly wrong. There is no safe degraded mode for reading a struct
 * with the wrong layout.
 */
void TracccGpuStandalone::adopt_detector_from_shm(const std::string& shm_name)
{
    const auto refuse = [&shm_name](const std::string& why) {
        throw std::runtime_error("detray-shm: REFUSED to load " + shm_name +
                                 " -- " + why);
    };

    void* base = detray_shm::map_region_shared(shm_name);
    auto* hdr = static_cast<detray_shm::header*>(base);

    if (hdr->magic != detray_shm::MAGIC) {
        refuse("bad magic -- not a detray-shm region");
    }
    // Acquire pairs with the producer's release store: seeing ready == 1
    // guarantees every field written before it is visible too.
    if (__atomic_load_n(&hdr->ready, __ATOMIC_ACQUIRE) != 1u) {
        refuse("region not published (torn or still being written)");
    }
    if (hdr->format_version != detray_shm::FORMAT_VERSION) {
        refuse("format v" + std::to_string(hdr->format_version) + ", expected v" +
               std::to_string(detray_shm::FORMAT_VERSION));
    }
    if (hdr->base_addr != reinterpret_cast<std::uint64_t>(base)) {
        refuse("produced at a different base address -- every pointer inside "
               "the region would be wrong");
    }
    if (hdr->detray_major != DETRAY_VERSION_MAJOR ||
        hdr->detray_minor != DETRAY_VERSION_MINOR ||
        hdr->detray_patch != DETRAY_VERSION_PATCH) {
        refuse("detray " + std::to_string(hdr->detray_major) + "." +
               std::to_string(hdr->detray_minor) + "." +
               std::to_string(hdr->detray_patch) + " in region, " +
               DETRAY_VERSION + " here");
    }
    if (hdr->vecmem_major != VECMEM_VERSION_MAJOR ||
        hdr->vecmem_minor != VECMEM_VERSION_MINOR) {
        refuse("vecmem version mismatch");
    }

    using view_t = traccc::itk_detector::view;
    if (hdr->view_bytes != sizeof(view_t)) {
        refuse("view is " + std::to_string(hdr->view_bytes) +
               " bytes in region, " + std::to_string(sizeof(view_t)) + " here");
    }

    view_t view;
    std::memcpy(&view, hdr->view, sizeof(view_t));

    /* detray::get_buffer already accepts a view: buffer_from_host_detector()
     * calls get_buffer(det, ...), which is itself defined as
     * get_buffer(det.get_data(), ...). The host detector is only where the view
     * usually comes from -- ours comes from the region instead, so no host
     * detector is built at all. The host->device copy still happens: kernels
     * need the detector in VRAM regardless. */
    m_device_detector.set<traccc::itk_detector>(
        detray::get_buffer(view, *m_device_mr, m_copy));

    std::cout << "Adopted detector from " << shm_name << ": "
              << (static_cast<double>(hdr->payload_bytes) / (1024.0 * 1024.0))
              << " MB, " << hdr->n_volumes << " volumes, " << hdr->n_surfaces
              << " surfaces, " << hdr->n_transforms
              << " transforms -- no JSON parsed" << std::endl;
}

TracccResults TracccGpuStandalone::run(
    std::vector<traccc::io::csv::cell> cells, bool show_stats
) {
    if (show_stats) start_total = std::chrono::high_resolution_clock::now();

    // Read cells
    if (show_stats) start_read = std::chrono::high_resolution_clock::now();
    traccc::edm::silicon_cell_collection::host read_out(m_host_mr);
    read_cells(read_out, cells);
    if (show_stats) end_read = std::chrono::high_resolution_clock::now();

    // Copy to device
    if (show_stats) start_copy_in = std::chrono::high_resolution_clock::now();
    traccc::edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(read_out.size()), m_cached_device_mr);
    m_copy(vecmem::get_data(read_out), cells_buffer)->ignore();
    if (show_stats) end_copy_in = std::chrono::high_resolution_clock::now();

    // Clusterization
    if (show_stats) start_cluster = std::chrono::high_resolution_clock::now();
    auto unsorted_measurements =
        m_clusterization(cells_buffer, m_device_det_descr, m_device_det_cond);
    auto measurements =
        m_measurement_sorting(unsorted_measurements);
    if (show_stats) end_cluster = std::chrono::high_resolution_clock::now();

    // Spacepoint formation
    if (show_stats) start_spacepoint = std::chrono::high_resolution_clock::now();
    auto spacepoints =
        m_spacepoint_formation(m_device_detector, measurements);
    if (show_stats) end_spacepoint = std::chrono::high_resolution_clock::now();

    // Seeding
    if (show_stats) start_seeding = std::chrono::high_resolution_clock::now();
    auto seeds = m_seeding(spacepoints);
    if (show_stats) end_seeding = std::chrono::high_resolution_clock::now();

    // Track parameter estimation
    if (show_stats) start_params = std::chrono::high_resolution_clock::now();
    auto track_params =
        m_track_parameter_estimation(m_field, measurements, spacepoints, seeds);
    if (show_stats) end_params = std::chrono::high_resolution_clock::now();

    // Track finding
    if (show_stats) start_finding = std::chrono::high_resolution_clock::now();
    auto track_candidates = m_finding(
        m_device_detector, m_field, measurements, track_params);
    if (show_stats) end_finding = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    if (show_stats) start_copy_out = std::chrono::high_resolution_clock::now();
    traccc::edm::track_container<traccc::default_algebra>::host
        track_states_host{m_host_mr};

    m_copy(track_candidates.tracks, track_states_host.tracks,
            vecmem::copy::type::device_to_host)->wait();
    m_copy(track_candidates.states, track_states_host.states,
            vecmem::copy::type::device_to_host)->wait();

    // copy measurements back to host
    traccc::edm::measurement_collection::host measurements_host(m_host_mr);
    m_copy(track_candidates.measurements, measurements_host, vecmem::copy::type::device_to_host)->wait();
    if (show_stats) end_copy_out = std::chrono::high_resolution_clock::now();

    if (show_stats) 
    {
        auto end_total = std::chrono::high_resolution_clock::now();

        // Print timing information
        std::cout << "\n=== Timing Information ===" << std::endl;
        std::cout << "Read cells:          " 
                << std::chrono::duration<double, std::milli>(end_read - start_read).count() 
                << " ms" << std::endl;
        std::cout << "Copy to device:      " 
                << std::chrono::duration<double, std::milli>(end_copy_in - start_copy_in).count() 
                << " ms" << std::endl;
        std::cout << "Clusterization:      " 
                << std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count() 
                << " ms" << std::endl;
        std::cout << "Spacepoint form.:    " 
                << std::chrono::duration<double, std::milli>(end_spacepoint - start_spacepoint).count() 
                << " ms" << std::endl;
        std::cout << "Seeding:             " 
                << std::chrono::duration<double, std::milli>(end_seeding - start_seeding).count() 
                << " ms" << std::endl;
        std::cout << "Track param. est.:   " 
                << std::chrono::duration<double, std::milli>(end_params - start_params).count() 
                << " ms" << std::endl;
        std::cout << "Track finding:       " 
                << std::chrono::duration<double, std::milli>(end_finding - start_finding).count() 
                << " ms" << std::endl;
        // std::cout << "Track fitting:       " 
        //         << std::chrono::duration<double, std::milli>(end_fitting - start_fitting).count() 
        //         << " ms" << std::endl;
        std::cout << "Copy to host:        " 
                << std::chrono::duration<double, std::milli>(end_copy_out - start_copy_out).count() 
                << " ms" << std::endl;
        std::cout << "------------------------" << std::endl;
        std::cout << "Total time:          " 
                << std::chrono::duration<double, std::milli>(end_total - start_total).count() 
                << " ms" << std::endl;
        std::cout << "=========================\n" << std::endl;

        std::cout << "Number of measurements: " << m_copy.get_size(measurements) << std::endl;
        std::cout << "Number of spacepoints: " << m_copy.get_size(spacepoints) << std::endl;
        std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
        std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
        traccc::edm::track_container<traccc::default_algebra>::host track_candidates_host{m_host_mr};
        m_copy(track_candidates.tracks, track_candidates_host.tracks,
               vecmem::copy::type::device_to_host)->wait();
        std::cout << "Number of smoothed tracks: " << track_states_host.tracks.size() << std::endl;

    }

    return {track_states_host, measurements_host};
}

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_from_array(
    const int64_t* cell_positions,
    const float* cell_properties,
    size_t num_cells,
    bool athena_ids = false)
{
    std::vector<traccc::io::csv::cell> cells;
    cells.reserve(num_cells);

    for (size_t i = 0; i < num_cells; ++i) 
    {

        traccc::io::csv::cell cell;

        if (athena_ids){
            cell.geometry_id = m_athena_to_detray_map.at(cell_positions[i * 4]);
        } else {
            cell.geometry_id = cell_positions[i * 4];
        }
        cell.measurement_id = cell_positions[i * 4 + 1];
        cell.channel0 = cell_positions[i * 4 + 2];
        cell.channel1 = cell_positions[i * 4 + 3];

        cell.timestamp = cell_properties[i * 2];
        cell.value = cell_properties[i * 2 + 1];

        cells.push_back(cell);
    }

    return cells;
}

std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>>
    TracccGpuStandalone::read_all_cells(
        const std::vector<traccc::io::csv::cell> &cells)
{
    std::unordered_map<std::uint64_t, std::vector<traccc::io::csv::cell>> result;

    // Pre-count cells per geometry_id to avoid reallocations
    std::unordered_map<std::uint64_t, size_t> counts;
    for (const auto &cell : cells) {
        counts[cell.geometry_id]++;
    }
    
    // Reserve space for each geometry_id
    for (const auto& [geom_id, count] : counts) {
        result[geom_id].reserve(count);
    }

    for (const auto &iocell : cells)
    {
        result[iocell.geometry_id].emplace_back(iocell.geometry_id, iocell.measurement_id, 
                            iocell.channel0, iocell.channel1, 
                            iocell.timestamp, iocell.value);
    }

    // Sorting happens on the client side!
    // put here again for redundancy
    for (auto& [_, cells] : result) 
    {
        std::sort(cells.begin(), cells.end(), ::cell_order());
    }

    return result;
}

void TracccGpuStandalone::read_cells(traccc::edm::silicon_cell_collection::host &out,
                const std::vector<traccc::io::csv::cell> &cells)
{
    out.resize(0);
    out.reserve(cells.size());

    // get the cells and modules in intermediate format
    auto cellsMap = read_all_cells(cells);

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [geometry_id, cellz] : cellsMap) {

        // Figure out the index of the detector description object for this
        // group of cells.
        unsigned int ddIndex = 0;
        auto it = m_geomIdMap.find(geometry_id);
        if (it == m_geomIdMap.end()) {
            throw std::runtime_error("Could not find geometry ID (" +
                                        std::to_string(geometry_id) +
                                        ") in the detector description");
        }
        ddIndex = it->second;

        // Add the cells to the output.
        for (auto& cell : cellz) {
            out.push_back({cell.channel0, cell.channel1, cell.value,
                             cell.timestamp, ddIndex});
        }
    }
}

#endif 