# Sharing the detray detector through memory

The ITk geometry reaches the traccc Triton backend as JSON — about 0.80 GiB
across three files — which the backend parses at startup into its own private
detector. Athena has already built that same detector a moment earlier, so the
parse produces nothing that did not exist.

Instead: **Athena builds the detector directly inside a `/dev/shm` region, and the
server maps that region and uses it without parsing anything.** No serialization
step exists anywhere. The detector is allocated through a memory resource that
hands out shared memory, so it is in shared memory the moment construction
finishes.

| | JSON | shared region |
|---|---|---|
| model load | 29.1 s | **13.0 s** |
| geometry | 0.80 GiB of JSON | **246 MB** resident |

Measured on lxplus902, Tesla T4, 2026-09-08. Track counts match the JSON path.

## The two halves

**The producer is Athena.** `JSONDeviceDetectorDescriptionProviderSvc`
(`Tracking/Acts/ActsGPUGeometry`) gains a `SharedMemoryRegion` property; when
set, `read_detector` allocates through the region instead of ordinary host
memory, and the service writes a header and publishes it. That change lives in
Athena, not in this repo.

**The consumer is the backend in this repo.** `initialize()` branches on
`TRACCC_DETRAY_SHM`: set, it validates the header and hands the region's view
straight to the GPU; unset, the original JSON path runs unchanged.

    standalone/src/shm_region.{hpp,cpp}       region layout and mapping
    standalone/src/shm_memory_resource.hpp    ~30 lines: a bump allocator that
                                              is a vecmem memory resource
    standalone/src/TracccGpuStandalone.hpp    the consumer path
    backend/traccc-gpu/src/traccc.cc          model init wrapped in try/catch

Athena vendors the first two, because both sides must agree on the region layout
exactly. They want one upstream home; detray is the natural candidate.

## Built against the release, not the image

These scripts build against an ATLAS ACTS release rather than `traccc-aas.sif`.
The release ships `tritonserver` **and** traccc, detray, vecmem and covfie, built
together with one compiler — so the server runs on exactly the stack Athena does.

That is not a convenience. The image and the release are separately built and had
already diverged: vecmem 1.27 changed `jagged_vector_view::size_type` from
`std::size_t` to `unsigned int`, and detray's grids are jagged containers, so a
region written by one and read by the other would resolve pointers at the wrong
offsets — silently. The header's version gates catch it, correctly. But with two
stacks the chase never ends; with one, the gates pass by construction.

## Running it

One GPU node for everything: `/tmp` is node-local and the backend is
`-march=native`.

    ./prepare_geometry.sh    # once per node
    ./build.sh               # backend, against the release
    ./run_server.sh          # JSON baseline — confirm READY first

### Producing the region from Athena

The producer side is an Athena change and lives in Athena, not here. Until it is
merged, it is on a branch:

    https://gitlab.cern.ch/tihuang/athena  branch detray-shm-producer

Build just that package on top of the same release — Athena and this server must
run the same one, or the region's ABI gates will refuse it:

    asetup main--ACTS,Athena,2026-09-07T2100
    lsetup git
    git atlas init-workdir ssh://git@gitlab.cern.ch:7999/tihuang/athena.git
    cd athena && git checkout detray-shm-producer && git atlas addpkg ActsGPUGeometry
    cd .. && mkdir build && cd build
    cmake ../athena/Projects/WorkDir && make -j$(nproc)
    source ./*/setup.sh

Then run any job that configures `JSONDeviceDetectorDescriptionProviderSvc` with
`SharedMemoryRegion` set. The package ships a minimal one — a zero-event job that
does nothing but build the detector into the region:

    athena.py ../athena/Tracking/Acts/ActsGPUGeometry/test/ActsDeviceSharedMemoryTest.py

It prints the region it wrote and the counts it published. The region deliberately
outlives the job.

### Adopting it

    SHM=/athena_itk_detector ./run_server.sh

Look for the line only this path prints:

    Adopted detector from /athena_itk_detector: 379 volumes, 60911 surfaces,
    61290 transforms -- no JSON parsed

## What the release needed repairing

All five are confined to a staged copy by `release-build.patch`; the repo itself
is untouched by them:

1. **`traccc::opts::detector` → plain strings.** The release's traccc has no
   `options` component. The struct only held five file paths.
2. **`traccc::performance` dropped** from the link — also absent, and unused.
3. **TritonCommon.** Its `Config.cmake` references a `triton-common-json` target
   missing from its installed `Targets.cmake`, *and* uses that target as its
   include guard — so `find_package` either errors or silently imports nothing.
   Include the targets file directly and declare the header-only target by hand.
4. **`find_package(Threads)` and `find_package(CUDAToolkit)`** added.
5. **The Eigen workaround generalised.** The exported targets advertise an
   `eigen3` directory inside the ACTS include tree that the release does not
   install.

Items 3 and 5 are ATLAS externals packaging bugs, worth reporting upstream.

Two environment notes: `asetup` puts only Athena's own areas on
`CMAKE_PREFIX_PATH`, so LCG paths are derived from `$ROOT_INCLUDE_PATH` (without
which Boost resolves to the system 1.75 instead of the release's 1.91); and the
release's `tritonserver` is built without HTTP or metrics, accepting only gRPC.

## Known limits

- **The fixed address is a shortcut.** Both processes map at
  `0x2000_0000_0000` so detray's stored pointers stay valid, rather than storing
  offsets and rebuilding the sub-views. Verified free in a full Athena job and in
  `tritonserver`, but rebuilding at an arbitrary base is the durable answer.
- **`map_region_shared()` is untested** beyond `instance_group { count: 1 }`. It
  maps once per process so N model instances share one mapping — necessary
  because an address can only be claimed once per address space.
- **Only `traccc::itk_detector`** is handled on the adopt path, while the JSON
  path is polymorphic over `detector_type_list`.
- **Lifecycle is unowned.** The mapping is never released, and nothing handles a
  producer restart, an IOV change, or cleanup after a crash.
- **The detector is not the whole geometry.** `read_detector_description` and the
  58,700-entry identifier map are separate payloads and still come from JSON.
  That is most of the remaining 13 s.
