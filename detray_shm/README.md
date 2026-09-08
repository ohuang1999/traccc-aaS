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

The producer in those measurements was a zero-event Athena job that does nothing
but initialize the service. Setting the property inside a full reconstruction job
is the same one-line change, but has not been run.

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

## Running it end to end

Three pieces, all on one GPU node: this server, an Athena job that produces the
region, and an Athena client that sends hits. `/tmp` is node-local and the backend
is `-march=native`, so nothing here travels between machines.

### 1. The server

    ./prepare_geometry.sh    # once per node; needs the eftracking e-group
    ./build.sh               # the backend, against the release
    ./run_server.sh          # JSON path — confirm READY before going further

Getting the baseline working first matters: if it fails, the problem is the build,
not the sharing.

### 2. Athena, producing the region

The producer side is an Athena change and lives in Athena. Until it is merged it
is on a branch:

    https://gitlab.cern.ch/tihuang/athena   branch detray-shm-producer

Build that one package against the **same release** this server uses — the
region's ABI gates compare what each side was compiled against, and refuse a
mismatch:

    asetup main--ACTS,Athena,2026-09-07T2100
    lsetup git
    git atlas init-workdir ssh://git@gitlab.cern.ch:7999/tihuang/athena.git
    cd athena && git checkout detray-shm-producer && git atlas addpkg ActsGPUGeometry
    cd .. && mkdir build && cd build
    cmake ../athena/Projects/WorkDir && make -j$(nproc)
    source ./*/setup.sh

The change adds one property to `JSONDeviceDetectorDescriptionProviderSvc`. Set it
wherever that service is configured and the detector is built into the region
instead of ordinary host memory:

    JSONDeviceDetectorDescriptionProviderSvcCfg(
        flags,
        SharedMemoryRegion="/athena_itk_detector",
        SharedMemoryRegionGB=2,      # tmpfs commits only what is written
        ...)

Leave it unset and Athena behaves exactly as before. When it is set the job logs
the region it wrote and the counts it published, and the region deliberately
outlives the job.

### 3. Adopting it, and running tracks

Restart the server pointed at the region:

    SHM=/athena_itk_detector ./run_server.sh

The line only this path prints:

    Adopted detector from /athena_itk_detector: 379 volumes, 60911 surfaces,
    61290 transforms -- no JSON parsed

Then send it work. The client is already in the release as `TracccTritonClient`,
so nothing extra needs building:

    Reco_tf.py --CA \
      --inputRDOFile <an ITk RDO> --outputAODFile AOD.pool.root \
      --steering doRAWtoALL \
      --preInclude "TracccTritonClient.TracccTritonClientConfigFlags.tracccTritonFlagsPreInclude" \
      --postInclude "TracccTritonClient.TracccTritonClientConfig.TritonTracccTrackMakerCfg" \
      --maxEvents 5

The `preInclude` is not optional: the Triton flags live in the package rather
than centrally, so they have to be registered before anything reads them. Their
defaults — `traccc-gpu` on `localhost:8001` — already match what
`run_server.sh` starts, so no `--preExec` is needed when both are on one node.
Elsewhere, override `flags.Tracking.Traccc.Triton.url` and `.port`.

`Number of tracks found:` in the log is the number to compare between the two
geometry sources. Expect agreement, not equality — traccc is nondeterministic.

### Which node runs what

**The producer must be on the server's node.** `/dev/shm` is per-machine, so a
region written anywhere else is invisible to the server.

**The client needs a GPU too**, which is easy to miss: `TritonTracccTrackMakerCfg`
configures `JSONDeviceDetectorDescriptionProviderSvc`, and that pulls in CUDA
memory resources. It is not a pure CPU job in this configuration. It can run on a
different GPU node if you point the flags at the server, but the simplest
arrangement is all three on one node.

That the client configures the same service is worth noticing for another reason:
**the client builds its own detector as well.** This work removes the server's
parse; the client's is untouched, and nobody has measured it.

The measurements above used an older client — release 25.0.45 plus a development
branch, from before `TracccTritonClient` was merged. The in-release client is the
right one to use now, but has not been run against this server here.

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
