# Sharing a detray detector through shared memory

Today the ITk geometry reaches the traccc Triton backend as JSON — about
0.80 GiB across three files — which the backend parses at startup into its own
private host detector. Because `TracccGpuStandalone` is built per Triton
*instance*, `instance_group { count: N }` means N parses and N detectors in RAM.

This folder does it differently: a producer builds the detector **directly inside
a `/dev/shm` region**, and the backend maps that region and uses the detector
without parsing anything. No serialization step exists anywhere — the detector is
allocated through a memory resource that hands out shared memory, so it is in
shared memory the moment construction finishes.

| | JSON path | shared region |
|---|---|---|
| model load, in the image (lxplus901) | 33.3 s | **13.5 s** |
| model load, on the ATLAS release (lxplus902) | 29.1 s | **13.2 s** |
| geometry size | 0.80 GiB of JSON | **246 MB** in RAM |

Track counts match the JSON path, verified with the Athena client.

**Two ways to build this, and `release/` is the one to use.** The scripts here
build against the `traccc-aas.sif` image; those in `release/` build against an
ATLAS ACTS release, which ships `tritonserver` *and* traccc/detray/vecmem
together. Building there puts the server on exactly the stack Athena uses, so the
region's version gates pass by construction instead of by keeping two stacks in
step — and Athena itself can then be the producer. See `release/README.md`.

## How it works

    producer                                backend (tritonserver)
    --------                                ----------------------
    mmap /dev/shm/<name> at FIXED_BASE      mmap the SAME region, same address
      |                                       |
    shm_memory_resource (a bump allocator)  validate the header:
      |                                       magic · ready · format
      |                                       base address · detray + vecmem
      |                                       versions · view size
    read_detector(det, shm_mr, json...)       (any mismatch REFUSES the load)
      |                                       |
    write header: counts, versions, view    memcpy the view out of the header
      |                                       |
    ready = 1  (atomic release)             detray::get_buffer(view, ...) -> GPU

The *payload* — volumes, surfaces, transforms, masks, materials, accelerators —
lives in the region, one copy, read by every consumer. The *handle* (the view: a
small bundle of pointers and sizes) is per-process and about a kilobyte. That
split is what makes the whole thing work.

**Why the fixed address.** detray's containers hold raw pointers, so a view is
only meaningful in the process that produced it. Rather than store offsets and
rebuild each sub-view, this version has both processes map at the same
hard-coded `FIXED_BASE`, which keeps every pointer valid and lets the view be
copied verbatim. Verified free inside a running `tritonserver`. Relocation is the
follow-up; until it exists, `MAP_FIXED_NOREPLACE` can lose to any mapping that
happens to sit there.

**Why the version gates are absolute.** Sharing *bytes* across a version gap
gives a parse error. Sharing a constructed *object* across one gives silent
memory corruption: the layout differs, nothing throws, and the tracks are quietly
wrong. There is no safe degraded mode, so every check refuses the load.

## Files

    common.sh              shared settings; locates the repo, holds every default
    prepare_geometry.sh    stage the ITk geometry on node-local disk (once per node)
    build.sh               rebuild the Triton backend outside the image
    run_server.sh          start tritonserver against that backend
    build_producer.sh      build the producer
    run_producer.sh        build the detector into /dev/shm, leave it there
    producer.cpp           the producer (stand-in for Athena)
    shm_memory_resource.hpp   ~30 lines: a vecmem resource over the mapped region
    CMakeLists.txt         builds the producer only

The backend side lives in the repo proper:

    standalone/src/shm_region.{hpp,cpp}      region layout + mapping
    standalone/src/TracccGpuStandalone.hpp   initialize() branches on TRACCC_DETRAY_SHM
    backend/traccc-gpu/src/traccc.cc         model init wrapped in try/catch

`shm_region.hpp` is deliberately **not** duplicated here: `build_producer.sh`
stages the repo's copy in, so the producer and the backend compile against the
same definition. Two copies free to drift would disagree about the region layout
in silence — the exact failure this design exists to prevent.

## Running it

Everything is override-able through the environment; `common.sh` lists the
defaults. The image (`traccc-aas.sif`, 11 GB) lives outside the repo — point
`SIF` at yours.

    ./prepare_geometry.sh        # once per node
    ./build.sh                   # 10-20 min the first time

Baseline first — the JSON path must work before the new one means anything:

    ./run_server.sh              # expect: traccc-gpu | 1 | READY

Then the shared region:

    ./build_producer.sh
    ./run_producer.sh            # prints parse time, counts, real RAM used
    SHM=/d3_itk_detector ./run_server.sh

Look for the line only the new path prints:

    Adopted detector from /d3_itk_detector: 379 volumes, 60911 surfaces,
    61290 transforms -- no JSON parsed

Stop the server between runs (Ctrl-C); `run_server.sh` refuses immediately if a
port is still held rather than failing after a 35-second model load.

## Constraints worth knowing

- **Build and run on the same node.** The backend compiles `-march=native`, and
  `/tmp` is node-local anyway.
- **Port 8000 is taken on every lxplus node** by a system service. Triton treats
  a failed HTTP bind as fatal — it reports READY, then exits. Hence
  `--http-port=8010`; gRPC 8001 is free so ordinary clients are unaffected.
- **`-Werror` is on** in the backend's CMakeLists. Any warning fails the build.
- **The image is never modified.** It is read-only and ships its own
  `libtriton_traccc.so`; `run_server.sh` hands Triton a model repository
  containing ours instead.
- **`backend/Dockerfile` line 27** rewrites a hard-coded NERSC geometry path at
  image build time, and that edit exists only in the Dockerfile. Building the
  repo as-is produces a backend that aborts on a missing `ITk_bfield.cvf`, so
  `build.sh` applies the same substitution to its staged copy.

## Known defects and open questions

- **`payload_bytes` in the header is wrong.** It reports the whole region
  (2048 MB), not the detector, because `vecmem::contiguous_memory_resource`
  claims its entire chunk from upstream up front — so `used()` measures the
  reservation. The 246 MB above came from `du -h /dev/shm/<region>` versus
  `du -h --apparent-size`, which also shows the oversized reservation costs no
  RAM: tmpfs only commits pages that are written. Fix: drop the wrapper and pass
  `shm_memory_resource` straight to `read_detector` — a bump allocator is
  contiguous by construction, so the wrapper buys nothing and hides the number
  worth reporting.
- **`map_region_shared()` is untested.** It maps once per process so
  `instance_group { count: N }` works, but has only run with `count: 1`.
- **The mapping is never released.** Fine while the region outlives the server,
  but lifecycle — IOV changes, producer restart, crash cleanup — is unowned.
- **Only `traccc::itk_detector` is handled.** The adopt path hard-codes that
  type while the JSON path is polymorphic over `detector_type_list`.
- **`read_detector_description` and the 58,700-entry identifier map** are
  separate payloads and still come from JSON. Sharing the detector does not share
  them; that is the remaining 13.5 s.
- **Version lockstep — only for the image path.** The image and Athena's release
  are separately built stacks that must be kept in step, and they have already
  diverged (vecmem 1.25 vs 1.27 changed `jagged_vector_view`'s layout). The gates
  refuse across that gap, correctly. `release/` removes the problem rather than
  managing it.
