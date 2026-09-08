# The server on Athena's own stack

Everything here does what the scripts one level up do — a producer builds the ITk detray
detector into `/dev/shm`, the Triton backend adopts it without parsing — but
built against an **ATLAS ACTS release** instead of the `traccc-aas.sif` image.

That single change removes what the plan called D4's main ongoing cost.

## Why

The image and Athena were two separately-built stacks that had to be kept in
step. They had already diverged: vecmem 1.25 in the image against 1.27 in the
release, and `jagged_vector_view::size_type` changed from `std::size_t` to
`unsigned int` between them. detray's grids are jagged containers, so a region
written by one and read by the other would resolve pointers at the wrong offsets,
silently. The version gates catch it, correctly — but the underlying chase never
ends: the nightly moves, the image follows.

The release ships `tritonserver` **and** traccc, detray, vecmem, covfie and the
Triton API packages, all built together with one compiler. Building the backend
there puts the server on exactly the stack Athena uses. The gates then pass by
construction rather than by maintenance, and there is no image to rebuild.

## Measured — lxplus902, Tesla T4, 2026-09-08

| | JSON | shared region |
|---|---|---|
| model load | 29.1 s | **13.2 s** |
| geometry | 0.80 GiB | **246 MB** |
| `read_detector` in the producer | 23.5 s | — |

    Adopted detector from /rel_itk_detector: 379 volumes, 60911 surfaces,
    61290 transforms -- no JSON parsed

Close to the image's 33.3 → 13.5 s, on a completely different toolchain: gcc 15,
vecmem 1.27, detray 0.111, CUDA 13.3, and no container at any point.

`view bytes: 976` here differs from the image's, because of that vecmem layout
change. The `view_bytes` gate would catch a region crossing the two stacks.

## Files

    common.sh                   release setup and shared defaults
    build_release.sh            build the backend against the release
    run_server_release.sh       start tritonserver from the release
    build_producer_release.sh   build the producer
    run_producer_release.sh     detector into /dev/shm
    CMakeLists.txt              builds the producer only
    release-build.patch         the five repairs, applied to a staged copy

## Running it

Everything on one GPU node — `/tmp` is node-local and the backend is
`-march=native`.

    ./build_release.sh
    ./run_server_release.sh                       # JSON baseline first

    ./build_producer_release.sh
    ./run_producer_release.sh
    SHM=/rel_itk_detector ./run_server_release.sh

## What the release needed repairing

All five live in `release-build.patch`, applied to a staged copy so the repo is
untouched:

1. **`traccc::opts::detector` → plain strings.** The release's traccc has no
   `options` component. The struct only held five file paths.
2. **`traccc::performance` dropped** from the link — also absent, and unused.
3. **TritonCommon.** Its `Config.cmake` references a `triton-common-json` target
   missing from its installed `Targets.cmake`, *and* uses that target as its
   include guard — so `find_package` either errors or silently imports nothing.
   Include the targets file directly and declare the header-only target by hand.
4. **`find_package(Threads)` and `find_package(CUDAToolkit)`** added; the Triton
   targets need both.
5. **The Eigen workaround generalised.** The exported targets advertise an
   `eigen3` directory inside the ACTS include tree that the release does not
   install. Any non-existent `eigen3` entry is now rewritten to the real one.

Items 3 and 5 are ATLAS externals packaging bugs, worth reporting upstream.

Two environment notes: `asetup` puts only Athena's own areas on
`CMAKE_PREFIX_PATH`, so the LCG paths are derived from `$ROOT_INCLUDE_PATH`
(without which Boost resolves to the system 1.75 instead of 1.91); and the
release's `tritonserver` is built without HTTP or metrics, accepting only gRPC
options — which incidentally makes the lxplus port-8000 problem irrelevant here.

## Still open

- **`map_region_shared()` is untested** beyond `instance_group { count: 1 }`.
- **Athena as the producer works** (see `sandbox_detray_shm/d4_athena` in the
  parent workspace): a patched `JSONDeviceDetectorDescriptionProviderSvc` builds
  the detector into the region and the server here adopts it, 29.1 s → 13.0 s.
  That change is not in this repo — it belongs in Athena.
- **`shm_region.{hpp,cpp}` now exists in three places** — here, the backend, and
  the Athena package. It needs one home before any of this goes upstream;
  detray is the natural candidate.
