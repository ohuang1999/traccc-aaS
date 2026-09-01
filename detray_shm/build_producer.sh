#!/bin/bash
# Build the producer inside the image, so it links against the same
# traccc/detray/vecmem the backend does. No GPU needed for this step.
#
# shm_region.{hpp,cpp} are copied in from the repo -- the SAME files the backend
# compiles -- so producer and consumer cannot disagree about the region layout.
# A local copy free to drift would mean the two processes read each other's data
# as garbage, with no error.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
d3_require_sif

REGION_SRC="${REPO}/standalone/src"
[ -f "${REGION_SRC}/shm_region.hpp" ] || {
    echo "FATAL: no shm_region.hpp in ${REGION_SRC}"; exit 1; }

mkdir -p "${PRODUCER_SRC}" "${PRODUCER_BUILD}"
cp -f "${HERE}/producer.cpp" "${HERE}/shm_memory_resource.hpp" \
      "${HERE}/CMakeLists.txt" "${PRODUCER_SRC}/"
cp -f "${REGION_SRC}/shm_region.hpp" "${REGION_SRC}/shm_region.cpp" "${PRODUCER_SRC}/"

apptainer exec --bind "${PRODUCER_SRC}:/src" --bind "${PRODUCER_BUILD}:/build" \
    "${SIF}" bash -lc '
  set -e
  cd /build
  cmake -S /src -B . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/traccc/install
  cmake --build . -- -j"$(nproc)"
'

echo
if [ -f "${PRODUCER_BUILD}/d3_producer" ]; then
    ls -lh "${PRODUCER_BUILD}/d3_producer"
    echo "OK -- now: ${HERE}/run_producer.sh"
else
    echo "FAILED: no d3_producer in ${PRODUCER_BUILD}"; exit 1
fi
