#!/bin/bash
# Build the producer against the ATLAS release, so it writes a region the
# release-built backend can adopt.
#
# This matters: a producer built in the .sif carries vecmem 1.25, whose
# jagged_vector_view has a different layout from 1.27 (size_type went from
# std::size_t to unsigned int). The backend's gates would refuse such a region,
# correctly -- detray's grids are jagged containers and the offsets would be wrong.
#
# shm_region.{hpp,cpp} come from the repo -- the same files the backend compiles,
# so the two sides cannot disagree about the region layout.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

REGION_SRC="${REPO}/standalone/src"
[ -f "${REGION_SRC}/shm_region.hpp" ] || {
    echo "FATAL: no shm_region.hpp in ${REGION_SRC}"; exit 1; }

mkdir -p "${PRODUCER_SRC}" "${PRODUCER_BUILD}"
cp -f "${HERE}/CMakeLists.txt" "${PRODUCER_SRC}/"
cp -f "${REPO}/detray_shm/producer.cpp" "${REPO}/detray_shm/shm_memory_resource.hpp" "${PRODUCER_SRC}/"
cp -f "${REGION_SRC}/shm_region.hpp" "${REGION_SRC}/shm_region.cpp" "${PRODUCER_SRC}/"

rb_asetup
echo "release ${AtlasVersion}  ${BINARY_TAG}  $(gcc --version | head -1)"

cmake -S "${PRODUCER_SRC}" -B "${PRODUCER_BUILD}" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH};${EXT};${LCG}"
cmake --build "${PRODUCER_BUILD}" -- -j"$(nproc)"

echo
if [ -f "${PRODUCER_BUILD}/rel_producer" ]; then
    ls -lh "${PRODUCER_BUILD}/rel_producer"
    echo "OK -- now: ${HERE}/run_producer_release.sh"
else
    echo "FAILED: no rel_producer in ${PRODUCER_BUILD}"; exit 1
fi
