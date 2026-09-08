#!/bin/bash
# Build the Triton backend against the ATLAS release instead of the .sif.
#
# The release's packaging needs five repairs, all in release-build.patch:
#   - traccc::opts::detector -> plain strings (the release has no options component)
#   - drop traccc::performance from the link (also absent, and unused)
#   - TritonCommon: its Config references a triton-common-json target it does not
#     install AND uses it as the include guard, so find_package either errors or
#     imports nothing. Include its Targets file directly instead.
#   - add find_package(Threads) and find_package(CUDAToolkit)
#   - generalise the Eigen workaround: the release does not install the eigen3
#     directory the exported targets advertise
#
# Build and run on the SAME node: /tmp is node-local, the result is -march=native.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

echo "repo    : ${REPO}"
echo "release : ${RELEASE}"
echo "build   : ${BACKEND_BUILD}"
echo "node    : $(hostname)"
echo

mkdir -p "${BACKEND_SRC}" "${BACKEND_BUILD}"
rsync -a --delete --exclude=.git "${REPO}/" "${BACKEND_SRC}/"
patch -d "${BACKEND_SRC}" -p1 --forward < "${HERE}/release-build.patch"

# The repo hard-codes a NERSC geometry path as the geoDir default. There is no
# container here, so point it at the real directory on this node.
sed -i "s|/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-01/itk-geo/|${GEO}/|g" \
    "${BACKEND_SRC}/standalone/src/TracccGpuStandalone.hpp"

rb_asetup
echo "release ${AtlasVersion}  ${BINARY_TAG}  $(gcc --version | head -1)"

cmake -S "${BACKEND_SRC}/backend/traccc-gpu" -B "${BACKEND_BUILD}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH};${EXT};${EXT}/share/covfie/cmake;${LCG}"
cmake --build "${BACKEND_BUILD}" -- -j"$(nproc)"

echo
if [ -f "${BACKEND_BUILD}/libtriton_traccc.so" ]; then
    ls -lh "${BACKEND_BUILD}/libtriton_traccc.so"
    echo "unresolved symbols: $(ldd -r "${BACKEND_BUILD}/libtriton_traccc.so" 2>&1 | grep -c 'not found')"
    echo "OK -- now: ${HERE}/run_server_release.sh"
else
    echo "FAILED: no libtriton_traccc.so in ${BACKEND_BUILD}"; exit 1
fi
