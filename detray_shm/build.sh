#!/bin/bash
# Rebuild the Triton backend OUTSIDE the image, against the image's own
# traccc/detray/vecmem. The .sif is read-only and ships a baked-in
# libtriton_traccc.so; this produces our own, which run_server.sh loads instead,
# so the image never has to be rebuilt to test a change.
#
# Two constraints shape this:
#   - /eos is not visible inside the container, so the repo is staged on
#     node-local disk first.
#   - the backend is compiled -march=native, so the result only runs on the node
#     it was built on. Build and run on the same machine.
#
# The first configure clones three Triton repos from GitHub (FetchContent) and
# takes 10-20 min. Keep BACKEND_BUILD between runs: later builds reuse them and
# recompile only what changed.
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
d3_require_sif

echo "repo  : ${REPO}"
echo "src   : ${BACKEND_SRC}   (node-local copy)"
echo "build : ${BACKEND_BUILD}"
echo "node  : $(hostname)"
echo

mkdir -p "${BACKEND_SRC}" "${BACKEND_BUILD}"
rsync -a --delete --exclude=.git "${REPO}/" "${BACKEND_SRC}/"

# The repo hard-codes a NERSC geometry path as the geoDir default; the image's
# Dockerfile rewrites it to /traccc/itk-geometry/ at image build time
# (backend/Dockerfile line 27). Building the repo as-is therefore does NOT
# reproduce the image's backend -- it aborts at startup on a missing
# ITk_bfield.cvf. Apply the same substitution to the staged copy. The repo itself
# is left untouched.
sed -i 's|/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-01/itk-geo/|/traccc/itk-geometry/|g' \
    "${BACKEND_SRC}/standalone/src/TracccGpuStandalone.hpp"
grep -q '"/traccc/itk-geometry/"' "${BACKEND_SRC}/standalone/src/TracccGpuStandalone.hpp" \
    || { echo "FATAL: geoDir substitution did not apply"; exit 1; }

# The backend's CMakeLists pulls three Triton repos with FetchContent. Inside the
# image, git gets a 401 on the second request of the HTTP/2 handshake and starts
# asking for a username no public repo needs. curl to the same URL returns 200
# and the host's git works, so it is git's HTTP/2 path in this image that is
# broken; forcing HTTP/1.1 fixes both ls-remote and shallow clone.
# GIT_TERMINAL_PROMPT=0 turns any remaining failure into an error message
# instead of a prompt that cannot be answered.
apptainer exec --nv --bind "${BACKEND_SRC}:/src" --bind "${BACKEND_BUILD}:/build" \
    --env GIT_TERMINAL_PROMPT=0 \
    --env GIT_CONFIG_COUNT=1 \
    --env GIT_CONFIG_KEY_0=http.version \
    --env GIT_CONFIG_VALUE_0=HTTP/1.1 \
    "${SIF}" bash -lc '
  set -e
  cd /build
  cmake -S /src/backend/traccc-gpu -B . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/traccc/install
  cmake --build . -- -j"$(nproc)"
'

echo
if [ -f "${BACKEND_BUILD}/libtriton_traccc.so" ]; then
    ls -lh "${BACKEND_BUILD}/libtriton_traccc.so"
    echo "OK -- now: $(dirname "${BASH_SOURCE[0]}")/run_server.sh"
else
    echo "FAILED: no libtriton_traccc.so in ${BACKEND_BUILD}"; exit 1
fi
