# Shared settings for the release-build scripts. Sourced, never run directly.
#
# These scripts build and run the traccc Triton backend against an ATLAS ACTS
# release. The release ships tritonserver, traccc, detray, vecmem, covfie and the
# Triton API packages, all built by one team with one compiler -- so the server
# ends up on exactly the stack Athena uses, and the region's ABI gates pass by
# construction rather than by keeping two stacks in step.

RB_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the traccc-aaS repo, whether this folder sits inside it or beside it.
_marker="backend/traccc-gpu/CMakeLists.txt"
if [ -z "${REPO:-}" ]; then
    if   [ -f "${RB_HERE}/../${_marker}" ];               then REPO="$(cd "${RB_HERE}/.." && pwd)"
    elif [ -f "${RB_HERE}/../../${_marker}" ];            then REPO="$(cd "${RB_HERE}/../.." && pwd)"
    elif [ -f "${RB_HERE}/../../traccc-aaS/${_marker}" ]; then REPO="$(cd "${RB_HERE}/../../traccc-aaS" && pwd)"
    else
        echo "FATAL: cannot find the traccc-aaS repo near ${RB_HERE}." >&2
        echo "       Set REPO=/path/to/traccc-aaS and re-run." >&2
        exit 1
    fi
fi
RELEASE="${RELEASE:-main--ACTS,Athena,2026-09-07T2100}"
# GEO must hold detray_detector_{geometry,material_maps,surface_grids}.json,
# ITk_bfield.cvf, ITk_digitization_config.json and athenaIdentifierToDetrayMap.txt.
# prepare_geometry.sh stages them from GEO_SRC onto node-local disk.
GEO_SRC="${GEO_SRC:-/eos/project/a/atlas-eftracking/GPU/ITk_data/FinalReport}"
GEO="${GEO:-/tmp/${USER}/itk-geo}"

# Node-local: /tmp is per-machine, and the backend is compiled -march=native.
BACKEND_SRC="${BACKEND_SRC:-/tmp/${USER}/rel_backend_src}"
BACKEND_BUILD="${BACKEND_BUILD:-/tmp/${USER}/rel_backend_build}"
MODELS="${MODELS:-/tmp/${USER}/rel_models}"

# The region is produced by Athena, not by anything here: a
# JSONDeviceDetectorDescriptionProviderSvc with its SharedMemoryRegion property
# set. SHM names the region to adopt; empty means parse JSON instead.
SHM="${SHM:-}"

# The release's tritonserver is built WITHOUT HTTP or metrics -- it accepts only
# gRPC options. That also makes the lxplus port-8000 problem irrelevant here.
GRPC_PORT="${GRPC_PORT:-8001}"

# atlasLocalSetup.sh and asetup return non-zero on success paths, so they must
# not run under `set -e`, or the script exits silently at that point. Sets
# AtlasVersion, BINARY_TAG, AtlasExternalsArea, CMAKE_PREFIX_PATH, the LCG
# compiler and CUDA.
rb_asetup() {
    set +e
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source "${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh" --quiet >/dev/null 2>&1
    asetup ${RELEASE} >/dev/null 2>&1
    set -e
    [ -n "${AtlasVersion}" ] || { echo "FATAL: asetup ${RELEASE} failed" >&2; exit 1; }
    EXT="${AtlasExternalsArea}/InstallArea/${BINARY_TAG}"
    # asetup puts only Athena's own areas on CMAKE_PREFIX_PATH; LCG packages
    # reach Athena through its own cmake layer, which a bare project does not
    # use. Without this, Boost resolves to the system 1.75 instead of 1.91.
    LCG=$(echo "${ROOT_INCLUDE_PATH}" | tr ':' '\n' \
          | grep -oE "^.*/${BINARY_TAG}" | sort -u | tr '\n' ';')
    export EXT LCG
}

rb_require_geo() {
    [ -f "${GEO}/detray_detector_geometry.json" ] || {
        echo "FATAL: no geometry in ${GEO}." >&2
        echo "       Run prepare_geometry.sh on this node." >&2
        exit 1; }
}
