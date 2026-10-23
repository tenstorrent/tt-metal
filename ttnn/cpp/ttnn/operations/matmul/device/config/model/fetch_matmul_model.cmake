# Fetches the matmul config model blob (README.md, "The model blob"): a one-layer OCI artifact on ghcr, downloaded by
# digest. An OCI layer's digest is the sha256 of its content, so the digest pins and verifies the file in one step.
#
# In:  MATMUL_MODEL_REPOSITORY  ghcr repository, e.g. tenstorrent/tt-metal/matmul-model
#      MATMUL_MODEL_SHA256      sha256 of the blob (its layer digest)
#      MATMUL_MODEL_FILE        where to put it
# A failed download only warns: without the blob, turning the model on logs a warning and keeps the default configs.
# Also runs standalone: cmake -DMATMUL_MODEL_REPOSITORY=... -DMATMUL_MODEL_SHA256=... -DMATMUL_MODEL_FILE=... -P <this>

function(tt_fetch_matmul_model repository sha256 file)
    if(EXISTS "${file}")
        file(SHA256 "${file}" have)
        if(have STREQUAL sha256)
            return()
        endif()
        message(STATUS "Matmul config model: ${file} is not sha256:${sha256}, fetching it again")
    endif()
    message(STATUS "Matmul config model: fetching ghcr.io/${repository}@sha256:${sha256}")

    # ghcr serves public packages to an anonymous pull token
    set(token_file "${file}.token")
    file(
        DOWNLOAD
            "https://ghcr.io/token?scope=repository:${repository}:pull"
            "${token_file}"
        STATUS status
        TIMEOUT 60
    )
    list(GET status 0 code)
    set(token "")
    if(code EQUAL 0)
        file(READ "${token_file}" response)
        string(JSON token ERROR_VARIABLE json_error GET "${response}" token)
    endif()
    file(REMOVE "${token_file}")
    if(token STREQUAL "")
        message(WARNING "Matmul config model: no pull token for ghcr.io/${repository} (${status}); not fetched")
        return()
    endif()

    # Download next to the target and rename, so an interrupted or wrong download never looks like the blob
    set(partial "${file}.partial")
    file(
        DOWNLOAD
            "https://ghcr.io/v2/${repository}/blobs/sha256:${sha256}"
            "${partial}"
        HTTPHEADER "Authorization: Bearer ${token}"
        STATUS status
        TIMEOUT 600
    )
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        file(REMOVE "${partial}")
        message(WARNING "Matmul config model: download of ghcr.io/${repository}@sha256:${sha256} failed (${status})")
        return()
    endif()
    file(SHA256 "${partial}" have)
    if(NOT have STREQUAL sha256)
        file(REMOVE "${partial}")
        message(WARNING "Matmul config model: downloaded blob is sha256:${have}, expected sha256:${sha256}")
        return()
    endif()
    file(RENAME "${partial}" "${file}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    tt_fetch_matmul_model("${MATMUL_MODEL_REPOSITORY}" "${MATMUL_MODEL_SHA256}" "${MATMUL_MODEL_FILE}")
endif()
