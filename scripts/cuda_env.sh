# Source before building or running gds_lab:  . scripts/cuda_env.sh
#
# Selecting a CUDA toolkit by path is not enough. Ubuntu 24.04 ships
# /usr/bin/nvcc from CUDA 12.0, which cannot target sm_120 (RTX 5080/5090), so
# every candidate prefix is checked against the installed GPU's compute
# capability before it is accepted.
#
# POSIX sh on purpose: sourced from bash/zsh, and from make via $(shell ...).

# Where scripts/install_cuda_redist.sh puts a user-local toolkit.
: "${GDSLAB_CUDA_PREFIX:=/data/cuda-13.0}"

# compute_cap "12.0" -> "compute_120"; fails when no GPU/driver is visible.
gdslab_gpu_arch() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
          head -1 | tr -d ' ')
    [ -n "$_cc" ] || return 1
    printf 'compute_%s' "$(printf '%s' "$_cc" | tr -d '.')"
}

# $1 = nvcc path, $2 = required compute_XXX (empty = accept any nvcc)
gdslab_nvcc_targets() {
    [ -x "$1" ] || return 1
    [ -n "$2" ] || return 0
    "$1" --list-gpu-arch 2>/dev/null | grep -qx "$2"
}

# One prefix per line, best first. Versioned directories come from `find` with a
# quoted pattern rather than a shell glob, because zsh aborts on no match.
gdslab_cuda_candidates() {
    printf '%s\n' "${CUDA_HOME:-}" "$GDSLAB_CUDA_PREFIX" /usr/local/cuda /opt/cuda
    find /usr/local /opt -maxdepth 1 -name 'cuda-*' -type d 2>/dev/null | sort -rV
    if command -v nvcc >/dev/null 2>&1; then
        dirname "$(dirname "$(command -v nvcc)")"
    fi
    return 0
}

# Prints "ok <prefix>" for a toolkit that can target this GPU, or
# "stale <prefix>" when only an nvcc too old for this GPU was found.
gdslab_find_cuda() {
    _want=$(gdslab_gpu_arch 2>/dev/null) || _want=""
    gdslab_cuda_candidates | {
        _fallback=""
        while IFS= read -r _prefix; do
            [ -n "$_prefix" ] || continue
            [ -x "$_prefix/bin/nvcc" ] || continue
            [ -n "$_fallback" ] || _fallback="$_prefix"
            if gdslab_nvcc_targets "$_prefix/bin/nvcc" "$_want"; then
                printf 'ok %s\n' "$_prefix"
                return 0
            fi
        done
        [ -n "$_fallback" ] && printf 'stale %s\n' "$_fallback"
        return 1
    }
}

_gdslab_found=$(gdslab_find_cuda)
case "$_gdslab_found" in
    ok\ *)
        CUDA_HOME=${_gdslab_found#ok }
        export CUDA_HOME
        PATH="$CUDA_HOME/bin:$PATH"
        # libcufile redist ships gdscheck.py / gds_stats / gdsio here, not in bin/.
        [ -d "$CUDA_HOME/tools" ] && PATH="$CUDA_HOME/tools:$PATH"
        export PATH
        for _libdir in "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
            [ -d "$_libdir" ] || continue
            LD_LIBRARY_PATH="$_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        done
        export LD_LIBRARY_PATH
        ;;
    stale\ *)
        echo "cuda_env.sh: ${_gdslab_found#stale }/bin/nvcc cannot target $(gdslab_gpu_arch 2>/dev/null)" >&2
        echo "cuda_env.sh: install a newer toolkit with scripts/install_cuda_redist.sh" >&2
        ;;
    *)
        echo "cuda_env.sh: no CUDA toolkit found (looked in \$CUDA_HOME, $GDSLAB_CUDA_PREFIX, /usr/local/cuda, /opt/cuda, \$PATH)" >&2
        ;;
esac
unset _gdslab_found _cc _want _prefix _fallback _libdir
