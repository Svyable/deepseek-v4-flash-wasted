#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# tests/run.sh — every check we have, in one place, exiting non-zero on the
# first real failure.
#
# MODIFIED from the upstream WASTE import (sqliteai/waste @ d9b919a) by the
# deepseek-v4-flash-wasted authors, as Apache-2.0 §4(b) requires. Added the
# "quantization decode", "checkpoint inventory" and "DeepSeek gate replays"
# sections and a recursive SPDX check. See UPSTREAM.md.
#
# Written after losing time twice to checks that silently did not run: once
# to objects compiled against a stale header, once to a stale test binary.
# So this rebuilds first, states what it is about to do, and never treats a
# missing prerequisite as a pass — it says SKIP, loudly.
#
#   tests/run.sh [model.waste]
#
# Env: WASTE_REF_MODEL  container to use for the end-to-end checks
#      WASTE_REF_SRC    source weights, for the container round-trip
#      WASTE_ORACLE     logits dumped by tools/kimi_ref.py for THE SAME
#                       token ids this script uses (see IDS below) — a dump
#                       from a different prompt will look like an engine bug

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${WASTE_REF_MODEL:-$HOME/models/kimi-linear.waste}}"
SRC="${WASTE_REF_SRC:-/Volumes/WasteDisk/kimi-linear}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Without a reference container, build a synthetic one: a few megabytes of
# deterministic noise in the real format. It cannot check the engine against
# the oracle — those logits belong to actual Kimi-Linear weights — but every
# check that compares the engine against itself works on it, which is what
# lets CI and a fresh clone run the engine at all instead of skipping it.
SYNTHETIC=0
if [ ! -d "$MODEL" ]; then
    if python3 tools/make_test_container.py "$TMP/tiny.waste" >/dev/null 2>&1; then
        MODEL="$TMP/tiny.waste"
        SYNTHETIC=1
    fi
fi

pass=0; fail=0; skip=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; pass=$((pass+1)); }
no()   { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=$((fail+1)); }
sk()   { printf "  \033[33mSKIP\033[0m  %s — %s\n" "$1" "$2"; skip=$((skip+1)); }
head_() { printf "\n\033[1m%s\033[0m\n" "$1"; }

head_ "build"
if make -s test >/dev/null 2>&1 && make -s >/dev/null 2>&1; then
    ok "make && make test"
else
    no "build failed"
    make test 2>&1 | grep -E "error" | head -5
    exit 1
fi

# ---------------------------------------------------------------- unit ----
head_ "kernels vs the reference implementations"

# Not a kernel, but it decides the default budget on every containerized
# host and runs in milliseconds against synthetic files, so it runs early
# and unconditionally. All of it runs everywhere: the reader takes its
# paths as parameters, so the cases that only fire on Linux are still
# compiled and checked on the platforms we actually develop on.
if ./test_memory "$TMP" 2>/dev/null | grep -q "^PASS"; then
    ok "cgroup-v2 limits vs the automatic budget's ceiling"
else
    no "cgroup-v2 limits"
fi

# The cpu list, in two halves with different reaches. Parsing runs
# everywhere and is the half that matters most: a typo in a cpu list is
# indistinguishable from the option not helping. Binding needs a platform
# that has the call, so it is a SKIP on macOS rather than a pass — exit 77,
# the one code this suite reads as "did not run".
if ./test_cpus parse >/dev/null 2>&1; then
    ok "cpu list parsing, including the typos that must be refused"
else
    no "cpu list parsing"
    ./test_cpus parse 2>&1 | head -3
fi

./test_cpus bind >"$TMP/cpus.log" 2>&1
case $? in
    0)  ok "compute pool binds to a cpu list ($(cat "$TMP/cpus.log"))" ;;
    77) sk "compute pool binds to a cpu list" \
           "$(sed 's/^SKIP: //' "$TMP/cpus.log" | head -1)" ;;
    *)  no "compute pool binds to a cpu list"; head -3 "$TMP/cpus.log" ;;
esac

if command -v uv >/dev/null 2>&1; then
    ./test_k3parts "$TMP/k3parts.bin" >/dev/null 2>&1
    if uv run --quiet --with torch --no-project python tools/k3parts_ref.py \
           "$TMP/k3parts.bin" 2>/dev/null | grep -q "^PASS"; then
        ok "K3 components (SiTU, both decay gates, AttnRes)"
    else
        no "K3 components"
    fi

    if KDA_T=24 KDA_H=4 KDA_K=32 KDA_V=32 uv run --quiet --with torch \
           --with fla-core --with einops --no-project python tools/kda_ref.py \
           2>/dev/null | grep -q "^PASS"; then
        ok "KDA kernel vs fla's naive_recurrent_kda"
    else
        no "KDA kernel"
    fi
else
    sk "kernel checks" "uv not installed"
fi

# -------------------------------------------------------------- download ----
head_ "download script"

# The shard downloader is the one tool whose failure modes only appear
# hours into a 1.4 TB pull, so its worker is exercised here against a local
# server instead: resume from a truncated file, skip an already-recorded
# shard without a request, and fall back to a clean restart when the server
# has no Range support.
fetch_worker() {                       # $1 = generated worker path
    { echo "DEST=$FT/dst; RAW=http://127.0.0.1:$1; STATE=\$DEST/.st; LOG=\$DEST/log"
      echo 'MAX_RETRY=2; MIN_FREE_GB=0; STAT_MODE='"$2"
      sed -n '/^cat > "\$DEST\/\.worker\.sh" <<WORKER$/,/^WORKER$/p' tools/fetch_weights.sh
    } > "$FT/gen.sh"
    bash "$FT/gen.sh"
}

# The port comes from the kernel, not from this file. Two fixed ones (8731,
# 8732) cost an afternoon on a machine already running something on the
# first: range_server.py could not bind, the worker talked to whatever else
# was listening, and the failure surfaced as "resume" and "state-file skip"
# — two checks blaming the code under test for the environment. Nothing
# here needs a *particular* port, only a free one, so ask for one and read
# back what was given. The `sleep 1` this replaces was the other half of
# the same bug: a fixed wait is either too long or, on a loaded machine,
# not long enough.
start_server() {                       # $@ = extra server args; sets PORT, RSRV
    local pf="$FT/port"
    rm -f "$pf" "$pf.tmp"
    python3 tests/range_server.py "$FT/srv" 0 --port-file "$pf" "$@" \
        >/dev/null 2>&1 &
    RSRV=$!
    PORT=
    for _ in $(seq 100); do            # 10 s, in 100 ms steps
        [ -s "$pf" ] && { PORT=$(cat "$pf"); break; }
        kill -0 "$RSRV" 2>/dev/null || break   # it died; stop waiting
        sleep 0.1
    done
    # A server that never reported is still a process, and the caller below
    # only kills the ones it was told about — CI reported exactly one
    # orphaned Python the first time this path fired.
    [ -n "$PORT" ] || { kill "$RSRV" 2>/dev/null; wait "$RSRV" 2>/dev/null; }
    [ -n "$PORT" ]
}

FT="$TMP/fetch"
mkdir -p "$FT/srv" "$FT/dst"
if stat --version >/dev/null 2>&1; then SM=gnu; else SM=bsd; fi
# The path goes in argv, not into the source: MSYS2 rewrites POSIX paths in
# a native program's arguments and cannot rewrite one quoted inside -c, so
# Windows Python was handed /tmp/... verbatim and could not find it.
python3 -c "import sys; open(sys.argv[1],'wb').write(bytes(range(256))*4096)" "$FT/srv/s.bin"

# The worker the downloader generates is curl, so without curl these three
# report the downloader broken when what is missing is a tool — which is the
# one thing this suite says it must never do. Debian slim and a bare MinGW
# both lack it.
if ! command -v curl >/dev/null 2>&1; then
    sk "a truncated shard resumes and verifies against Content-Length" "curl not installed"
    sk "a completed shard is skipped without a request" "curl not installed"
    NO_CURL=1
elif ! start_server; then
    no "range server did not start (resume, state-file skip not run)"
else
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if cmp -s "$FT/srv/s.bin" "$FT/dst/s.bin" && grep -q "^s.bin$" "$FT/dst/.st"; then
        ok "a truncated shard resumes and verifies against Content-Length"
    else
        no "resume"
    fi
    # second pass: recorded in the state file, so not even a HEAD goes out
    before=$(wc -l < "$FT/dst/log")
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if [ "$(wc -l < "$FT/dst/log")" = "$before" ]; then
        ok "a completed shard is skipped without a request"
    else
        no "state-file skip"
    fi
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

if [ -n "${NO_CURL:-}" ]; then
    sk "a server without Range support restarts the shard instead of giving up" "curl not installed"
elif ! start_server --no-range; then
    no "range server did not start (no-range fallback not run)"
else
    rm -f "$FT/dst/s.bin" "$FT/dst/.st" "$FT/dst/log"
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if cmp -s "$FT/srv/s.bin" "$FT/dst/s.bin"; then
        ok "a server without Range support restarts the shard instead of giving up"
    else
        no "no-range fallback"
    fi
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

# ---------------------------------------------------------------- image ----
head_ "image"

# No model needed: the loader is checked against closed-form arithmetic on
# a solid colour it writes itself.
if ./test_image "$TMP" 2>/dev/null | grep -q "^IMAGE OK"; then
    ok "decode, patch grid, normalization and rejection of non-images"
else
    no "image loader"
fi

# ------------------------------------------------------------ container ----
head_ "container"

if [ -d "$MODEL" ]; then
    # One bank per call, and the count of records read is checked as well
    # as the count of problems. The glob used to be passed whole, so
    # `experts-L2.bin` landed where the record count goes, atoi made it 0,
    # and the check reported "0 records read, 0 problems" — a pass, from a
    # run that opened one file and read nothing out of it.
    banks=0; recs=0; bad=0
    for bank in "$MODEL"/experts-L*.bin; do
        [ -f "$bank" ] || continue
        banks=$((banks + 1))
        out=$(./test_container "$bank" 2 2>/dev/null) || { bad=1; continue; }
        n=$(printf '%s' "$out" | sed -n 's/^\([0-9]*\) records read, \([0-9]*\) problems$/\1 \2/p')
        set -- $n
        [ "${1:-0}" -gt 0 ] || bad=1
        [ "${2:-1}" -eq 0 ] || bad=1
        recs=$((recs + ${1:-0}))
    done
    if [ "$banks" -gt 0 ] && [ "$bad" = 0 ]; then
        ok "expert records read through the C structs ($recs records over $banks banks)"
    elif [ "$banks" = 0 ]; then
        no "expert record layout — no expert bank in $MODEL"
    else
        no "expert record layout"
    fi

    if [ "$SYNTHETIC" = 1 ]; then
        sk "container round-trip" "synthetic container has no source weights"
    elif [ -d "$SRC" ] && command -v uv >/dev/null 2>&1; then
        if uv run --quiet --with torch --no-project python tools/verify_container.py \
               --container "$MODEL" --src "$SRC" --experts 1 2>/dev/null \
               | grep -q "^PASS"; then
            ok "dequantized weights match the source"
        else
            no "container round-trip"
        fi
    else
        sk "container round-trip" "source weights not at $SRC"
    fi
else
    sk "container checks" "no container at $MODEL"
fi

# The checksum is only worth writing if something reads it. Every expert
# record carries a crc32, and until it was checked on the read path a
# container that rotted after conversion answered with whatever the
# damaged bytes decoded to — which on a single flipped bit is the same
# argmax and slightly different numbers, i.e. invisible.
#
# Its own container, built here and thrown away: the check has to damage
# one, and neither the reference container nor CI's should be the victim.
CRC="$TMP/crc.waste"
if python3 tools/make_test_container.py "$CRC" >/dev/null 2>&1; then
    IDS_CRC=3,7,11,5,9,13,2,17
    damage() {                        # $1 = bank file, $2 = byte to flip
        python3 - "$1" "$2" <<'PY'
import struct, sys
p = sys.argv[1]
d = bytearray(open(p, "rb").read())
rec = struct.unpack_from("<I", d, 16)[0] * 4096     # rec_4k_blocks
# Every record, because a prompt only routes to some of the experts and a
# check that depends on which ones is a check that passes by luck.
for e in range(len(d) // rec):
    d[e * rec + int(sys.argv[2])] ^= 0x01
open(p, "wb").write(d)
PY
    }

    if ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 >/dev/null 2>&1; then
        damage "$CRC"/experts-L1.bin 148          # inside the gate payload
        out=$(WASTE_VERIFY=1 ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 2>&1); rc=$?
        if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "checksum mismatch" &&
           printf '%s' "$out" | grep -qE "expert [0-9]+ of layer 1"; then
            ok "with verification on, a flipped bit is an error that names the record"
        else
            no "corrupted expert record not caught with verification on (rc=$rc)"
        fi
        # The default is off, and asserting it is the point: the same
        # container runs to completion and answers, because the checksum
        # costs ~5% and is a choice the caller makes. If this ever starts
        # failing, the default flipped without anyone deciding to.
        if ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 >/dev/null 2>&1; then
            ok "by default the checksum is not read, and a damaged record answers"
        else
            no "verification is on by default — it is meant to be opt-in"
        fi
    else
        no "the undamaged synthetic container does not run"
    fi

    # A bank cut short of a whole record: the pread succeeds against the
    # record before it, so only the header identity catches this one — and
    # no WASTE_VERIFY here, deliberately. The header checks are O(1) and
    # always on; only the checksum is opt-in.
    if python3 tools/make_test_container.py "$TMP/trunc.waste" >/dev/null 2>&1; then
        python3 - "$TMP/trunc.waste/experts-L1.bin" <<'PY'
import os, sys
os.truncate(sys.argv[1], os.path.getsize(sys.argv[1]) - 4096)
PY
        out=$(./test_forward "$TMP/trunc.waste" "$IDS_CRC" /dev/null 0 2>&1); rc=$?
        if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE "short read|record header"; then
            ok "a bank truncated mid-record is refused"
        else
            no "truncated bank not caught (rc=$rc)"
        fi
    fi
else
    sk "expert record verification" "cannot build a synthetic container"
fi

# A container with a tensor_prefix carries tensors the loader declines to
# load, and that skip used to leave `group` at zero for them while the
# row-scratch sizing divided by it. The architecture decided what that
# meant: arm64's sdiv answers 0, x86's idiv raises #DE — so `waste info` on
# K3 was an instant SIGFPE on every x86 build, Linux included, while this
# suite stayed green on a container that has no prefix and therefore no
# tensor that takes the skip (issue #10). This check is load-bearing on x86
# in any build, since the process dies; on arm64 it needs `make asan`,
# whose UBSan reports the division whatever the hardware does with it.
if python3 tools/make_test_container.py "$TMP/pfx.waste" \
        --prefix language_model. >/dev/null 2>&1; then
    if ./waste info "$TMP/pfx.waste" >/dev/null 2>&1; then
        ok "a container whose tensors are not all under its prefix loads"
    else
        no "a prefixed container does not load"
    fi
else
    sk "prefixed container" "cannot build a synthetic container"
fi

# ----------------------------------------------------------------- e2e ----
head_ "engine"

if [ -d "$MODEL" ]; then
    # The oracle fixture pins these to Kimi-Linear's vocabulary; a synthetic
    # container has 256 entries, and an id past the table is an out-of-range
    # read rather than a different answer.
    if [ "$SYNTHETIC" = 1 ]; then
        IDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
    else
        IDS=1008,10484,318,15383,387,11,316,276,10484,318,19509,387,31082,13,646,10484
    fi

    ./test_forward "$MODEL" "$IDS" "$TMP/seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 ./test_forward "$MODEL" "$IDS" "$TMP/chunk.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/chunk.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
d = max(abs(x - y) for x, y in zip(a, b))
sys.exit(0 if d < 1e-3 and a.index(max(a)) == b.index(max(b)) else 1)
PY
    then ok "chunked prefill == token-at-a-time"
    else no "chunked prefill diverges"
    fi

    # WASTE_Q8=0 makes the entire trunk resident as f32 — 8x a 4-bit one, so
    # K3's 26 GiB trunk asks for ~210 GB and no host runs this. Say so
    # instead of reporting the refusal as a divergence.
    q8_why=$(python3 - "$MODEL" <<'PY'
import json, os, subprocess, sys
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))
try:
    m = json.load(open(os.path.join(sys.argv[1], "manifest.json")))
except Exception:
    sys.exit(0)
need = 0
for t in m.get("trunk", []):
    # what src/model.c leaves out of the resident set at load, in both
    # modes, is not part of what this would allocate: the tower, which the
    # text path never touches, and embed_tokens, whose rows are pread one
    # per token — 1.41 GiB of a 7.50 GiB f32 Kimi-Linear trunk
    name = t.get("name", "")
    if name.startswith(("vision_tower.", "mm_projector.")) or \
       name.endswith("embed_tokens.weight"):
        continue
    n = 1
    for d in t.get("shape", []):
        n *= d
    need += n * 4
r = subprocess.run([WASTE, "plan", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    # usable, not physical: this decides whether the host can hold an f32
    # trunk, and in a container the host's RAM is not what it can hold
    phys = json.loads(r.stdout)["usable_ram_bytes"]
except Exception:
    phys = 0
if phys and need > phys // 2:
    print(f"an f32 trunk is {need / 2**30:.0f} GB of {phys / 2**30:.0f} GB of RAM")
PY
)
    if [ -n "$q8_why" ]; then
        sk "quantized trunk storage == f32 weights" "$q8_why"
    else
        WASTE_Q8=0 ./test_forward "$MODEL" "$IDS" "$TMP/f32.bin" 0 >/dev/null 2>&1
        if python3 - "$TMP/seq.bin" "$TMP/f32.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "quantized trunk storage == f32 weights"
        else no "quantized storage changes results"
        fi
    fi

    WASTE_BACKEND=cpu ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cpu.bin"; then
        ok "SIMD backend bit-identical to the CPU baseline"
    else
        # a difference here is allowed to be tiny, but must be tiny
        if python3 - "$TMP/seq.bin" "$TMP/cpu.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "SIMD backend matches the CPU baseline (within fp noise)"
        else no "SIMD backend diverges from the CPU baseline"
        fi
    fi

    WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" "$TMP/cache.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cache.bin"; then
        ok "expert cache is bit-identical to no cache"
    else
        no "expert cache changes results"
    fi

    # Read-ahead is on by default, so the synchronous path — the fallback,
    # and the thing every earlier measurement was made on — is the one no
    # check would otherwise run. It shipped broken for exactly one build:
    # a synchronous claim took a pin that never expired, the victim sampler
    # ran out of slots, and the forward pass answered with the experts it
    # had instead of failing.
    WASTE_IO_THREADS=0 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/sync.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/cache.bin" "$TMP/sync.bin"; then
        ok "read-ahead is bit-identical to synchronous reads"
    else
        no "read-ahead changes results"
    fi

    # A trace-driven simulator is only worth having if it models *this*
    # cache. Before this check it did not: it kept a frequency count across
    # evictions that ec_claim resets and sampled 32 victims where EC_SAMPLE
    # is 16, and read 36.6% where the engine measured 30.4%. A simulator
    # that disagrees quietly is how a policy question gets the wrong answer
    # for a week, so the agreement is asserted rather than remembered.
    if [ "$SYNTHETIC" != 1 ] && command -v python3 >/dev/null 2>&1; then
        TR="$TMP/sim.trace"
        rm -f "$TR"
        eng=$(WASTE_LOOKAHEAD=0 WASTE_DUMP_ROUTE="$TR" WASTE_CACHE_MB=1024 \
              ./test_forward "$MODEL" "$(echo "$IDS" | tr ' ' ',')" /dev/null 8 2>&1 \
              | sed -n 's/.*= \([0-9.]*\)% hit.*/\1/p')
        sim=$(python3 tools/routing_stats.py simulate "$TR" --data "$MODEL" \
              --cache-gb 1.0 2>/dev/null |
              awk '/%/ && NF == 5 { v = $4 } END { gsub("%", "", v); print v }')
        if [ -n "$eng" ] && [ -n "$sim" ] && python3 -c "
import sys; sys.exit(0 if abs($eng - $sim) <= 5 else 1)" 2>/dev/null; then
            ok "trace simulator agrees with the engine's cache (${eng}% vs ${sim}%)"
        else
            no "trace simulator disagrees with the engine (${eng:-?}% vs ${sim:-?}%)"
        fi
    else
        sk "trace simulator vs the engine" "needs a real container and python3"
    fi

    # The router lookahead starts reads on a guess. The guess must never
    # reach the arithmetic: the real router stays authoritative and the
    # prefetch only decides when bytes move, so the logits cannot shift.
    WASTE_LOOKAHEAD=6 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/look.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/cache.bin" "$TMP/look.bin"; then
        ok "router lookahead is bit-identical to no lookahead"
    else
        no "router lookahead changes results"
    fi

    # A purged slot reads back as zeros, so the whole prototype rests on the
    # engine noticing before it multiplies one. This does not create memory
    # pressure — it checks that the volatile/nonvolatile traffic itself does
    # not disturb a record. Vacuously true off macOS, where the flag is a
    # no-op and says so.
    WASTE_PURGEABLE=1 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/purge.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/cache.bin" "$TMP/purge.bin"; then
        ok "purgeable slots are bit-identical to ordinary ones"
    else
        no "purgeable slots change results"
    fi

    # kimi_ref.py computes its logits *from* a WASTE container, so an oracle
    # is only comparable against the container that produced it — and not
    # merely against its trunk width. The expert codebooks are k-means, and
    # the same seed on a different --device trains different books: splicing
    # one layer of cpu-trained books into an mps container moved the logits
    # by 1.24 max against this 1e-3 threshold. No shipped fixture can be
    # portable across conversions, so generate one from the container under
    # test — 16.9 s on Kimi-Linear, and it needs the container, not the
    # 92 GB of source shards. The fixture stays for hosts without uv, where
    # its provenance has to be checked instead (see the sidecar): a
    # cross-conversion diff reads as an engine bug and is not one.
    ORACLE="${WASTE_ORACLE:-tests/fixtures/oracle_kimilinear_16tok.bin}"
    oracle_arch=$(./waste info "$MODEL" --json 2>/dev/null | python3 -c \
        "import json,sys; print(json.load(sys.stdin).get('arch',''))" 2>/dev/null || true)
    if [ "$SYNTHETIC" = 1 ]; then
        sk "engine vs the PyTorch oracle" "synthetic container has no reference logits"
    elif [ -n "$oracle_arch" ] && [ "$oracle_arch" != "kimi-linear" ]; then
        # the ids above and the fixture's vocabulary are Kimi-Linear's
        sk "engine vs the PyTorch oracle" \
           "the oracle prompt is Kimi-Linear's, this container is $oracle_arch"
    else
        GEN=""
        if [ -z "${WASTE_ORACLE:-}" ] && command -v uv >/dev/null 2>&1; then
            uv run --no-project --with torch --with fla-core --with einops \
                python tools/kimi_ref.py --container "$MODEL" --prompt-ids "$IDS" \
                --tokens 0 --dump "$TMP/oracle.bin" >/dev/null 2>&1 || true
            [ -s "$TMP/oracle.bin" ] && GEN="$TMP/oracle.bin"
        fi
        oracle_why=""
        if [ -z "$GEN" ] && [ -z "${WASTE_ORACLE:-}" ] && [ -f "${ORACLE%.bin}.json" ]; then
            oracle_why=$(python3 - "$MODEL" "${ORACLE%.bin}.json" <<'PY'
import json, os, subprocess, sys
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))
meta = json.load(open(sys.argv[2]))
want = meta.get("trunk")
r = subprocess.run([WASTE, "info", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    got = json.loads(r.stdout)["quantization"].split("trunk ", 1)[1]
except Exception:
    sys.exit(0)                       # let the diff speak if info cannot
if want and got != want:
    print(f"no uv to generate one, and the fixture is from a {want} trunk "
          f"against this container's {got} — see {os.path.basename(sys.argv[2])}")
PY
)
        fi
        if [ -n "$oracle_why" ]; then
            sk "engine vs the PyTorch oracle" "$oracle_why"
        elif [ -n "$GEN" ] || [ -f "$ORACLE" ]; then
            if python3 - "$TMP/seq.bin" "${GEN:-$ORACLE}" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
            then
                if [ -n "$GEN" ]
                then ok "engine matches a PyTorch oracle built from this container"
                else ok "engine matches the shipped PyTorch oracle"
                fi
            else no "engine diverges from the oracle"
            fi
        else
            sk "oracle diff" "no fixture; regenerate with tools/kimi_ref.py --dump"
        fi
    fi

    if ./test_state "$MODEL" 2>/dev/null | grep -q "^STATE OK"; then
        ok "saved session resumes identically"
    else
        no "session state does not round-trip"
    fi

    # learned hotlist: a second run should start warmer than the first
    if [ "$SYNTHETIC" = 1 ]; then
        sk "learned hotlist" "synthetic container carries no tokenizer"
    else
    rm -f "$MODEL/usage.waste"
    # Read hits and misses together. Counting misses alone cannot tell a
    # perfect warm run from a run that never happened, and both come out 0:
    # a 4-bit trunk leaves enough of the 5G budget for the whole working
    # set, the hotlist lands 100%, and demanding "misses > 0" failed the
    # best result the check can produce. The empty line is the real
    # "nothing ran" signal.
    cold=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G --learn 2>&1 \
           | grep -oE "[0-9]+ hit / [0-9]+ miss" || true)
    warm=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G 2>&1 \
           | grep -oE "[0-9]+ hit / [0-9]+ miss" || true)
    rm -f "$MODEL/usage.waste"
    cold_m=${cold##*/ }; cold_m=${cold_m% miss}
    warm_m=${warm##*/ }; warm_m=${warm_m% miss}
    if [ -z "$cold" ] || [ -z "$warm" ]; then
        no "hotlist check read no cache line from the run"
    elif [ "$cold_m" -eq 0 ] && [ "$warm_m" -eq 0 ]; then
        # The cold run had nothing to teach, so neither outcome is
        # demonstrated. PASS here would be passing by luck, and this suite
        # does not treat an absent prerequisite as one.
        sk "learned hotlist" "the cold run already missed nothing"
    elif [ "$warm_m" -lt "$cold_m" ]; then
        ok "learned hotlist warms the cache ($cold_m -> $warm_m misses)"
    else
        no "hotlist did not reduce misses ($cold_m -> $warm_m)"
    fi
    fi
else
    sk "engine checks" "no container at $MODEL"
fi

# --------------------------------------------------------------- budget ----
head_ "RAM budget"

# The default budget is the one path check_budget.sh cannot reach, because
# it always passes --budget. With no flag the engine chooses, and that
# choice is all that stands between a model whose recommendation exceeds
# the machine — K3 asks for 80.63 GB — and a swap storm. So assert the
# rule, not a number, and it holds on any host: the engine steps down a
# whole token working set at a time and takes the largest of
# floor + 3x, 2x, 1x that fits under 7/8 of the RAM this process may use,
# or the floor when not even one multiple does, less at most one expert
# record of slot rounding. Filling the cap instead is what put a 27 GB
# cache on a 64 GB machine and cost 8x throughput — docs/LEARNED.md §16.
default_budget() {
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

# subprocess does not go through the shell, so it does not inherit Git-Bash's
# habit of resolving a bare name to the .exe next to it.
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))

def j(*a):
    r = subprocess.run([WASTE, *a, "--json"], capture_output=True, text=True)
    return json.loads(r.stdout)

plan, info = j("plan", sys.argv[1]), j("info", sys.argv[1])
# From the engine rather than os.sysconf, which does not exist in Windows
# CPython at all and would read the host's RAM inside a container anyway.
# This is the same number the engine sized itself against, so what the
# check still tests is the rule — floor + the largest whole working set
# under 7/8 of it — and not the RAM reading, which has its own platform
# code and no business being written twice. It is a capacity and not a
# pressure reading, so it is the same in this process and in the `info`
# one below; a ceiling that moved between the two would make this check
# fail as an engine bug on any busy machine.
phys = plan["usable_ram_bytes"]
if not phys:
    print("usable RAM unknown on this host")
    sys.exit(0)
cap = phys - phys // 8
# what the engine actually holds: the plan's mandatory parts plus the
# cache it really allocated, which is what `info` reports
held = plan["floor_bytes"] - plan["min_expert_cache"] + info["expert_cache_bytes"]
# recommended_bytes is floor + 3 * one token's working set, by construction
ws = (plan["recommended_bytes"] - plan["floor_bytes"]) // 3
want = plan["floor_bytes"]
for k in (3, 2, 1):
    if plan["floor_bytes"] + ws * k <= cap:
        want = plan["floor_bytes"] + ws * k
        break
rec = plan["min_expert_cache"] // (2 * info["top_k"]) if info["top_k"] else 0
G = 1 << 30
print(f"{held/G:.2f} GB held, ceiling {want/G:.2f} GB, usable {phys/G:.2f} GB")
sys.exit(0 if want - rec - 1 <= held <= want else 1)
PY
}

# params_total is the number that ends up in a model card, and it is
# derived rather than stored: the routed experts, three matrices each and
# each as wide as the expert's input, plus the trunk that runs on every
# token. In a latent MoE that width is the latent, not the hidden — K3
# reported 5.44 T instead of 2.72 T for exactly one wrong field. Mirroring
# the engine's arithmetic here would only prove this script and the engine
# agree, so the expert count is also weighed against the bytes on disk: at
# 3 bits per weight the experts have to fit their bank, give or take one
# fp16 scale per output row and the record's 4 KiB alignment.
params_rule() {
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run([os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else "")), "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c, lay = man["config"], man["layers"]

width = c.get("routed_expert_hidden_size") or c["hidden_size"]
inter = c["moe_intermediate_size"]
per_expert = 3 * width * inter
moe_layers = c["num_hidden_layers"] - c.get("first_k_dense_replace", 0)

# the trunk, as the engine counts it: the language model only, and a token
# reads one row of the embedding table rather than all of it
pref = man.get("tensor_prefix", "")
trunk = trunk_active = 0
for t in man["trunk"]:
    if pref and not t["name"].startswith(pref):
        continue
    n = 1
    for s in t["shape"]:
        n *= s
    trunk += n
    if "embed_tokens.weight" not in t["name"]:
        trunk_active += n

total = per_expert * c["num_experts"] * moe_layers + trunk
active = per_expert * info["top_k"] * moe_layers + trunk_active

bits = man["expert_quant"]["bits_per_weight"]
rec = lay[next(iter(lay))]
on_disk = rec["bytes"] // rec["experts"]
lo = per_expert * bits // 8
hi = lo + 2 * (2 * inter + width) + 4096          # scales, then alignment

def h(x):
    if x >= 1e12: return f"{x/1e12:.2f} T"
    if x >= 1e9:  return f"{x/1e9:.2f} B"
    return f"{x/1e6:.2f} M"

print(f"{h(total)} total, {h(active)} active, {h(trunk)} trunk, "
      f"{on_disk/(1<<20):.2f} MiB/expert on disk")
sys.exit(0 if (moe_layers == len(lay)
               and info["params_total"] == total
               and info["params_active"] == active
               and lo <= on_disk <= hi) else 1)
PY
}

if [ -d "$MODEL" ]; then
    if ./waste plan "$MODEL" >/dev/null 2>&1; then ok "waste plan"; else no "waste plan"; fi
    # capture first: `set -o pipefail` would otherwise propagate the
    # deliberate non-zero exit of the command under test
    refusal=$(./waste run "$MODEL" x --budget 1M 2>&1 || true)
    if printf '%s' "$refusal" | grep -q "below the model's floor"; then
        ok "a budget under the floor is refused, not swapped into"
    else
        no "under-floor budget not refused"
    fi

    if out=$(default_budget "$MODEL" 2>/dev/null); then
        ok "no --budget picks a ceiling the machine can hold ($out)"
    else
        no "default budget off the rule (${out:-no output})"
    fi
    # A container from a future layout must be refused, not read against the
    # old rules — the field was written from the first converter and read by
    # nobody until it was wired up.
    FV=$(mktemp -d)
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m["format_version"] = 999
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out=$(./waste info "$FV" 2>&1); rc=$?
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m.pop("format_version", None)
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out2=$(./waste info "$FV" 2>&1); rc2=$?
    rm -rf "$FV"
    if [ "$rc" -ne 0 ] && [ "$rc2" -ne 0 ] &&
       printf '%s' "$out"  | grep -q "format version mismatch" &&
       printf '%s' "$out2" | grep -q "format version missing"; then
        ok "a container from another format version is refused"
    else
        no "format_version not enforced (rc=$rc rc2=$rc2)"
    fi

    if [ -n "${WASTE_SANITIZED:-}" ]; then
        sk "peak RSS inside the budget" "sanitizer shadow memory makes RSS meaningless"
    elif [ "$SYNTHETIC" = 1 ]; then
        sk "peak RSS inside the budget" "needs a tokenizer to drive the CLI"
    elif tests/check_budget.sh "$MODEL" 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the configured budget"
    else
        no "peak RSS exceeded the budget"
    fi
else
    sk "budget checks" "no container at $MODEL"
fi

# The small model cannot catch budget accounting that is wrong in
# proportion to the model: K3 overran by 2-3 GB on scratch that Kimi-Linear
# sizes in single megabytes. Run the same check against K3 when it is here.
#
# None of it survives a sanitizer build, and for two different reasons: RSS
# is meaningless next to ASan's shadow memory, and ASan's allocator refuses
# the 27 GB mapping the trunk needs at all, so anything that *opens* K3
# fails rather than measuring anything. Both mean SKIP. This only bites on
# a machine that has the K3 container — CI does not, which is why `make
# asan` stayed green there while failing on the developer's own laptop.
BIG="${BIG_MODEL:-$HOME/models/k3.waste}"
if [ -n "${WASTE_SANITIZED:-}" ] && [ -f "$BIG/manifest.json" ]; then
    sk "every K3 check" "sanitizer cannot open a 27 GB trunk"
    BIG=/nonexistent-under-sanitizer
fi
if [ -f "$BIG/manifest.json" ]; then
    if tests/check_budget.sh "$BIG" 32 long 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the budget on K3 too"
    else
        no "peak RSS exceeded the budget on K3"
    fi

    # K3 is the only model here whose recommendation can exceed the
    # machine, so it is the one that makes the step-down bite at all: on a
    # 64 GB host the default lands on floor + 1x working set = 46.24 GB,
    # rather than the floor + 3x = 80.63 GB asked for. On a host large
    # enough to hold the recommendation this still passes — it checks the
    # rule, not the clamp.
    if out=$(default_budget "$BIG" 2>/dev/null); then
        ok "no --budget is capped to the machine on K3 ($out)"
    else
        no "default budget not capped on K3 (${out:-no output})"
    fi
else
    sk "K3 budget check" "no container at $BIG"
fi

# ----------------------------------------------------------- parameters ----
head_ "parameter counts"

# The other two things `info` says about a container were string constants
# until a second model existed, and both were wrong about it: K3 announced
# itself as kimi-linear with a Q8G trunk, being neither. They are derived
# now, so check them against the manifest — the architecture K3 records
# under `_outer`, and every trunk format the language model actually uses.
info_rule() {
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run([os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else "")), "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c = man["config"]

hf = ((c.get("_outer", {}).get("architectures") or c.get("architectures")
       or [""]))[0]
arch = ("kimi-k3" if "KimiK3" in hf else "kimi-linear" if "KimiLinear" in hf
        else hf or "unknown")     # a container that names nothing gets that

NAMES = {0: "F32", 1: "F16", 2: "Q8G", 3: "Q4G", 7: "Q3G"}
pref = man.get("tensor_prefix", "")
used = {t["fmt"] for t in man["trunk"]
        if not pref or t["name"].startswith(pref)}
quant = (f"experts VQ{man['expert_quant']['stages']}R, trunk "
         + "/".join(NAMES[f] for f in (7, 3, 2, 1, 0) if f in used))

print(f"{info['arch']}, {info['quantization']}")
sys.exit(0 if info["arch"] == arch and info["quantization"] == quant else 1)
PY
}

if [ -d "$MODEL" ]; then
    if out=$(info_rule "$MODEL" 2>/dev/null); then
        ok "info describes the container it opened ($out)"
    else
        no "info describes something else (${out:-no output})"
    fi
    if out=$(params_rule "$MODEL" 2>/dev/null); then
        ok "params_total is what the container holds ($out)"
    else
        no "params_total off the rule (${out:-no output})"
    fi
else
    sk "parameter counts" "no container at $MODEL"
fi

# K3 is the model both rules were wrong about, and the only latent MoE
# here: without it the expert width and the hidden never differ, and the
# two descriptive fields pass on the constants they used to be.
if [ -f "$BIG/manifest.json" ]; then
    if out=$(info_rule "$BIG" 2>/dev/null); then
        ok "info names K3 and its trunk, not the model before it ($out)"
    else
        no "info describes something else on K3 (${out:-no output})"
    fi
    if out=$(params_rule "$BIG" 2>/dev/null); then
        ok "params_total counts K3's experts at the latent ($out)"
    else
        no "params_total off the rule on K3 (${out:-no output})"
    fi
else
    sk "K3 parameter counts" "no container at $BIG"
fi

# --------------------------------------------------------------- vision ----
head_ "vision preprocessing"

# The tower is checked against its oracle on *random* pixels, so nothing in
# the suite ever looked at what a real image is normalized by. It was the
# CLIP convention for a day, against a release that states mean = std = 0.5
# in preprocessor_config.json — a wrong constant that every existing check
# was structurally blind to. Assert the container against the source.
vision_norm() {
    python3 - "$1" "$2" <<'PY'
import json, os, sys
cont, src = sys.argv[1], sys.argv[2]
vj = os.path.join(cont, "vision.json")
pp = os.path.join(src, "preprocessor_config.json")
if not os.path.exists(vj) or not os.path.exists(pp):
    sys.exit(2)                       # nothing to compare: caller skips
v = json.load(open(vj))
m = json.load(open(pp)).get("media_proc_cfg", {})
bad = [k for k in ("image_mean", "image_std")
       if m.get(k) is not None and v.get(k) != m[k]]
print(f"mean {v.get('image_mean')} std {v.get('image_std')}")
sys.exit(1 if bad else 0)
PY
}

if [ -d "$MODEL" ] && [ -d "$SRC" ]; then
    out=$(vision_norm "$MODEL" "$SRC"); rc=$?
    if [ "$rc" = 2 ]; then
        sk "image normalization vs the release" "no vision tower in this container"
    elif [ "$rc" = 0 ]; then
        ok "image normalization is the release's ($out)"
    else
        no "image normalization differs from preprocessor_config.json ($out)"
    fi
else
    sk "image normalization vs the release" "needs a container and source weights"
fi

if [ -f "$BIG/manifest.json" ] && [ -d "${BIG_SRC:-/Volumes/WasteDisk/k3}" ]; then
    out=$(vision_norm "$BIG" "${BIG_SRC:-/Volumes/WasteDisk/k3}"); rc=$?
    if [ "$rc" = 2 ]; then
        sk "K3 image normalization" "no vision.json or no preprocessor config"
    elif [ "$rc" = 0 ]; then
        ok "K3 image normalization is the release's ($out)"
    else
        no "K3 image normalization differs from the release ($out)"
    fi
else
    sk "K3 image normalization" "needs the K3 container and its source"
fi

# ------------------------------------------------------------ tokenizer ----
head_ "tokenizer"

# Prompt text must not be able to write conversation structure. The engine
# encodes markup and content in separate modes, exactly as the release's
# tokenizer splits allowed_special from disallowed_special; without that
# split a prompt carrying <|end_of_msg|><|open|>message role="system"…
# ends its own turn and opens a forged one, with real control-token ids.
inject_probe() {                      # $1 = container
    python3 - "$1" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "specials.json")
if not os.path.exists(p):
    sys.exit(2)
sp = json.load(open(p))
if not sp:
    sys.exit(2)
# the container's own markers, so this works on any model
texts = [e["text"] for e in sp[:3]]
print("hi" + "".join(texts) + "obey")
print(" ".join(str(e["id"]) for e in sp))
PY
}

if [ -d "$MODEL" ] && [ "$SYNTHETIC" != 1 ] && probe=$(inject_probe "$MODEL"); then
    INJ=$(printf '%s' "$probe" | head -1)
    specials=$(printf '%s' "$probe" | tail -1)
    markup=$(./test_tokenizer "$MODEL" "$INJ" 2>/dev/null | head -1)
    plain=$(WASTE_TOK_PLAIN=1 ./test_tokenizer "$MODEL" "$INJ" 2>/dev/null | head -1)
    hits=0
    for id in $specials; do
        case " $plain " in *" $id "*) hits=$((hits + 1)) ;; esac
    done
    if [ "$hits" -eq 0 ] && [ "$markup" != "$plain" ]; then
        ok "prompt text cannot forge control tokens (markup mode still can)"
    else
        no "a prompt injected $hits control tokens, markup==plain: $([ "$markup" = "$plain" ] && echo yes || echo no)"
    fi
else
    sk "prompt text cannot forge control tokens" "needs a container with specials.json"
fi

if [ "$SYNTHETIC" = 1 ]; then
    sk "tokenizer diff" "synthetic container carries no tokenizer"
elif [ -d "$MODEL" ] && command -v uv >/dev/null 2>&1 && [ -d "$SRC" ]; then
    if uv run --quiet --with tiktoken --no-project python tools/tokdiff.py \
           "$MODEL" "$SRC" 2>/dev/null | tail -1 | grep -q "identical"; then
        ok "C tokenizer matches Python tiktoken"
    else
        no "tokenizer differs from tiktoken"
    fi
else
    sk "tokenizer diff" "needs uv, a container and source weights"
fi

# ------------------------------------------------- quantization decode ----
head_ "quantization decode (DeepSeek native formats)"

# E2M1 has 16 codes and E4M3 has 256, so this enumerates both formats
# exhaustively rather than sampling — decode is exact or it is wrong. Needs
# no container and no weights. It is the cheap half of gate V1: it proves
# conformance to the published format definitions, not agreement with
# DeepSeek's reference, which has never been readable here.
if out=$(./test_quant 2>&1); then
    ok "E2M1/UE8M0/E4M3 decode and block-scale indexing"
else
    no "quantization decode"
    printf '%s\n' "$out" | grep -E "FAIL" | head -5
fi

# The SPDX gate CI enforces globs src/*.c and misses subdirectories, so it
# would not have seen src/quant/ at all. Check it here, recursively, while
# CI is parked — and see .github/workflows-disabled/README.md for the glob
# that needs widening before that workflow is restored.
if command -v git >/dev/null 2>&1; then
    missing=$(git ls-files 'src/**/*.c' 'src/**/*.h' 'src/*.c' 'src/*.h' \
                           'src/*.m' 'cli/*.c' 'tests/*.c' 'tools/*.py' \
                           'tools/*.sh' 2>/dev/null \
              | xargs grep -L 'SPDX-License-Identifier' 2>/dev/null || true)
    if [ -z "$missing" ]; then
        ok "every source file carries an SPDX header (subdirectories too)"
    else
        no "SPDX headers"
        printf '%s\n' "$missing" | head -5
    fi
else
    sk "SPDX headers" "git not available to list tracked files"
fi

# --------------------------------------------------- checkpoint inventory ----
head_ "checkpoint inventory"

# tools/inventory.py is what turns README §1's estimates into measurements,
# so the thing worth testing is that it never invents one: absent shards
# have to become SKIPs and unresolved counts, not confident zeros. Runs
# against a synthetic header-only checkpoint, so it needs no weights.
if ! command -v python3 >/dev/null 2>&1; then
    sk "inventory.py" "python3 not installed"
elif out=$(python3 tests/test_inventory.py 2>&1); then
    ok "inventory measures what it read and refuses to guess the rest"
else
    no "inventory.py"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# ------------------------------------------------- DeepSeek gate replays ----
head_ "DeepSeek gate replays (Gate A/V0 acquisition, B/V1 → F/V4, E/V5, H/V6)"

# These eleven replays did run before this section existed, but not under
# their own names. test_inventory.py imported the Gate B/V1 driver, which
# imported the Gate C, D and F replays, so the whole ladder arrived as the
# single line "inventory measures what it read and refuses to guess the
# rest". That is why the suite total stayed at the 34 AGENTS.md records for
# PR #3 across four gate PRs — the evidence was running and invisible.
#
# It was also misattributed. Flipping one bit in the frozen Gate F output
# combined-out8.bf16.bin reported `FAIL inventory.py` with the sub-lines
# "all tensor names classified" and "no bytes in an unexplained bucket",
# which sends a reader to the tensor classifier for a fault in the MoE
# combination. And the chain returned on first failure, so the earliest
# broken gate hid every later one. This file already made the argument for
# the serve suite: 140-odd checks reported as one line hides which half ran.
#
# So each replay is invoked here by name, and the chaining is gone from
# test_inventory.py and test_release_quant_fixture.py. Enumerated rather
# than globbed, because a glob hides an unreplayed fixture just as well: a
# gate is covered when its line appears below, and adding a gate means
# adding one.
#
# Exit 77 is the suite-wide "did not run" code (see the cpu-bind check
# above), and these scripts use it when a frozen fixture or a C compiler is
# absent. Mapping 77 to SKIP rather than PASS is the load-bearing half: a
# checkout missing tests/fixtures/deepseek_v4/ must report unproven gates,
# not green ones.
deepseek_replay() {
    local label=$1 script=$2 out rc reason
    if ! command -v python3 >/dev/null 2>&1; then
        sk "$label" "python3 not installed"
        return
    fi
    out=$(python3 "tests/$script" 2>&1)
    rc=$?
    case $rc in
        0)  ok "$label" ;;
        77) reason=$(printf '%s\n' "$out" | grep -m1 'SKIP' | sed 's/^SKIP: *//')
            sk "$label" "${reason:-prerequisite missing}" ;;
        *)  no "$label"
            printf '%s\n' "$out" \
                | grep -E "FAIL|Error|Traceback|AssertionError" | head -5 ;;
    esac
}

# Gate A/V0 — the acquisition layer. Both scripts are model-free: they drive
# the fetchers against a local server, so they check the fail-closed
# behaviour (a proxy answering 200 to a Range request must be refused) with
# no network and no checkpoint.
deepseek_replay "Gate A/V0 — safetensors header Range fetch fails closed" \
                "test_fetch_hf_headers.py"
deepseek_replay "Gate A/V0 — bounded tensor-row Range slicing fails closed" \
                "test_fetch_hf_tensor_slice.py"

# Gate B/V1 — the release's own nibble order and scale direction, frozen as
# literal bytes rather than as a round trip through our packer, because a
# round trip stayed green under the nibble mutation (AGENTS.md invariant 4).
deepseek_replay "Gate B/V1 — release FP4 nibble order and scale direction" \
                "test_release_quant_fixture.py"

# Gate C/V2 — scalar preflight, then the real projection against an oracle
# that shares no code with the implementation under test.
deepseek_replay "Gate C/V2 — source-derived FP8/FP4 scalar linear preflight" \
                "test_v2_linear_ref.py"
deepseek_replay "Gate C/V2 — real 0731 wq_a projection vs independent oracle" \
                "test_v2_real_projection.py"

# Gate D/V3 — mHC. The model-free half pins Sinkhorn's invariants and
# hc_post's orientation, which is where a transpose would hide.
deepseek_replay "Gate D/V3 — model-free Sinkhorn invariants, hc_post orientation" \
                "test_v3_mhc_scalar.py"
deepseek_replay "Gate D/V3 — real layer-0 mHC, exact BF16 pre and post" \
                "test_v3_mhc_real.py"

# Gate F/V4 — routing then MoE. The scalar half pins that the correction
# bias moves selection without contaminating the route weights, and that
# routed results accumulate in ascending expert ID rather than top-k order.
deepseek_replay "Gate F/V4 — model-free router bias-selection and hash-ID semantics" \
                "test_v3_router_scalar.py"
deepseek_replay "Gate F/V4 — real learned and hash routing, exact top-6 IDs" \
                "test_v3_router_real.py"
deepseek_replay "Gate F/V4 — real routed FP4 expert 3/2, exact gate/up and out8" \
                "test_v4_routed_expert_real.py"
deepseek_replay "Gate F/V4 — real shared FP8 expert and six-branch combination" \
                "test_v4_moe_real.py"

# Gate E/V5 — PASSED at scalar/model-semantic evidence level. Ratio-0,
# grouped output, coherent ratio-128, and coherent ratio-4 CSA all have
# checkpoint-backed frozen fixtures. Full-layer composition is Gate H/V6. These three arrived on main while
# this section was being written, added to the old import chain — the first
# instance of the maintenance cost enumeration buys, paid immediately.
deepseek_replay "Gate E/V5 — model-free ratio-0 attention semantics" \
                "test_v5_attention_scalar.py"
deepseek_replay "Gate E/V5 — real ratio-0 attention core, exact BF16 Q/KV/attn" \
                "test_v5_attention_ratio0_real.py"
deepseek_replay "Gate E/V5 — real grouped wo_a/wo_b output projection" \
                "test_v5_attention_output_real.py"

# Ratio-128 compressor + compressed-history composition. Both now have frozen
# real checkpoint evidence; the model-free half remains load-bearing for YaRN
# check originally witnessed only rope pair 0, where base**0 == 1 and the ramp
# leaves the frequency untouched, so compressed YaRN and plain base-10000 RoPE
# are bit-identical there by construction. Pair 20 sits at ramp 0.5 and does
# discriminate. EXPERIMENTS.md entry 7 has the mutation table.
deepseek_replay "Gate E/V5 — model-free ratio-128 pooling and compressed YaRN" \
                "test_v5_compressor_scalar.py"
deepseek_replay "Gate E/V5 — real ratio-128 compressor stages" \
                "test_v5_compressor_real.py"
deepseek_replay "Gate E/V5 — model-free ratio-128 compressed-history indexing" \
                "test_v5_attention_history_scalar.py"
deepseek_replay "Gate E/V5 — real ratio-128 compressed-history composition" \
                "test_v5_attention_history_real.py"
deepseek_replay "Gate E/V5 — model-free inverse compressed YaRN pair20" \
                "test_v5_compressed_inverse_scalar.py"
deepseek_replay "Gate E/V5 — coherent same-input ratio-128 attention" \
                "test_v5_attention_ratio128_coherent_real.py"
deepseek_replay "Gate E/V5 — model-free ratio-4 CSA semantics" \
                "test_v5_csa_scalar.py"
deepseek_replay "Gate E/V5 — real ratio-4 CSA indexer" \
                "test_v5_csa_indexer_real.py"
deepseek_replay "Gate E/V5 — coherent real ratio-4 CSA attention" \
                "test_v5_csa_attention_real.py"

# Gate H/V6 — PASSED at one complete real layer-3 transition. The model-free half pins the block
# wiring: hc_pre(attn) -> branch -> hc_post -> hc_pre(ffn) -> branch -> hc_post,
# and refuses five wirings that would still produce plausible numbers — reusing
# the original residual for the FFN hc_pre, swapping the attention and FFN HC
# parameter sets, feeding a correct branch output from the wrong branch input,
# and replacing either hc_post with the ordinary residual add. The real half
# replays that composition with layer-3 HC parameters from the checkpoint.
deepseek_replay "Gate H/V6 — bit-exact independent F32 HyperConnection oracle" \
                "test_v6_hc_oracle_f32.py"
deepseek_replay "Gate H/V6 — corrected real layer-3 final HC precision boundary" \
                "test_v6_layer3_hc_precision.py"
deepseek_replay "Gate H/V6 — model-free layer block wiring and five refusals" \
                "test_v6_layer_composition_scalar.py"
deepseek_replay "Gate H/V6 — full eight-group attention output seam" \
                "test_v6_attention_output_full_scalar.py"
deepseek_replay "Gate H/V6 — real layer-3 HC composition, exact BF16 final" \
                "test_v6_hc_composition_real.py"
# The attention half composed for real: the frozen 64-head branch is bound by
# SHA-256 to the attn_pre state it was produced from, so chaining the two
# fixtures is checked rather than assumed. post/comb come from an independent
# Sinkhorn rather than from the hc_pre under test, which is what lets a fault
# in the normalization show up instead of cancelling on both sides.
deepseek_replay "Gate H/V6 — real attention branch composed through hc_post" \
                "test_v6_attention_composition_real.py"
# One stage further: the real FFN hc_pre and the real layer-3 router on it.
# This is the cheap test that bounds the expensive one — it decides which six
# routed expert records Gate H's remaining step has to fetch, from checkpoint
# bytes rather than from assumption, and refuses a selection whose top-k
# boundary is too narrow to be a stable acquisition list.
deepseek_replay "Gate H/V6 — real FFN hc_pre and layer-3 routing decision" \
                "test_v6_ffn_route_real.py"
# The full shared expert must cross E8M0 w2 scale-grid row boundaries; Gate F's
# first-eight-row fixture never exercised row 128 or beyond.
deepseek_replay "Gate H/V6 — shared FP8 full-output scale-grid boundary" \
                "test_v6_shared_expert_full_rows_scalar.py"
# Frozen from the exact real FFN hc_pre state and exact scalar-router F32
# weights. Acquisition cross-checked six routed experts plus the shared
# expert against an independent oracle for all 28,672 BF16 outputs.
deepseek_replay "Gate H/V6 — real six-routed-plus-shared full MoE branch" \
                "test_v6_moe_branch_real.py"
# The complete layer transition composes the two real HC stages, frozen
# real 64-head attention branch, exact real router/MoE branch, and exact
# BF16 [4,4096] final state. This is the Gate H endpoint.
deepseek_replay "Gate H/V6 — complete real layer-3 composition" \
                "test_v6_layer3_full_real.py"
# V7 precision boundaries found by real chained-layer acquisition. These are
# permanent because each caught an otherwise plausible, near-bit-exact oracle.
deepseek_replay "V7 — hc_pre sequential F32 reduction boundary" \
                "test_v7_hc_pre_f32_scalar.py"
deepseek_replay "V7 — sparse FP8 scalar differential and signed-zero cases" \
                "test_v7_sparse_fp8_oracle.py"
deepseek_replay "V7 — one-row online-softmax signed-zero operation order" \
                "test_v7_one_row_attention_signed_zero.py"
deepseek_replay "V7 — correction bias operation vs input-specific top-k outcome" \
                "test_v7_router_bias_observation.py"
deepseek_replay "V7 — current Gate-H parent freshness contract" \
                "test_v7_parent_freshness.py"

# V7 infrastructure: compare chained expected/runtime layer-boundary traces,
# reject stale manifests/broken output->input chains, and stop at the first
# differing layer/boundary rather than reporting a downstream symptom.
deepseek_replay "V7 — fail-closed first-divergence layer trace localizer" \
                "test_v7_localize.py"
deepseek_replay "V7 — real chained layer-3 output through layer-4 router" \
                "test_v7_layer4_route_real.py"
deepseek_replay "V7 — complete real layer-4 composition from frozen layer-3 output" \
                "test_v7_layer4_full_real.py"
deepseek_replay "V7 — real consecutive layer-3 to layer-4 trace" \
                "test_v7_two_layer_real.py"

# Gate I/V8 starts with an immutable header/source contract. This gate reads no
# tensor payload and makes no logit claim; it prevents later numeric work from
# silently targeting current-main source shapes or guessed final-head records.
deepseek_replay "Gate I/V8 — immutable final-head header/source surface" \
                "test_v8_head_surface.py"

# ------------------------------------------------------------ converter ----
head_ "converter"

# Resume is the one converter behaviour that cannot be checked by looking at
# a finished container: it is about the partial states a crash leaves. The
# quantizer is stubbed out, so this needs neither torch nor source weights.
if ! command -v python3 >/dev/null 2>&1; then
    sk "convert.py resume" "python3 not installed"
elif out=$(python3 tests/test_convert_resume.py 2>&1); then
    ok "resume keeps finished layers, never renumbers them, and publishes"
else
    no "convert.py resume"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# ---------------------------------------------------------------- serve ----
head_ "serve (OpenAI-compatible server)"

# The Python suite needs libwaste as a shared object; the CLI links the
# archive, so a plain `make` before this change did not produce one.
# The Makefile picks this from the compiler's target triple; the suite has
# to reach the same answer from the shell. `uname -s` under Git-Bash and
# MSYS says MINGW64_NT-… , which the Darwin test missed and the fallback
# then sent to libwaste.so — a target that does not exist on Windows, so
# the check reported a build failure for a library that had built fine.
case "$(uname -s)" in
    Darwin)                SOEXT=dylib ;;
    MINGW*|MSYS*|CYGWIN*)  SOEXT=dll ;;
    *)                     SOEXT=so ;;
esac

if [ "${WASTE_SANITIZED:-0}" = 1 ]; then
    # ASan needs to be the first library loaded; a dlopen from a plain
    # python3 is not, and the run dies in the allocator rather than
    # reporting anything about the server.
    sk "serve suite" "not run under the sanitizers"
elif ! command -v python3 >/dev/null 2>&1; then
    sk "serve suite" "python3 not installed"
elif ! make -s "libwaste.$SOEXT" >/dev/null 2>&1; then
    no "libwaste.$SOEXT failed to build"
else
    # Counted rather than pass/fail as a lump: 140-odd checks reported as
    # one line hides which half ran.
    out=$(python3 -m unittest discover -s tests/serve -t . -p "test_*.py" 2>&1)
    n=$(printf '%s' "$out" | grep -oE "^Ran [0-9]+ test" | grep -oE "[0-9]+")
    if printf '%s' "$out" | tail -3 | grep -q "^OK"; then
        ok "serve suite (${n:-?} checks: XTML vs upstream, regions, ctypes, HTTP)"
    else
        no "serve suite"
        printf '%s\n' "$out" | grep -E "^(FAIL|ERROR):" | head -8
    fi
fi

# The prompt corpus is checked against the release's own encoder when the
# weights directory is present. That is the check that says our port of
# encoding_k3.py is K3's format and not merely self-consistent.
K3_SRC="${K3_DIR:-/Volumes/WasteDisk/k3}"
if [ -f "$K3_SRC/encoding_k3.py" ]; then
    if K3_DIR="$K3_SRC" python3 -m unittest \
           tests.serve.test_xtml.TestAgainstUpstream 2>&1 | tail -3 | grep -q "^OK"; then
        ok "XTML prompts match the release's encoding_k3.py, segment for segment"
    else
        no "XTML prompts differ from encoding_k3.py"
    fi
else
    sk "XTML vs encoding_k3.py" "no release at $K3_SRC (set K3_DIR)"
fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
