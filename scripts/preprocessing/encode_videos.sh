#!/usr/bin/env bash

set -e

ROOT="${KILIFISH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${KILIFISH_DATA_ROOT:-$ROOT/data}"
RAW_DATA_ROOT="${KILIFISH_RAW_DATA_ROOT:-$DATA_ROOT/raw}"
PROCESSED_DATA_ROOT="${KILIFISH_PROCESSED_DATA_ROOT:-$DATA_ROOT/processed}"
SRC="${KILIFISH_V2_ROOT:-$RAW_DATA_ROOT/killifish-v2}"
DST="${KILIFISH_V2_ENCODED_ROOT:-$PROCESSED_DATA_ROOT/killifish-v2-encoded}"

JOBS="${JOBS:-4}"            # how many videos to encode in parallel
# FFMPEG="ffmpeg"             # uses conda ffmpeg in this env
FFMPEG="${FFMPEG:-/usr/bin/ffmpeg}"

convert_one() {
    infile="$1"
    rel="${infile#$SRC/}"
    outdir="$DST/$(dirname "$rel")"
    mkdir -p "$outdir"

    base="$(basename "$rel")"
    name="${base%.*}"
    outfile="$outdir/${name}.mp4"

    echo "Converting:"
    echo "  in :  $infile"
    echo "  out: $outfile"

    "$FFMPEG" -y \
    -i "$infile" \
    -c:v libx264 \
    -preset veryfast \
    -crf 22 \
    -vf scale=-1:540 \
    -an \
    "$outfile"
}

export SRC DST FFMPEG
export -f convert_one

find "$SRC" -type f \( -iname "*.mov" -o -iname "*.mp4" -o -iname "*.m4v" \) -print0 \
| xargs -0 -n1 -P"$JOBS" bash -c 'convert_one "$0"'
