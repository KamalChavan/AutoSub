#!/usr/bin/env bash
# Cross-compile a tiny (~4-6 MB) static audio-only Windows x64 ffmpeg.exe for AutoSub.
#
# AutoSub only uses FFmpeg for:
#   1. Demuxing video/audio containers and decoding audio to 16 kHz mono f32le PCM (-vn)
#   2. Optional `silencedetect` audio filter
#   3. Encoding compact VAD chunks to 48 kbps mono MP3 (libmp3lame)
#
# By stripping all video encoders/decoders/filters and GPU SDKs, ffmpeg.exe shrinks from
# ~106 MB (gyan.dev essentials) down to ~4-6 MB (~1.8 MB inside the LZMA2 installer).
#
# Prerequisites (Ubuntu / Debian / WSL):
#   sudo apt-get update && sudo apt-get install -y \
#     gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 nasm make pkg-config curl tar xz-utils
#
# Usage:
#   bash packaging/windows/build_minimal_ffmpeg.sh [output_path]
# Default output:
#   build/cache/ffmpeg-minimal.exe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_PATH="${1:-${PROJECT_ROOT}/build/cache/ffmpeg-minimal.exe}"

FFMPEG_VERSION="${FFMPEG_VERSION:-7.1.1}"
LAME_VERSION="${LAME_VERSION:-3.100}"
HOST_TRIPLE="x86_64-w64-mingw32"
JOBS="$(nproc 2>/dev/null || echo 4)"

for tool in "${HOST_TRIPLE}-gcc" "${HOST_TRIPLE}-strip" nasm make curl tar; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[ERROR] Missing required tool: $tool" >&2
    echo "Install on Ubuntu/WSL: sudo apt-get install -y gcc-mingw-w64-x86-64 nasm make pkg-config curl tar xz-utils" >&2
    exit 1
  fi
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/autosub_ffmpeg_min.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
PREFIX="${WORK_DIR}/prefix"
mkdir -p "$PREFIX" "$(dirname "$OUT_PATH")"

echo "[1/3] Building static libmp3lame ${LAME_VERSION} (${HOST_TRIPLE})..."
curl -fsSL "https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz" -o "${WORK_DIR}/lame.tar.gz"
tar -xzf "${WORK_DIR}/lame.tar.gz" -C "$WORK_DIR"
(
  cd "${WORK_DIR}/lame-${LAME_VERSION}"
  ./configure \
    --host="${HOST_TRIPLE}" \
    --prefix="${PREFIX}" \
    --disable-shared \
    --enable-static \
    --disable-frontend \
    --disable-gtktest \
    --disable-decoder \
    CFLAGS="-O2"
  make -j"${JOBS}"
  make install
)

echo "[2/3] Configuring & building audio-only FFmpeg ${FFMPEG_VERSION} (${HOST_TRIPLE})..."
curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o "${WORK_DIR}/ffmpeg.tar.xz"
tar -xJf "${WORK_DIR}/ffmpeg.tar.xz" -C "$WORK_DIR"
(
  cd "${WORK_DIR}/ffmpeg-${FFMPEG_VERSION}"
  ./configure \
    --arch=x86_64 \
    --target-os=mingw32 \
    --cross-prefix="${HOST_TRIPLE}-" \
    --pkg-config=pkg-config \
    --prefix="${PREFIX}" \
    --extra-cflags="-I${PREFIX}/include -Os" \
    --extra-ldflags="-L${PREFIX}/lib -static -static-libgcc -static-libstdc++" \
    --enable-static \
    --disable-shared \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --disable-ffprobe \
    --disable-network \
    --disable-autodetect \
    --disable-everything \
    --enable-small \
    --enable-libmp3lame \
    --enable-protocol=file,pipe \
    --enable-demuxer=mov,matroska,mpegts,mp3,wav,ogg,flac,aac,flv,asf,avi,aiff,m4v,f32le,s16le \
    --enable-parser=aac,aac_latm,mpegaudio,opus,vorbis,flac,ac3 \
    --enable-decoder=aac,aac_latm,mp3,mp3float,mp2,mp2float,opus,vorbis,flac,alac,ac3,eac3,wmav1,wmav2,wmapro,amrnb,amrwb,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le,pcm_u8,pcm_alaw,pcm_mulaw,adpcm_ima_wav,adpcm_ms \
    --enable-encoder=libmp3lame,aac,pcm_f32le,pcm_s16le \
    --enable-muxer=mp3,adts,ipod,wav,f32le,null \
    --enable-filter=aresample,aformat,anull,silencedetect,volume
  make -j"${JOBS}" ffmpeg.exe
  "${HOST_TRIPLE}-strip" --strip-all ffmpeg.exe
  cp ffmpeg.exe "${OUT_PATH}"
)

SIZE_HUMAN="$(du -h "${OUT_PATH}" | cut -f1)"
echo "[3/3] Built audio-only Windows ffmpeg.exe: ${OUT_PATH} (${SIZE_HUMAN})"
