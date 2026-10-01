#!/bin/bash
# Download the LAMP data archives from the GitHub release, verify them, and extract them into data/.
set -e
RELEASE_URL=https://github.com/jalal-gilgiti/LAMP/releases/download/v1.0
ARCHIVES="lamp_dynamic_benchmark lamp_fold_stats lamp_predictions"
cd "$(dirname "$0")/.."
mkdir -p downloads
for a in $ARCHIVES; do
    [ -f downloads/$a.tar.zst ] || wget -O downloads/$a.tar.zst "$RELEASE_URL/$a.tar.zst"
done
(cd downloads && sha256sum -c ../SHA256SUMS)
for a in $ARCHIVES; do
    echo "extracting $a"
    zstd -dc downloads/$a.tar.zst | tar -x
done
