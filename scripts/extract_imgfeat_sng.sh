# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

IMG_DIR=$1
OUT_DIR=$2

set -e

echo "extracting image features..."

singularity run --nv --bind $IMG_DIR:/img,$OUT_DIR:/output --pwd /src docker://chenrocks/butd-caffe:nlvr2 bash -c "python tools/generate_npz.py --gpu 0"

echo "done"
