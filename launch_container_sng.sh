# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4

export SINGULARITY_CACHEDIR="/fastdata/${USER}/singularity-cachedir"
export SINGULARITY_LOCALCACHEDIR="${SINGULARITY_CACHEDIR}"
mkdir -p ${SINGULARITY_CACHEDIR}

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi


singularity run --nv \
   --bind $(pwd):/src,$OUTPUT:/storage,$PRETRAIN_DIR:/pretrain:ro,$TXT_DB:/txt:ro,$IMG_DIR:/img:ro --pwd /src docker://chenrocks/uniter
