# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2
SPLIT_SEED=$3
FEAT_DIR=$4

set -e

echo $SPLIT_SEED
echo $OUT_DIR
ls $OUT_DIR

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

for SPLIT in 'train' 'val'; do
    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        --mount src=$FEAT_DIR,dst=/img_feat,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro_kbvqa.py --annotation /ann/nlqs_dump.json \
                         --output /txt_db/kbvqa_split.db \
                         --split_seed ${SPLIT_SEED} --split ${SPLIT}
done

echo "done"
