# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2
FEAT_DIR=$3

set -e

echo $OUT_DIR
echo $ANN_DIR
echo $FEAT_DIR

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
        python prepro_vqa9a.py --annotation0 /ann/v2_mscoco_${SPLIT}2014_annotations.json \
                         --annotation1 /ann/v2_OpenEnded_mscoco_${SPLIT}2014_questions.json \
                         --output /txt_db/vqa_${SPLIT}.db \
                         --split ${SPLIT}
done

echo "done"
