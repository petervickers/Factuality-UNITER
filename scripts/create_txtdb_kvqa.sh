# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2
FEAT_DIR=$3

set -e

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

COUNTER=1
for SPLIT in 'train_questions' 'val_questions' 'test_questions'; do

    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$FEAT_DIR,dst=/img_feat,type=bind,readonly \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro_kvqa.py --annotation_path /ann/ \
                         --output /txt_db/kvqa_${SPLIT}.db \
                         --split ${COUNTER}
    let "COUNTER++"
done

echo "done"
