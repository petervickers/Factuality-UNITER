# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2
FEAT_DIR=$3
USE_FACTS=$4
FOLD=$5

set -e

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi


echo $OUT_DIR
ls $OUT_DIR
COUNTER=1

echo "With use facts status ${USE_FACTS}"
for SPLIT in 'train_questions' 'val_questions' 'test_questions'; do

    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$FEAT_DIR,dst=/img_feat,type=bind,readonly \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro_kvqa_kb.py --annotation /ann/dataset.json \
                         --output /txt_db/kvqa_${SPLIT}.db \
                         --split ${COUNTER} --use_facts ${USE_FACTS} \
                         --fold ${FOLD}
    let "COUNTER++"
done

echo "done"
