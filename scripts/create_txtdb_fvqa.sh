# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2
SPLIT_DIR=$3
FEAT_DIR=$4

set -e

echo $SPLIT_DIR
echo $OUT_DIR
ls $OUT_DIR

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

for SPLIT in 'train_list_0' 'test_list_0'; do
    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        --mount src=$SPLIT_DIR,dst=/split,type=bind,readonly \
        --mount src=$FEAT_DIR,dst=/img_feat,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro_fvqa.py --annotation /ann/all_qs_dict_release.json \
                         --output /txt_db/fvqa_${SPLIT}.db \
                         --split_file /split/${SPLIT}.txt
done

echo "done"
