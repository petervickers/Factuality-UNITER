# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OUT_DIR=$1
ANN_DIR=$2

set -e

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
if [ ! -d $ANN_DIR ]; then
    mkdir -p $ANN_DIR
fi

for SPLIT in 'train_balanced_questions' 'val_balanced_questions' 'testdev_balanced_questions'; do
    if [ ! -f $ANN_DIR/$SPLIT.json ]; then
        echo "downloading ${SPLIT} annotations..."
        wget $URL/$SPLIT.json -O $ANN_DIR/$SPLIT.json
    fi

    echo "preprocessing ${SPLIT} annotations..."
    docker run --ipc=host --rm -it \
        --mount src=$(pwd),dst=/src,type=bind \
        --mount src=$OUT_DIR,dst=/txt_db,type=bind \
        --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
        -w /src chenrocks/uniter \
        python prepro_gqa.py --annotation /ann/$SPLIT.json \
                         --output /txt_db/nlvr2_${SPLIT}.db
done

echo "done"
