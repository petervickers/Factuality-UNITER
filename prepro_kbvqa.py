"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess GQA annotations into LMDB
"""
import argparse
import json
import os
from os.path import exists
import string
from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split

from data.data import open_lmdb


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_kbvqa(jsonl, db, tokenizer, split_seed, split, missing=None, ans2idx={}):
    id2len = {}
    txt2img = {}  # not sure if useful

    read_only_ans = False
    if ans2idx:
	# make readonly
        read_only_ans = True
   
    accept = set()
    total = 0
    skipped = 0

    json_loaded = json.load(jsonl)   
    train_data, val_data = train_test_split(json_loaded, test_size=0.2, random_state=int(split_seed))

    if split == 'train':
        json_loaded = train_data
    elif split == 'val':
        json_loaded = val_data
    else:
        raise ValueError('split is not defined')

    for idx, data in tqdm(enumerate(json_loaded), desc='processing KBVQA'):
        example = data
        
        img_id = 'nlvr2_COCO_val2014_'+example['img_id'].zfill(12)
        total += 1
            
        img_path = os.path.join('/img_feat', img_id+'.npz')
        if not os.path.isfile(img_path):
            print(f'Features not generated for {img_path}, skipping...')
            skipped += 1
            continue
        
        # make fvqa similar to nlvr structure
        example['identifier'] = idx
        example['sentence'] = example['nlq']
        example['target'] = example['nla'].lower().translate(str.maketrans('', '', string.punctuation))
        
        if not read_only_ans:
            if example['target'] not in ans2idx:
                print(f'Adding new answer {example["target"]} at position {len(ans2idx)}')
                ans2idx[example['target']] = len(ans2idx)
            
            
        
        id_ = str(example['nlq_idx'])
        img_id = str(example['img_id'])
        img_fname = (f'nlvr2_COCO_val2014_{example["img_id"].zfill(12)}.npz')
        input_ids = tokenizer(example['sentence'])
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        
        
        db[id_] = example
    
    ans2idx['UNK'] = len(ans2idx)
    idx2ans = {k:v for v,k in ans2idx.items()}
    print('Skipped', skipped*100/total)
    return id2len, txt2img, idx2ans


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    meta['bert'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)
    
    if exists(f'{opts.output}/idx2ans.json'):
        with open(f'{opts.output}/idx2ans.json', 'w') as f:
            idx2ans = json.load(f)
            ans2idx = {v:k for k,v in idx2ans.items()}
    else:
            ans2idx = {}

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation) as ann:
            if opts.missing_imgs is not None:
                missing_imgs = set(json.load(open(opts.missing_imgs)))
            else:
                missing_imgs = None
            id2lens, txt2img, idx2ans = process_kbvqa(ann, db, tokenizer, opts.split_seed, opts.split, missing_imgs, ans2idx)
            
    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/idx2ans.json', 'w') as f:
        json.dump(idx2ans, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--split_seed', required=True,
                    help='split seed')
    parser.add_argument('--split', required=True,
                    help='split (train or val)')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
