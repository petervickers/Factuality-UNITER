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


def process_fvqa(jsonl, db, tokenizer, split_file, missing=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    ans2idx = {}
    accept = set()
    total = 0
    skipped = 0
    
    
    print(split_file)
    with open(split_file, 'r') as in_file:
        split_lines = in_file.readlines()
        for img_name in split_lines:
            accept.add('nlvr2_'+img_name.split('.')[0])
        
    
    json_loaded = json.load(jsonl)   
    for idx, data in tqdm(json_loaded.items(), desc='processing FVQA'):
        example = data
        
        img_id = 'nlvr2_'+example['img_file'].split('.')[0]
        if img_id not in accept:
            continue
        total += 1
            
        img_path = os.path.join('/img_feat', img_id+'.npz')
        if not os.path.isfile(img_path):
            print(f'Features not generated for {img_path}, skipping...')
            skipped += 1
            continue
        
        # make fvqa similar to nlvr structure
        example['identifier'] = idx
        example['sentence'] = example['question']
        example['target'] = example['entity']
        
        if example['target'] not in ans2idx:
            print(f'Adding new answer {example["target"]} at position {len(ans2idx)}')
            ans2idx[example['target']] = len(ans2idx)
            
        
        id_ = example['identifier']
        img_id = example['img_file'].split('.')[0]
        img_fname = (f'nlvr2_{img_id}.npz')
        input_ids = tokenizer(example['sentence'])
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        
        
        db[id_] = example
        
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

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.annotation) as ann:
            if opts.missing_imgs is not None:
                missing_imgs = set(json.load(open(opts.missing_imgs)))
            else:
                missing_imgs = None
            id2lens, txt2img, idx2ans = process_fvqa(ann, db, tokenizer, opts.split_file, missing_imgs)
            
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
    parser.add_argument('--split_file', required=True,
                    help='split specifier file')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
