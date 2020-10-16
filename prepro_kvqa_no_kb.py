"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess KVQA annotations into LMDB
"""
import argparse
import json
import os
from os.path import exists

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


def process_kvqa(jsonl, db, tokenizer, missing=None, split=1):
    id2len = {}
    txt2img = {}  # not sure if useful
    ans2idx = {}
    idx2type = {}
    idx2question = {}

    json_loaded = json.load(jsonl)   
    total = 0
    processed = 0
    all_qs = set()
    for data_idx, data in tqdm(json_loaded.items(), desc='processing KVQA'):
        img_id = data['imgPath'].split('/')[-1]
        img_fname = (f'nlvr2_{img_id[:-4]}.npz')
        img_path = os.path.join('/img_feat', img_fname)
        if not os.path.isfile(img_path):
            #print(f'Features not generated for {img_path}, skipping...')
            continue
        if len(data['Type of Question']) != len(data['Questions']):
            #print(f'WARNING: question type labels broken at index {data_idx}')
            continue
        for q_idx in range(len(data['Questions'])):
            total += 1
            
            # handle KVQA split format
            if int(data['split'][0]) != int(split):
                continue 
            processed += 1

            example = {}        
        
            example['identifier'] = data_idx+f'_{q_idx}'
            example['sentence'] = data['Questions'][q_idx]
            all_qs.add(example['sentence'])
            example['target'] = data['Answers'][q_idx]
            example['type'] = data['Type of Question'][q_idx]
            example['Qids'] = data['Qids']
        
            if example['target'] not in ans2idx:
                #print(f'Adding new answer {example["target"]} at position {len(ans2idx)}')
                ans2idx[example['target']] = len(ans2idx)
            
        
            id_ = example['identifier']
            input_ids = tokenizer(example['sentence'])
            txt2img[id_] = img_fname
            id2len[id_] = len(input_ids)
            example['input_ids'] = input_ids
            example['img_fname'] = img_fname

            idx2type[id_] = example['type']
            idx2question[id_] = example['sentence']
            if example is not None:
                db[id_] = example
     
    print(f'For split {split} processed {processed/total*100}%')        
    print(f'Number of questions: {len(all_qs)}, list of all questions:')
    ans2idx['UNK'] = len(ans2idx)
    idx2ans = {k:v for v,k in ans2idx.items()}
    return id2len, txt2img, idx2ans, idx2type, idx2question


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
            id2lens, txt2img, idx2ans, idx2type, idx2question = process_kvqa(ann, \
                                                                             db, \
                                                                             tokenizer, \
                                                                             missing_imgs, \
                                                                             split=opts.split)
            
    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/idx2ans.json', 'w') as f:
        json.dump(idx2ans, f)
    with open(f'{opts.output}/idx2type.json', 'w') as f:
        json.dump(idx2type, f)
    with open(f'{opts.output}/idx2question.json', 'w') as f:
        json.dump(idx2question, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to use')
    parser.add_argument('--split', default='1',
                        help='which split to process')
    args = parser.parse_args()
    main(args)
