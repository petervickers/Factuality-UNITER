"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

KVQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index

max_story_len = 16

class KvqaDataset(DetectFeatTxtTokDataset):
    def __init__(self, ans2label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = len(ans2label)
        self.ans2label = ans2label

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        
        facts = example['facts']
        
        target = self.ans2label[example['target']]       
        
        target_torch = torch.tensor(target, dtype=torch.long)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, facts, img_feat, img_pos_feat, attn_masks, target_torch



def vectorize(facts, story_len, s_sent_len):
    ret_facts = []
    for story in facts:
        tmp_story = story
        story = []
        for sent in tmp_story:
            sent += [0] * (s_sent_len - len(sent))
            story.append(sent)
        while len(story) < story_len:
            story.append([0] * s_sent_len)
        story = story[:story_len] # use recent episodes in reverse order

        ret_facts.append(story)
    return ret_facts
    
def kvqa_collate(inputs):
    (input_ids, facts, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))


    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    story = facts
    story_len = min(max_story_len, max([len(s) for s in story]))
    s_sent_len = max([len(sent) for s in story for sent in s])
    
    vec_facts = vectorize(facts, story_len, s_sent_len)
        
    vec_facts = torch.LongTensor(vec_facts)

    targets = torch.stack(targets)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'facts': vec_facts,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class KvqaEvalDataset(KvqaDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        
        
        facts = example['facts']
        
        # Handle new targets unseen in trains
        target = self.ans2label.get(example['target'], self.ans2label['UNK'])    
        
        target_torch = torch.tensor(target, dtype=torch.long)


        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, facts, img_feat, img_pos_feat, attn_masks, target_torch


def kvqa_eval_collate(inputs):
    (qids, input_ids, facts, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    
    story = facts
    story_len = min(max_story_len, max([len(s) for s in story]))
    s_sent_len = max([len(sent) for s in story for sent in s])
    
    vec_facts = vectorize(facts, story_len, s_sent_len)
        
    vec_facts = torch.LongTensor(vec_facts)
    
    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'facts': vec_facts,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch
