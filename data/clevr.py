"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

GQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index


class GqaDataset(DetectFeatTxtTokDataset):
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
        
        target = self.ans2label[example['target']]       
        
        target_torch = torch.tensor(target, dtype=torch.long)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target_torch


def gqa_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    targets = torch.stack(targets)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class GqaEvalDataset(GqaDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        # Handle new targets unseen in trains
        target = self.ans2label.get(example['target'], self.ans2label['UNK'])    
        
        target_torch = torch.tensor(target, dtype=torch.long)


        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target_torch


def gqa_eval_collate(inputs):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
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
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch
