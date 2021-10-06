"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for KVQA model
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
from .memnet import MemNN

class BERTMemNetFusion(nn.Module):
    def __init__(self, config, memnet_size):
        super(BERTMemNetFusion, self).__init__()
        self.dense = nn.Linear(memnet_size+config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, fused_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(fused_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# TODO: move to config
MEMNET_HIDDEN_DIM = 512

class UniterForKnowledgeQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for KVQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.memnet = MemNN(vocab_size=config.vocab_size, embd_size=32, 
                            ans_size=MEMNET_HIDDEN_DIM, max_story_len=16)
        if torch.cuda.is_available():
            self.memnet.cuda()
        #self.memnet_optimizer = torch.optim.Adam(self.memnet.parameters())
        #self.memnet_loss_fn = nn.NLLLoss()
        
        self.pooler = BERTMemNetFusion(config, MEMNET_HIDDEN_DIM)
        
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)
        
        print('Outsize =', num_answer)
        
        
    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        facts = batch['facts']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        memnet_output = self.memnet(facts, input_ids)
        
        uniter_hidden = sequence_output[:, 0]
        
        fused = torch.cat([memnet_output, uniter_hidden], dim=1)

        pooled_output = self.pooler(fused)
        answer_scores = self.vqa_output(pooled_output)
        
        #print(self.memnet.A[1].weight.data)
             
        if compute_loss:
            targets = batch['targets']
            kvqa_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return kvqa_loss
        else:
            return answer_scores
