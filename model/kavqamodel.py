from collections import defaultdict

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
from .kgtransformer import QuestionGraphTransformer

from .cti_model.fcnet import FCNet
from .cti_model.tcnet import TCNet
from .cti_model.triattention import TriAttention


kgtransformer_hidden_size = 1024


class CTIModel(nn.Module):
    """
        Instance of a Compact Trilinear Interaction model (see https://arxiv.org/pdf/1909.11874.pdf)
    """

    def __init__(
        self, v_dim, q_dim, kg_dim, glimpse, h_dim=512, h_out=1, rank=32, k=1,
    ):
        super(CTIModel, self).__init__()

        self.glimpse = glimpse

        self.t_att = TriAttention(
            v_dim, q_dim, kg_dim, h_dim, 1, rank, glimpse, k, dropout=[0.2, 0.5, 0.2],
        )

        t_net = []
        q_prj = []
        kg_prj = []
        for _ in range(glimpse):
            t_net.append(
                TCNet(
                    v_dim,
                    q_dim,
                    kg_dim,
                    h_dim,
                    h_out,
                    rank,
                    1,
                    k=2,
                    dropout=[0.2, 0.5, 0.2],
                )
            )
            q_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))
            kg_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))

        self.t_net = nn.ModuleList(t_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.kg_prj = nn.ModuleList(kg_prj)

        self.q_pooler = FCNet([q_dim, h_dim * 2])
        self.kg_pooler = FCNet([kg_dim, h_dim * 2])

    # def forward(self, v, q, kg):
    def forward(self, v_emb, q_emb_raw, kg_emb_raw):
        """
            v: [batch, num_objs, obj_dim]
            b: [batch, num_objs, b_dim]
            q: [batch_size, seq_length]
        """
        b_emb = [0] * self.glimpse
        att, logits = self.t_att(v_emb, q_emb_raw, kg_emb_raw)

        q_emb = self.q_pooler(q_emb_raw)
        kg_emb = self.kg_pooler(kg_emb_raw)

        for g in range(self.glimpse):
            b_emb[g] = self.t_net[g].forward_with_weights(
                v_emb, q_emb_raw, kg_emb_raw, att[:, :, :, :, g]
            )

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            kg_emb = self.kg_prj[g](b_emb[g].unsqueeze(1)) + kg_emb

        joint_emb = q_emb.sum(1) + kg_emb.sum(1)
        return joint_emb, att
        
# Custom libraries
# From KILBERT
### CLASS DEFINITION ###
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class KgUniterFusion(nn.Module):
    def __init__(self, config, memnet_size):
        super(KgUniterFusion, self).__init__()
        self.dense = nn.Linear(kgtransformer_hidden_size+config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, fused_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(fused_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output 
        

class UniterForKGVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for KAVQA
    """
    def __init__(self, config, img_dim, kg_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.kgtransformer = QuestionGraphTransformer(config=None, split=None, kg_dim=kg_dim)
        if torch.cuda.is_available():
            self.kgtransformer.cuda()
            
        self.aggregator = CTIModel(
                v_dim=768,
                q_dim=768,
                kg_dim=768,
                glimpse=2,
                h_dim=256,
                h_out=1,
                rank=32,
                k=1,
            )
        
        self.pooler = KgUniterFusion(config, kgtransformer_hidden_size)
        
        self.kavqaprediction = SimpleClassifier(512, 512 * 2, num_answer, 0.5)
        
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
        txt_lens = batch['txt_lens']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        kg_feat = batch['kg_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        facts = batch['facts']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        
        
        txt_hidden_states = torch.zeros((len(txt_lens), max(txt_lens), sequence_output.size(2))).cuda()
        
        img_hidden_states = torch.zeros((len(txt_lens), sequence_output.size(1)-min(txt_lens), sequence_output.size(2))).cuda()
        
        for i in range(txt_hidden_states.size(0)):
          txt_hidden_states[i,0:txt_lens[i],:] = sequence_output[i,0:txt_lens[i],:]
          img_hidden_states[i,0:sequence_output.size(1)-txt_lens[i],:] = sequence_output[i,txt_lens[i]:,:]
        
                      
        txtkg_txt_hidden_states, txtkg_kg_hidden_states = self.kgtransformer(input_ids,
                                                   kg_feat,
                                                   extended_attention_mask=attn_masks,
                                                   output_all_encoded_layers=False)
                                                   
        result_vector, result_attention = self.aggregator(
                img_hidden_states, txt_hidden_states, txtkg_kg_hidden_states,
            )
        
        answer_scores = self.kavqaprediction(result_vector)
                     
        if compute_loss:
            targets = batch['targets']
            kvqa_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return kvqa_loss
        else:
            return answer_scores



