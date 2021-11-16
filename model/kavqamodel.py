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

class SuperSimpleNet(nn.Module):
    """
        Simple class for non-linear fully-connected network
    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super(SuperSimpleNet, self).__init__()

        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 256)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        return (x)


### UTILS FUNCTION DEFINITION ###
def mode_product(tensor, matrix_1, matrix_2, matrix_3, matrix_4, n_way=3):

    # mode-1 tensor product
    tensor_1 = (
        tensor.transpose(3, 2)
        .contiguous()
        .view(
            tensor.size(0),
            tensor.size(1),
            tensor.size(2) * tensor.size(3) * tensor.size(4),
        )
    )
    tensor_product = torch.matmul(matrix_1, tensor_1)
    tensor_1 = tensor_product.view(
        -1, tensor_product.size(1), tensor.size(4), tensor.size(3), tensor.size(2)
    ).transpose(4, 2)

    # mode-2 tensor product
    tensor_2 = (
        tensor_1.transpose(2, 1)
        .transpose(4, 2)
        .contiguous()
        .view(
            -1, tensor_1.size(2), tensor_1.size(1) * tensor_1.size(3) * tensor_1.size(4)
        )
    )
    tensor_product = torch.matmul(matrix_2, tensor_2)
    tensor_2 = (
        tensor_product.view(
            -1,
            tensor_product.size(1),
            tensor_1.size(4),
            tensor_1.size(3),
            tensor_1.size(1),
        )
        .transpose(4, 1)
        .transpose(4, 2)
    )
    tensor_product = tensor_2

    if n_way > 2:
        # mode-3 tensor product
        tensor_3 = (
            tensor_2.transpose(3, 1)
            .transpose(4, 2)
            .transpose(4, 3)
            .contiguous()
            .view(
                -1,
                tensor_2.size(3),
                tensor_2.size(2) * tensor_2.size(1) * tensor_2.size(4),
            )
        )
        tensor_product = torch.matmul(matrix_3, tensor_3)
        tensor_3 = (
            tensor_product.view(
                -1,
                tensor_product.size(1),
                tensor_2.size(4),
                tensor_2.size(2),
                tensor_2.size(1),
            )
            .transpose(1, 4)
            .transpose(4, 2)
            .transpose(3, 2)
        )
        tensor_product = tensor_3

    if n_way > 3:
        # mode-4 tensor product
        tensor_4 = (
            tensor_3.transpose(4, 1)
            .transpose(3, 2)
            .contiguous()
            .view(
                -1,
                tensor_3.size(4),
                tensor_3.size(3) * tensor_3.size(2) * tensor_3.size(1),
            )
        )
        tensor_product = torch.matmul(matrix_4, tensor_4)
        tensor_4 = (
            tensor_product.view(
                -1,
                tensor_product.size(1),
                tensor_3.size(3),
                tensor_3.size(2),
                tensor_3.size(1),
            )
            .transpose(4, 1)
            .transpose(3, 2)
        )
        tensor_product = tensor_4

    return tensor_product


### CLASS DEFINITION ###
class TCNet(nn.Module):
    def __init__(
        self,
        v_dim,
        q_dim,
        kg_dim,
        h_dim,
        h_out,
        rank,
        glimpse,
        act="ReLU",
        k=1,
        dropout=[0.2, 0.5, 0.2],
    ):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.kg_dim = kg_dim
        self.h_out = h_out
        self.rank = rank
        self.h_dim = h_dim * k
        self.hv_dim = int(h_dim / rank)
        self.hq_dim = int(h_dim / rank)
        self.hkg_dim = int(h_dim / rank)
        
        self.basic_nn = torch.nn.Linear(768, 256)

        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        self.v_tuckers_simple = SuperSimpleNet(0)
        self.kg_tucker = FCNet([kg_dim, self.h_dim], act=act, dropout=dropout[2])

        if self.h_dim < 1024:
            self.kg_tucker = FCNet([kg_dim, self.h_dim], act=act, dropout=dropout[2])
            self.q_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0])
                    for _ in range(rank)
                ]
            )
            self.v_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1])
                    for _ in range(rank)
                ]
            )
            self.kg_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hkg_dim], act=act, dropout=dropout[2])
                    for _ in range(rank)
                ]
            )

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(
                torch.Tensor(
                    1, rank, self.hv_dim, self.hq_dim, self.hkg_dim, glimpse, h_out
                ).normal_()
            )
        self.dropout = nn.Dropout(dropout[1])

    def forward(self, v, q, kg):
        f_emb = 0

        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        kg_tucker = self.kg_tucker(kg)

        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            kg_ = self.kg_net[r](kg_tucker)
            f_emb = (
                mode_product(self.T_g[:, r, :, :, :, :, :], v_, q_, kg_, None) + f_emb
            )

        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q, kg, w):
        v_ = self.v_tucker(v).transpose(2, 1)  # b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)  # b x d x q x 1
        kg_ = self.kg_tucker(kg).transpose(2, 1).unsqueeze(3)  # b x d x kg

        logits = torch.einsum("bdv,bvqa,bdqi,bdaj->bdij", [v_, w, q_, kg_])
        logits = logits.squeeze(3).squeeze(2)

        return logits


### CLASS DEFINITION ###
class TriAttention(nn.Module):
    def __init__(
        self,
        v_dim,
        q_dim,
        kg_dim,
        h_dim,
        h_out,
        rank,
        glimpse,
        k,
        dropout=[0.2, 0.5, 0.2],
    ):
        super(TriAttention, self).__init__()
        self.glimpse = glimpse
        self.TriAtt = TCNet(
            v_dim, q_dim, kg_dim, h_dim, h_out, rank, glimpse, dropout=dropout, k=k
        )

    def forward(self, v, q, kg):
        v_num = v.size(1)
        q_num = q.size(1)
        kg_num = kg.size(1)
        logits = self.TriAtt(v, q, kg)

        mask = (
            (0 == v.abs().sum(2))
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(logits.size())
        )
        logits.data.masked_fill_(mask.data, -float("inf"))

        p = torch.softmax(
            logits.contiguous().view(-1, v_num * q_num * kg_num, self.glimpse), 1
        )

        return p.view(-1, v_num, q_num, kg_num, self.glimpse), logits



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
        
        self.basic_nn = nn.Linear(768, 256)
        
        self.kavqaprediction = SimpleClassifier(512, 512 * 2, num_answer, 0.5)
        
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
        
                      
        txtkg_kg_hidden_states, txtkg_kg_hidden_pooled = self.kgtransformer(input_ids,
                                                   kg_feat,
                                                   extended_attention_mask=attn_masks,
                                                   output_all_encoded_layers=False)
         
        #print(txt_hidden_states.shape, img_hidden_states.shape, txtkg_kg_hidden_states.shape)
        #print('txt, img, kg')
        
        #kg_emb_test = self.basic_nn(txtkg_kg_hidden_states)
        #exit()
        
        #txt_img_kg_concat = torch.cat((txt_hidden_states, img_hidden_states, txtkg_kg_hidden_states.float()), dim=1)
        #print(txt_img_kg_concat.shape)
        #exit()
                                          
        result_vector, result_attention = self.aggregator(
                img_hidden_states.half(), txt_hidden_states.half(), txtkg_kg_hidden_states,
            )
        
        answer_scores = self.kavqaprediction(result_vector)
                     
        if compute_loss:
            targets = batch['targets']
            kvqa_loss = F.cross_entropy(
                answer_scores, targets, reduction='none')
            return kvqa_loss
        else:
            return answer_scores



