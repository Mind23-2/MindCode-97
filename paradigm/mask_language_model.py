import mindspore.nn as nn
import mindspore.ops as P
import mindspore
from mindspore import Parameter
import mindspore.nn as nn
from mindspore.common import dtype as mstype

import math
from mindspore.common.initializer import initializer, TruncatedNormal

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None, cfg=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.cfg=cfg
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.weight.set_data(initializer(TruncatedNormal(self.cfg.initializer_range), self.weight.shape))
        
            

class LossCell(nn.Cell):
   

    def __init__(self):
        super(LossCell, self).__init__()
        self.softmax=nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = 'mean')
        self.reshape = P.Reshape()
    def construct(self, fc_out, mask_label):
        mask_label = self.reshape(mask_label, (-1,))
        loss=self.softmax(fc_out,mask_label)
        return loss
        """Defines the computation performed."""

class Model(nn.Cell):

    def __init__(self, bert_cfg):

        super(Model, self).__init__()
        self.bert_cfg=bert_cfg
        self._hidden_act = bert_cfg.hidden_act
        self._voc_size = bert_cfg.vocab_size
        self._emb_size = bert_cfg.hidden_size
        self.dense=Dense(in_channels=self._emb_size, out_channels=self._emb_size, has_bias=True,activation=self._hidden_act, cfg=self.bert_cfg).to_float(mindspore.float16)
        self.layernorm = nn.LayerNorm((self._emb_size,),begin_norm_axis=0).to_float(mindspore.float16)
        self.matmul = P.MatMul(transpose_b=True)
        self.mask_lm_out_bias_attr = Parameter(initializer(0, self._voc_size, mindspore.float32))
        self.reshape = P.Reshape()
        self.add=P.Add()
        self.cast = P.Cast()
        self.gather=P.Gather()
        self.output = P.Gather()
        self.unstack = P.Unstack()
        self.trans = P.Transpose()
    def construct(self, enc_out, cls_feats, embedding_tables, mask_pos):
        #enc_out, _, embedding_tables = self.bert(src_ids, sent_ids, input_mask)
        mask_pos = self.reshape(mask_pos,(-1,))
        # extract the first token feature in each sentence
        reshaped_emb_out = self.reshape(enc_out, (-1, self._emb_size))
        # extract masked tokens' feature
        mask_feat = self.gather(reshaped_emb_out, mask_pos, 0)

        # transform: fc
        mask_trans_feat = self.dense(mask_feat)

        #
        # transform: layer norm
        #mask_trans_feat = self.layernorm(mask_trans_feat)
        #mask_trans_feat = self.reshape(mask_trans_feat, (-1, self._emb_size))


        #mask_trans_feat = self.cast(mask_trans_feat, mindspore.float32)
        embedding_tables_16 = self.cast(embedding_tables, mindspore.float16)
        fc_out = self.matmul(mask_trans_feat, embedding_tables_16)
        fc_out = self.cast(fc_out, mindspore.float32)
        fc_out = self.add(fc_out,self.mask_lm_out_bias_attr) 
            
        return fc_out

class ModelwithLoss(nn.Cell):
    def __init__(self, bert_cfg):

        super(ModelwithLoss, self).__init__()
        self.model=Model(bert_cfg=bert_cfg)
        self.loss=LossCell()
    def construct(self, enc_out, cls_feats, embedding_tables, mask_label, mask_pos):
        fc_out=self.model(enc_out, cls_feats, embedding_tables, mask_pos)
        loss=self.loss(fc_out, mask_label)
        return loss