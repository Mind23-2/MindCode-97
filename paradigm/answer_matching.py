import mindspore.nn as nn
import mindspore.ops as P
import mindspore
from mindspore import Parameter
import mindspore.nn as nn
from mindspore.common import dtype as mstype
import math
from mindspore.common.initializer import initializer, TruncatedNormal

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(TruncatedNormal(0.02), self.weight.shape))
        if self.has_bias:
            self.bias.set_data(initializer(0, self.bias.shape))



class LossCell(nn.Cell):
   
    def __init__(self):
        super(LossCell, self).__init__()
        self.softmax=nn.SoftmaxCrossEntropyWithLogits(sparse = True, reduction = 'mean')
        self.reshape = P.Reshape()
    def construct(self, logits, labels):
        labels = self.reshape(labels, (-1,))
        loss=self.softmax(logits,labels)
        return loss
        """Defines the computation performed."""


class Model(nn.Cell):

    def __init__(self, bert_cfg):
        super(Model, self).__init__()

        self.dropout=nn.Dropout(keep_prob=0.9, dtype=mindspore.float32)
        self.hidden_size = bert_cfg.hidden_size
        self.dense = Dense(self.hidden_size,2).to_float(mindspore.float16)
        self.cast = P.Cast()

    def construct(self, enc_out, cls_feats, embedding_tables):
        #_, cls_feats, _ = self.bert(src_ids, sent_ids, input_mask)
        cls_feats = self.dropout(cls_feats)
        logits = self.dense(cls_feats)
        logits = self.cast(logits, mindspore.float32)
        return logits

class ModelwithLoss(nn.Cell):
    def __init__(self, bert_cfg):

        super(ModelwithLoss, self).__init__()
        self.model=Model(bert_cfg=bert_cfg)
        self.loss=LossCell()
    def construct(self, enc_out, cls_feats, embedding_tables, labels):
        logits=self.model(enc_out, cls_feats, embedding_tables)
        loss=self.loss(logits, labels)
        return loss