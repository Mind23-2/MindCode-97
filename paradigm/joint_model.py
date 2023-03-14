import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import dtype as mstype
from mindspore import Tensor 
import mindspore


class Joint_Model(nn.Cell):

    def __init__(self,
                 models,bert_model):
        super(Joint_Model, self).__init__()
        self.rc = models[0]
        self.mlm = models[1]
        self.am = models[2]
        self.bert_model=bert_model
        self.stack = P.Stack()
        self.onehot = nn.OneHot(depth=3)
        self.mul = nn.MatMul()
        self.sum= P.ReduceSum()

    def construct(self, task_id, src_ids, pos_ids, sent_ids, input_mask, start_positions, end_positions, mask_label, mask_pos, labels):
        enc_out, cls_feats, embedding_tables = self.bert_model(src_ids, sent_ids, input_mask)
        rc_loss =self.rc(enc_out, cls_feats, embedding_tables,start_positions,end_positions)
        mlm_loss =self.mlm(enc_out, cls_feats, embedding_tables, mask_label, mask_pos)
        am_loss=self.am(enc_out, cls_feats, embedding_tables, labels)
        all_loss=self.stack([rc_loss,mlm_loss,am_loss])
        one = self.onehot(task_id)
        loss=self.mul(one,all_loss)
        loss=self.sum(loss)
        return loss

        