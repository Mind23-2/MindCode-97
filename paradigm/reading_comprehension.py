import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import dtype as mstype
import math
from mindspore.common.initializer import initializer, TruncatedNormal
import collections
import os
import mindspore

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
        self.softmax=nn.SoftmaxCrossEntropyWithLogits(sparse = True , reduction = 'mean')
        self.reshape = P.Reshape()

    def construct(self, start_logits, end_logits,start_positions,end_positions):
        start_positions = self.reshape(start_positions, (-1,))
        end_positions = self.reshape(end_positions, (-1,))

        start_loss=self.softmax(start_logits,start_positions)
        end_loss=self.softmax(end_logits,end_positions)
        return (start_loss + end_loss) / 2.0
        """Defines the computation performed."""
        
class Model(nn.Cell):

    def __init__(self, bert_cfg):
        super(Model, self).__init__()
        
        self.hidden_size = bert_cfg.hidden_size
        self.dense=Dense(self.hidden_size,2).to_float(mindspore.float16)
        self.perm = (2, 0, 1)
        self.unstack=P.Unstack()
        self.trans = P.Transpose()
        self.cast = P.Cast()
    def construct(self, enc_out, cls_feats, embedding_tables):
        #enc_out, _, _ = self.bert(src_ids, sent_ids, input_mask)
        logits = self.dense(enc_out)
        logits = self.trans(logits, self.perm)
        logits = self.cast(logits, mindspore.float32)
        start_logits, end_logits = self.unstack(logits)
        return start_logits, end_logits


class ModelwithLoss(nn.Cell):
    def __init__(self,
                 bert_cfg,
                 is_training=True,
                 use_one_hot_embeddings=False):

        super(ModelwithLoss, self).__init__()
        self.model=Model(bert_cfg=bert_cfg)
        self.loss=LossCell()
    def construct(self, enc_out, cls_feats, embedding_tables,start_positions,end_positions):
        start_logits, end_logits=self.model(enc_out, cls_feats, embedding_tables)
        loss=self.loss(start_logits, end_logits,start_positions,end_positions)
        return loss

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
def postprocess(fetch_results):
    np_unique_ids= fetch_results['unique_id']
    np_start_logits= fetch_results['start_logits']
    np_end_logits= fetch_results['end_logits']
    ret = []
    for idx in range(np_unique_ids.shape[0]):                                                                                                                                          
        if np_unique_ids[idx] < 0:
            continue
        unique_id = int(np_unique_ids[idx])
        start_logits = [float(x) for x in np_start_logits[idx].asnumpy()]
        end_logits = [float(x) for x in np_end_logits[idx].asnumpy()]
        ret.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return ret
     

    
def global_postprocess(pred_buf, processor, mtl_args, task_args):
    if not os.path.exists(mtl_args.checkpoint_path):
        os.makedirs(mtl_args.checkpoint_path)
    output_prediction_file = os.path.join(mtl_args.checkpoint_path, "predictions.json")
    output_nbest_file = os.path.join(mtl_args.checkpoint_path, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(mtl_args.checkpoint_path, "null_odds.json")

    processor.write_predictions(pred_buf, task_args.n_best_size, task_args.max_answer_length,
                                task_args.do_lower_case, output_prediction_file,
                                output_nbest_file, output_null_log_odds_file,
                                task_args.with_negative,
                                task_args.null_score_diff_threshold, task_args.verbose)
