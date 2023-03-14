# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""run multi-task  models"""
import os
import sys
import time
import argparse
import importlib
import collections
import numpy as np
import multiprocessing
import mindspore.dataset as ds
from mindspore.communication.management import init, get_rank


from utils.configure import PDConfig
from utils.configure import JsonConfig, ArgumentGroup, print_arguments
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context,Tensor
from mindspore.context import ParallelMode
from mindspore.dataset import MindDataset
import mindspore
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor

from backbone.bert_model import BertModel
backbones = {
    "bert_model": BertModel
}
sys.path.append("reader")
import joint_reader
from joint_reader import create_reader

sys.path.append("optimizer")
sys.path.append("paradigm")
sys.path.append("backbone")
from joint_model import Joint_Model 
TASKSET_PATH = "config"
import threading

class MyThread(threading.Thread):
    def __init__(self, func, args = ()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def predict(multitask_config): 
    print("Loading multi_task configure...................")
    args = PDConfig(yaml_file=[multitask_config])
    args.build()
    
    index = 0
    reader_map_task = dict()
    task_args_list = list()
    reader_args_list = list()
    id_map_task = {index: args.main_task}
    print("Loading main task configure....................")
    main_task_name = args.main_task
    task_config_files = [i for i in os.listdir(TASKSET_PATH) if i.endswith('.yaml')]
    main_config_list = [config for config in task_config_files if config.split('.')[0] == main_task_name]
    main_args = None
    for config in main_config_list: 
        main_yaml = os.path.join(TASKSET_PATH, config)
        main_args = PDConfig(yaml_file=[multitask_config, main_yaml])
        main_args.build()
        main_args.Print()
        if not task_args_list or main_task_name != task_args_list[-1][0]: 
            task_args_list.append((main_task_name, main_args))
        reader_args_list.append((config.strip('.yaml'), main_args))
        reader_map_task[config.strip('.yaml')] = main_task_name

    print("Loading auxiliary tasks configure...................")
    aux_task_name_list = args.auxiliary_task.strip().split()
    for aux_task_name in aux_task_name_list: 
        index += 1
        id_map_task[index] = aux_task_name
        print("Loading %s auxiliary tasks configure......." % aux_task_name)
        aux_config_list = [config for config in task_config_files if config.split('.')[0] == aux_task_name]
        for aux_yaml in aux_config_list: 
            aux_yaml = os.path.join(TASKSET_PATH, aux_yaml)
            aux_args = PDConfig(yaml_file=[multitask_config, aux_yaml])
            aux_args.build()
            aux_args.Print()
            if aux_task_name != task_args_list[-1][0]: 
                task_args_list.append((aux_task_name, aux_args))
            reader_args_list.append((aux_yaml.strip('.yaml'), aux_args))
            reader_map_task[aux_yaml.strip('.yaml')] = aux_task_name
 # import tasks reader module and build joint_input_shape
    input_shape_list = []
    reader_module_dict = {}
    input_shape_dict = {}
    for params in task_args_list: 
        task_reader_mdl = "%s_reader" % params[0]
        reader_module = importlib.import_module(task_reader_mdl)
        reader_servlet_cls = getattr(reader_module, "get_input_shape")
        reader_input_shape = reader_servlet_cls(params[1])
        reader_module_dict[params[0]] = reader_module
        input_shape_list.append(reader_input_shape)
        input_shape_dict[params[0]] = reader_input_shape
    train_input_shape, test_input_shape, task_map_id = joint_reader.joint_input_shape(input_shape_list)


    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    
    from mindspore import context
    if args.device:
        if args.device_id:
            context.set_context(mode=context.GRAPH_MODE, device_target=args.device, device_id=args.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    from mindspore.profiler import Profiler
    #profiler = Profiler()
    # import backbone model
    backbone_mdl = args.backbone_model
    backbone_cls = "Model"
    backbone_servlet = backbones[backbone_mdl]
    #backbone_servlet = getattr(backbone_module, backbone_cls)
    # build backbone model
    print('building model backbone...')
    conf = vars(args)
    if args.pretrain_config_path is not None:
        model_conf = JsonConfig(args.pretrain_config_path).asdict()
        for k, v in model_conf.items():
            if k in conf:
                assert k == conf[k], "ERROR: argument {} in pretrain_model_config is NOT consistent with which in main.yaml"
        conf.update(model_conf)
    
    backbone_inst = backbone_servlet(conf, is_training=True)
    print('loading pretrained...')
    param_dict = load_checkpoint('bert.ckpt')
    unloaded = load_param_into_net(backbone_inst, param_dict)
    print(unloaded)
    from paradigm.reading_comprehension import Model
    main_model=Model(backbone_inst._bertconfig)
    param_dict = load_checkpoint('rc.ckpt')
    unloaded = load_param_into_net(main_model, param_dict)
    params = reader_args_list[0]
    generator_cls = getattr(reader_module_dict[reader_map_task[params[0]]], "DataProcessor")
    generator_inst = generator_cls(params[1])
    reader_generator = generator_inst.data_generator(phase='predict', shuffle=True)
    pred_buf = []
    for i in reader_generator():
        src_ids, pos_ids, sent_ids, input_mask, unique_id = i
        a,b,c=backbone_inst(Tensor(src_ids, mindspore.int32), Tensor(sent_ids, mindspore.int32), Tensor(input_mask, mindspore.int32))
        start_logits, end_logits = main_model(a,b,c)
        task_net_mdl = importlib.import_module(main_task_name)
        postprocess = getattr(task_net_mdl, "postprocess")
        global_postprocess = getattr(task_net_mdl, "global_postprocess")
        fetch_results = {'start_logits':start_logits,'end_logits':end_logits,'unique_id':unique_id}
        res = postprocess(fetch_results)
        
        if res is not None:
            pred_buf.extend(res)
    global_postprocess(pred_buf, generator_inst, args, main_args)

def train(multitask_config): 
    print("Loading multi_task configure...................")
    args = PDConfig(yaml_file=[multitask_config])
    args.build()
    
    index = 0
    reader_map_task = dict()
    task_args_list = list()
    reader_args_list = list()
    id_map_task = {index: args.main_task}
    print("Loading main task configure....................")
    main_task_name = args.main_task
    task_config_files = [i for i in os.listdir(TASKSET_PATH) if i.endswith('.yaml')]
    main_config_list = [config for config in task_config_files if config.split('.')[0] == main_task_name]
    main_args = None
    for config in main_config_list: 
        main_yaml = os.path.join(TASKSET_PATH, config)
        main_args = PDConfig(yaml_file=[multitask_config, main_yaml])
        main_args.build()
        main_args.Print()
        if not task_args_list or main_task_name != task_args_list[-1][0]: 
            task_args_list.append((main_task_name, main_args))
        reader_args_list.append((config.strip('.yaml'), main_args))
        reader_map_task[config.strip('.yaml')] = main_task_name

    print("Loading auxiliary tasks configure...................")
    aux_task_name_list = args.auxiliary_task.strip().split()
    for aux_task_name in aux_task_name_list: 
        index += 1
        id_map_task[index] = aux_task_name
        print("Loading %s auxiliary tasks configure......." % aux_task_name)
        aux_config_list = [config for config in task_config_files if config.split('.')[0] == aux_task_name]
        for aux_yaml in aux_config_list: 
            aux_yaml = os.path.join(TASKSET_PATH, aux_yaml)
            aux_args = PDConfig(yaml_file=[multitask_config, aux_yaml])
            aux_args.build()
            aux_args.Print()
            if aux_task_name != task_args_list[-1][0]: 
                task_args_list.append((aux_task_name, aux_args))
            reader_args_list.append((aux_yaml.strip('.yaml'), aux_args))
            reader_map_task[aux_yaml.strip('.yaml')] = aux_task_name
 # import tasks reader module and build joint_input_shape
    input_shape_list = []
    reader_module_dict = {}
    input_shape_dict = {}
    for params in task_args_list: 
        task_reader_mdl = "%s_reader" % params[0]
        reader_module = importlib.import_module(task_reader_mdl)
        reader_servlet_cls = getattr(reader_module, "get_input_shape")
        reader_input_shape = reader_servlet_cls(params[1])
        reader_module_dict[params[0]] = reader_module
        input_shape_list.append(reader_input_shape)
        input_shape_dict[params[0]] = reader_input_shape
    train_input_shape, test_input_shape, task_map_id = joint_reader.joint_input_shape(input_shape_list)


    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    
    from mindspore import context
    if args.device:
        if args.device_id:
            context.set_context(mode=context.GRAPH_MODE, device_target=args.device, device_id=args.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    from mindspore.profiler import Profiler
    #profiler = Profiler()
    # import backbone model
    backbone_mdl = args.backbone_model
    backbone_cls = "Model"
    backbone_servlet = backbones[backbone_mdl]
    #backbone_servlet = getattr(backbone_module, backbone_cls)
    # build backbone model
    print('building model backbone...')
    conf = vars(args)
    if args.pretrain_config_path is not None:
        model_conf = JsonConfig(args.pretrain_config_path).asdict()
        for k, v in model_conf.items():
            if k in conf:
                assert k == conf[k], "ERROR: argument {} in pretrain_model_config is NOT consistent with which in main.yaml"
        conf.update(model_conf)
    
    backbone_inst = backbone_servlet(conf, is_training=True)
    print('loading pretrained...')
    param_dict = load_checkpoint('6.ckpt')
    unloaded = load_param_into_net(backbone_inst, param_dict)
    print(unloaded)

    if args.do_train: 
        #create joint pyreader
        print('creating readers...')
        gens = []
        main_generator = ""
        threads = []
        for params in reader_args_list: 
            generator_cls = getattr(reader_module_dict[reader_map_task[params[0]]], "DataProcessor")
            generator_inst = generator_cls(params[1])
            t = MyThread(generator_inst.data_generator, args = ('train', True, ))
            threads.append(t)
            t.start()
            #reader_generator = generator_inst.data_generator(phase='train', shuffle=True)
            if not main_generator: 
                main_generator = generator_inst
            #gens.append((reader_generator, params[1].mix_ratio, reader_map_task[params[0]]))
        i = 0
        for t in threads:
            t.join()
            params = reader_args_list[i]
            gens.append((t.get_result(), params[1].mix_ratio, reader_map_task[params[0]]))
            i += 1
    
        print('building task models...')
        num_train_examples = main_generator.get_num_examples()
        if main_args.in_tokens:
            max_train_steps = int(main_args.epoch * num_train_examples) // (
                    main_args.batch_size // main_args.max_seq_len) 
        else:
            max_train_steps = int(main_args.epoch * num_train_examples) // (
                main_args.batch_size) 
        mix_ratio_list = [task_args[1].mix_ratio for task_args in task_args_list]
        args.max_train_steps = int(max_train_steps * (sum(mix_ratio_list) / main_args.mix_ratio))
        print("Max train steps: %d" % args.max_train_steps)
        joint_generator = create_reader(train_input_shape, True, task_map_id, args.max_train_steps, gens)
    
    # labels=['task_id', 'src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'start_positions', 'end_positions', 'mask_label', 'mask_pos', 'labels']
    # joint_generator = MindDataset("./a.mindrecord" , labels, num_parallel_workers =8 )

    all_loss_list = []
    for i in range(len(task_args_list)): 
        task_name = task_args_list[i][0]
        task_args = task_args_list[i][1]

        if hasattr(task_args, 'paradigm'):
            task_net = task_args.paradigm
        else:
            task_net = task_name

        task_net_mdl = importlib.import_module(task_net)
        task_net_cls = getattr(task_net_mdl, "ModelwithLoss")
        model = task_net_cls(backbone_inst._bertconfig)
        all_loss_list.append(model)
        
    joint_model = Joint_Model(all_loss_list,backbone_inst)
    print('begin training...')
    from mindspore.nn import  Adam
    optimizer = Adam(params=joint_model.trainable_params(), learning_rate=args.learning_rate)#, weight_decay = args.weight_decay)
                             
    import mindspore.nn as nn
    from mindspore.train.model import Model

    train_network = nn.TrainOneStepCell(joint_model, optimizer)
    train_network.set_train()
    model = Model(train_network)
    
    # cnt = 0
    # for i in joint_generator.create_dict_iterator():
    #     print(cnt)
    #     cnt += 1
        
    #     joint_model(i['task_id'],i['src_ids'],i['pos_ids'],i['sent_ids'],i['input_mask'],i['start_positions'],i['end_positions'],i['mask_label'],i['mask_pos'],i['labels'])
        
    # epochs = int(main_args.epoch*args.max_train_steps/4)
    model.train(main_args.epoch, joint_generator, callbacks=[tm,lm])
    #profiler.analyse()

    main_model = all_loss_list[0].model
    mindspore.save_checkpoint(main_model,'rc.ckpt')
    mindspore.save_checkpoint(backbone_inst,'bert.ckpt')

    
    

    
    
    

    
    

    
    
tm=TimeMonitor()
lm=LossMonitor()

if __name__ == '__main__':

    multitask_config = "mtl_config.yaml"
    train(multitask_config)
    predict(multitask_config)

    
