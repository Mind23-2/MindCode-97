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
#encoding=utf8
import os
import sys
import random
import numpy as np
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter



def repeat(reader):
    """Repeat a generator forever"""
    generator = reader()
    while True:
        try:
            yield next(generator)
        except StopIteration:
            generator = reader()
            yield next(generator)

def create_joint_generator(input_shape, generators, task_map_id, is_multi_task=True,max_train_steps=500):

    def empty_output(input_shape, batch_size=1):
        results = []
        for i in range(len(input_shape)):
            if input_shape[i][1] == 'int32':
                dtype = np.int32
            if input_shape[i][1] == 'int64':
                dtype = np.int64
            if input_shape[i][1] == 'float32':
                dtype = np.float32
            if input_shape[i][1] == 'float64':
                dtype = np.float64
            shape = input_shape[i][0]
            shape[0] = batch_size
            pad_tensor = np.zeros(shape=shape, dtype=dtype)
            results.append(pad_tensor)
        return results

    def wrapper(): 
        generators_inst = [repeat(gen[0]) for gen in generators]
        generators_ratio = [gen[1] for gen in generators]
        weights = [ratio/sum(generators_ratio) for ratio in generators_ratio]

        task_names = [gen[2] for gen in generators]
        task_names_ids = [0]
        for i in range(1, len(task_names)):  
            if task_names[i] == task_names[i - 1]: 
                task_names_ids.append(task_names_ids[-1])
            else: 
                task_names_ids.append(task_names_ids[-1] + 1)
        run_task_id = range(len(generators))
        
        for i in range(max_train_steps):
            idx = np.random.choice(run_task_id, p=weights)
            gen_results = next(generators_inst[idx])
            if not gen_results:
                break
            batch_size = gen_results[0].shape[0]
            results = empty_output(input_shape, batch_size)
            task_id_tensor = np.array([task_names_ids[idx]]).astype("int32")
            results[0] = task_id_tensor

            backbone_range_start = task_map_id[0][0]
            backbone_range_end = task_map_id[0][1]

            for i in range(backbone_range_start, backbone_range_end):
                results[i] = gen_results[i - 1]
            cur_gene_task = task_names_ids[idx] + 1
            for j in range(task_map_id[cur_gene_task][0], task_map_id[cur_gene_task][1]): 
                results[j] = gen_results[i]
                i += 1
            yield tuple(results)
    
    return wrapper

def create_reader(input_shape, is_multi_task, task_map_id,max_train_steps, *gens):
    """
    build reader for multi_task_learning
    """
    
    labels=['task_id', 'src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'start_positions', 'end_positions', 'mask_label', 'mask_pos', 'labels']
    writer = FileWriter(file_name='a.mindrecord', shard_num=1)
    nlp_schema = {}
    tmp=0
    for i in labels:
        nlp_schema[i]={"type": input_shape[tmp][1], "shape":input_shape[tmp][0]}
        tmp+=1
    writer.add_schema(nlp_schema, "proprocessed dataset")
    data=[]
    wrapper = create_joint_generator(input_shape, gens[0], task_map_id, is_multi_task=is_multi_task,max_train_steps=max_train_steps)
    dataset= ds.GeneratorDataset(wrapper, labels,  num_parallel_workers=8)
    return dataset
    for i in wrapper():
        data_dict = {}
        tmp=0
        for j in i:
            data_dict[labels[tmp]]=j
            tmp+=1
        data.append(data_dict)
    return
    writer.write_raw_data(data)
    writer.commit()
    # return ds.NumpySlicesDataset(data_dict,  num_parallel_workers=8,  num_shards=8, shard_id=0)
    
    
    return dataset

def joint_input_shape(input_shape_list): 
    """
    joint main task and auxiliary tasks input shape
    """
    joint_test_input_shape = input_shape_list[0][1]["backbone"] + input_shape_list[0][1]["task"]
    
    joint_train_input_shape = [([1, 1], 'int32')] # task_id_shape
    backbone_input_shape = input_shape_list[0][0]["backbone"]
    joint_train_input_shape.extend(backbone_input_shape)
    task_map_id = [(1, len(input_shape_list[0][0]["backbone"]) + 1)]

    for input_shape in input_shape_list: 
        task_input_shape = input_shape[0]["task"]
        joint_train_input_shape.extend(task_input_shape)
        task_map_id.append((task_map_id[-1][1], task_map_id[-1][1] + len(task_input_shape)))
    return joint_train_input_shape, joint_test_input_shape, task_map_id