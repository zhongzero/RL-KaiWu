#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块包装并实现了强化学习概念中的agent的预测过程，用户自定义的预测器可以在本模块中实现

'''
@Project :1v1 
@File    :local_predictor.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

from framework.common.utils.tf_utils import *


class BasePredictor(object):
    """The BasePredictor class is an abstract base class.

    这是一个抽象基类
    """
    
    def __init__(self):
        pass

    def load_model(self, model_name):
        raise NotImplementedError

    def inference(self, input_list, output_list):
        raise NotImplementedError


class LocalTFPredictor(BasePredictor):
    """An LocalTFPredictor object is used to perform model loading and
    inference operations for tensorflow models.
    None of the methods are thread safe. The object is intended to be used
    by a single thread and simultaneously calling different methods
    with different threads is not supported and will cause undefined
    behavior.

    LocalTFPredictor对象用于执行TensorFlow模型的模型加载和推理操作。这些方法都不是线程安全的。该对象旨在由单个线程使用，不支持同时使用不同线程调用不同方法，这将导致未定义的行为。
    """

    def __init__(self, graph, cpu_num):
        """
        Args:
            graph: tf.Graph
                A tf.Graph object that contains a set of tf.Operation and
                tf.Tensor objects.
            model_pool_addr: list of str
                The address of model_pool, e.g. ['localhost:10013:10014']
            cpu_num: int
                cpu number for tf session config
            
        参数：
            graph: tf.Graph对象包含一组tf.Operation和tf.Tensor对象。
            model_pool_addr: model_pool的地址， 例如['localhost:10013:10014']
            cpu_num: tf会话配置的CPU数量
        """
        super().__init__()
        
        # init session
        self._graph = graph
        self._sess_config = tf.ConfigProto(
            device_count={"CPU": cpu_num},
            inter_op_parallelism_threads=cpu_num,
            intra_op_parallelism_threads=cpu_num,
            log_device_placement=True,
            )
        self._sess_config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph, config=self._sess_config)

    def load_model(self, model_path):
        """Request the model_pool to get the specified model file,
        then load or reload the model.

        Parameters
        ----------
        model_key : str
            The name of the model to be loaded.

        Returns
        ------
        bool
            The result of load_model.

        发送请求到model_pool获取指定的模型文件，然后加载或重新加载模型。
        参数：
            model_path: 想要加载的模型路径
        返回值：模型是否加载成功
        """
        return self._tf_load_api(model_path)
    
    def load_last_model(self, model_path):
        return self.tf_load_api(model_path)

    def inference(self, input_list, output_list):
        """Use tf.Session.run to run TensorFlow operations.
        Feed tensors in 'input_list' and evaluate tensors in 'output_list'.

        使用tf.Session.run来运行TensorFlow操作。在'input_list'中提供张量作为输入，并在'output_list'中评估张量。
        """

        input_names = [inp.name for inp in input_list]
        input_datas = [inp.data for inp in input_list]
        feed_dict = dict(zip(input_names, input_datas))
        output_names = [output.name for output in output_list]
        output_datas = self._sess.run(output_names, feed_dict=feed_dict)
        for output, data in zip(output_list, output_datas):
            output.data = data
        return output_list

    def _tf_load_api(self, model_path):
        raise NotImplementedError
    
    def tf_load_api(self, model_path):
        raise NotImplemented


class LocalCkptPredictor(LocalTFPredictor):
    """An LocalCkptPredictor object is used to perform model loading and
    inference operations for tensorflow models saved as checkpoint.

    Parameters
    ----------
    graph: tf.Graph
        A tf.Graph object that contains a set of tf.Operation and
        tf.Tensor objects.
    model_pool_addr: list of str
        The address of model_pool, e.g. ['localhost:10013:10014']
    cpu_num: int
        cpu number for tf session config
    ckpt_name: str
        The prefix of model file, default as 'model.ckpt'

    LocalCkptPredictor对象用于执行以checkpoint形式保存的TensorFlow模型的模型加载和推理操作。
    参数：
        graph: tf.Graph对象包含一组tf.Operation和tf.Tensor对象。
        model_pool_addr: model_pool的地址， 例如['localhost:10013:10014']
        cpu_num: tf会话配置的CPU数量
        ckpt_name: 模型文件的前缀，默认为'model.ckpt'
    """

    def __init__(self, graph, cpu_num=1, ckpt_name="model.ckpt"):
        super().__init__(graph, cpu_num)
        self._ckpt_name = ckpt_name
        with self._graph.as_default():
            self._saver = tf.train.Saver(tf.global_variables())

    def _tf_load_api(self, model_path):
        """Load checkpoint.

        加载checkpoint文件
        """
        ckpt_path = "%s/%s" % (model_path, self._ckpt_name)
        self._saver.restore(self._sess, ckpt_path)
        return True
    
    def tf_load_api(self, model_path):
        # if evaluate mode, directly load the checkpoint
        # 如果是评估模式，直接读取某个ckpt
        if CONFIG.run_mode == 'eval':
            self._saver.restore(self._sess, model_path)
            return
        
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            # load the latest checkpoint
            # 加载最新的模型
            self._saver.restore(self._sess, ckpt.all_model_checkpoint_paths[-1])