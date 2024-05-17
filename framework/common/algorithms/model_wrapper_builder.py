#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.common.config.config_control import CONFIG
from framework.common.config.algo_conf import AlgoConf
from framework.common.algorithms.network_builder import NetworkBuilder
from framework.common.algorithms.model_wrapper_pytorch import ModelWrapperPytorch
from framework.common.algorithms.model_wrapper_tcnn import ModelWrapperTcnn
from framework.common.algorithms.model_wrapper_tensorflow_simple import ModelWrapperTensorflowSimple
from framework.common.algorithms.model_wrapper_tensorflow_complex import ModelWrapperTensorflowComplex
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.algorithms.model_wrapper_tensorrt import ModelWrapperTensorRT


'''
对ModelWrapper类的封装, 目前存在多种使用场景, 待在算法同事实际交付过程中再沉淀, 目前支持多种预测方案
1. tensorflow, 框架加载业务类, 定义graph, session, 业务使用
2. tensorflow, 框架加载业务类, 业务定义graph, session, 业务使用
3. 采用pytorch类
4. 采用tcnn类
5. 采用其他方式
'''
class ModelWrapperBuilder:
    def __init__(self) -> None:
        pass
    
    '''
    根据配置项加载不同的ModelWraper
    '''
    def create_model_wrapper(self, model, logger, server = None):
        
        # 根据配置项选择不同的ModelWrapper类进行实例化并返回
        if KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
            return ModelWrapperTensorflowSimple(model, logger, server)

        elif KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            return ModelWrapperTensorflowComplex(model, logger, server)

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            return ModelWrapperPytorch(model, logger, server)

        elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
            return ModelWrapperTcnn(model, logger, server)
        
        elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            return ModelWrapperTensorRT(model, logger, server)
            
        else:
            # 如果配置项不匹配任何已知的ModelWrapper类，则返回None
            return None