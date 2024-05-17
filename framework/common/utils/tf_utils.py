#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
与TensorFlow相关的公共函数
'''
import tensorflow.compat.v1 as tf
from framework.common.config.config_control import CONFIG

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.tools import freeze_graph

# 禁用tensorflow2的eager图
tf.disable_eager_execution()
tf.disable_v2_behavior()

# 设置查看ops和tensors是分配到什么设备上
tf.debugging.set_log_device_placement(True)

# tensorflow1.x
TF_VERSION_MAJOR = int(tf.__version__.split('.')[0])

# 设置TensorFlow日志级别, 在开发测试期间打开DEBUG日志, 线上采用INFO日志
def set_tensorflow_log_level():
    if CONFIG.tensorflow_log_level == 'DEBUG':
        tf.logging.set_verbosity(tf.logging.DEBUG)
    elif CONFIG.tensorflow_log_level == 'INFO':
        tf.logging.set_verbosity(tf.logging.INFO)
    elif CONFIG.tensorflow_log_level == 'WARN':
        tf.logging.set_verbosity(tf.logging.WARN)
    elif CONFIG.tensorflow_log_level == 'ERROR':
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif CONFIG.tensorflow_log_level == 'FATAL':
        tf.logging.set_verbosity(tf.logging.FATAL)

'''
固化session
'''
def freeze_session(session, graph_def, output_signature, clear_devices=False, black_list=[]):
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        session, graph_def, [name.split(':')[0] for name in output_signature.values()] + black_list)
    if clear_devices:
        for node in frozen_graph_def.node:
            node.device = ""
    return frozen_graph_def

'''
保存模型
'''
def save_frozen_model(session, output_dir, input_signature, output_signature, clear_devices=True, config=None):

    training_model_asset_collection = tf.compat.v1.get_default_graph().get_collection(
        tf.compat.v1.GraphKeys.ASSET_FILEPATHS)

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_dir)

    asset_path_nodes = [v.name.split(':')[0] for v in
                        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)]
    black_list = ['init_all_tables'] + asset_path_nodes
    frozen_graph_def = freeze_session(session, session.graph_def, output_signature, black_list=black_list)

    with tf.compat.v1.Session(graph=tf.Graph(), config=config) as frozen_sess:
        tf.import_graph_def(frozen_graph_def, name="")
        new_graph = tf.compat.v1.get_default_graph()
        for asset_file in training_model_asset_collection:
            new_graph.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS, asset_file)
        inputs = {k: new_graph.get_tensor_by_name(tensor_name) for
                  k, tensor_name in input_signature.items()}
        outputs = {k: new_graph.get_tensor_by_name(tensor_name) for
                   k, tensor_name in output_signature.items()}
        init_table_op = new_graph.get_operation_by_name("init_all_tables")
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), init_table_op)
        builder.add_meta_graph_and_variables(
            frozen_sess,
            [tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(inputs=inputs,
                                                                                       outputs=outputs),
            },
            assets_collection=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS),
            main_op=init_op,
            clear_devices=clear_devices)
        builder.save()

'''
打印模型的变量和权重
'''
def print_variables(sess, name='learner'):
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    with open(f'{CONFIG.log_dir}/variables_info_{name}.txt', 'w') as f_out:
        for key, var in zip(variable_names, values):
            f_out.write("variables: " + str(key) + "\n")
            f_out.write("weights: " + str(var) + "\n")


'''
判断机器上GPU是否安装成功
'''
def is_gpu_available():
    return tf.test.is_gpu_available()


'''
tensorflow的.pb模型和.pbtxt转换
'''
def convert_pb_to_pbtxt(pb_file_name, target_directory, pb_txt_file_name):
    if not pb_file_name or not target_directory or not pb_txt_file_name:
        return
    
    with tf.gfile.GFile(pb_file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, target_directory, pb_txt_file_name, as_text=True)

from google.protobuf import text_format
def convert_pbtxt_to_pb(pb_txt_file_name, target_directory, pb_file_name):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    if not pb_txt_file_name or not target_directory or not pb_file_name:
        return

    with tf.gfile.FastGFile(pb_txt_file_name, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, target_directory, pb_file_name, as_text=False)

'''
检查网络onnx文件所在的网络(检查IR是否形成良好), 查看网络: https://netron.app/
'''
def check_onnx_model_valid(onnx_file_name):
    if not onnx_file_name:
        return False

    # need pip3 install onnx    
    import onnx

    model = onnx.load(onnx_file_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    return True

'''
判断某个model文件是合理的
'''
def tensorflow_model_file_valid(model_path):
    if not model_path:
        return False
    
    ckpt = tf.train.get_checkpoint_state(model_path)
    if not ckpt:
        return False
    
    with tf.Graph().as_default():
        # 创建图形和Saver对象
        saver = tf.train.Saver(tf.global_variables())
    
        with tf.Session() as sess:
            try:
                # 加载模型 
                saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
                return True
            except Exception as e:
                return False