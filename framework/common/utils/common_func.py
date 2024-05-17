#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
import shutil
import threading
import time
import hashlib
import os
import tarfile
import uuid
import socket
import re
# need pip install schedule
import schedule
import lz4.block
import zstd
import subprocess
import psutil
import signal
import random
import json
import fcntl
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.config.config_control import CONFIG

try:
    import _pickle as pickle
except:
    import pickle

# TimeIt
class TimeIt:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.end = time.monotonic()
        self.interval = self.end - self.start

# Context
class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, other):
        self.__dict__.update(other)

# ip:port
def str_to_addr(addr_str):
    fields = addr_str.split(":")
    assert len(fields) == 2, "addr_str format must be ip:port"

    return fields[0], int(fields[1])

# hash
def hashlib_md5(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

'''
读取某个文件, 前4096个字节计算下md5sum值
'''
def md5sum(file_name):
    if not file_name:
        return ""
    
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

# 获取GPU
def get_local_rank():
    local_rank = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    return int(local_rank)

def get_local_size():
    local_size = os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE', '1')
    return int(local_size)

'''
Python中没有基于DCE的, 所以uuid2可以忽略
uuid4存在概率性重复, 由无映射性, 最好不用
若在Global的分布式计算环境下, 最好用uuid1
若有名字的唯一性要求,最好用uuid3或uuid5
'''
def get_uuid():
    uuid_value = uuid.uuid1()
    # 获取的数据控制在int32范围内(-2147483648, 2147483648), 即取字符串形式的后8位
    return int(str(uuid_value.int)[-8:])

'''
定时器, 支持按照间隔时间, 执行操作, 支持秒, 分钟, 小时, 天等单位
'''
def set_schedule_event(time_interval, run_func, op_gap='minutes'):
    if op_gap == 'minutes':
        schedule.every(time_interval).minutes.do(run_func)
    elif op_gap == 'seconds':
        schedule.every(time_interval).seconds.do(run_func)
    elif op_gap == 'hour':
        schedule.every(time_interval).hour.do(run_func)
    elif op_gap == 'day':
        schedule.every(time_interval).day.do(run_func)
    else:
        # 需要按照需求, 添加功能
        pass

'''
获取某个文件第一行和最后一行
注意存在多个进程操作同一个文件的场景, 比如M个进程读操作N个进程写操作, 这样就需要加锁
'''
def get_first_last_line_from_file(file_name):
    first_line = None
    last_line = None

    if not os.path.exists(file_name):
        return first_line, last_line

    file_size = os.path.getsize(file_name)
    if file_size == 0:
        return first_line, last_line

    blocksize = 1024
    with open(file_name, 'rb') as dat_file:
        # 获取文件锁
        fcntl.flock(dat_file.fileno(), fcntl.LOCK_SH)

        headers = dat_file.readline().strip()
        if file_size > blocksize:
            maxseekpoint = file_size // blocksize
            dat_file.seek(maxseekpoint * blocksize)
        else:
            maxseekpoint = 0
            dat_file.seek(0)
        lines = dat_file.readlines()
        if lines:
            last_line = lines[-1].strip()

            # 检查最后一行是否完整
            if last_line[-1:] != b'\n':
                dat_file.seek(0, os.SEEK_END)
                while dat_file.read(1) != b'\n':
                    dat_file.seek(-2, os.SEEK_CUR)
                last_line = dat_file.readline().strip()

        # 释放文件锁
        fcntl.flock(dat_file.fileno(), fcntl.LOCK_UN)

    return headers.decode() if headers else None, last_line.decode() if last_line else None

'''
获取文件首行和末行的内容, 实际运行起来在文件内容小于2行时报错
'''
def get_first_line_and_last_line_from_file(file_name):
    first_line = None
    last_line = None

    if not os.path.exists(file_name):
        return first_line, last_line

    with open(file_name, 'rb') as f:  # 打开文件
        # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
        first_line = f.readline()  # 取第一行
        offset = -50  # 设置偏移量
        while True:
            """
            file.seek(off, whence=0), 从文件中移动off个操作标记(文件指针)，正往结束方向移动，负往开始方向移动。
            如果设定了whence参数,就以whence设定的起始位为准,0代表从头开始,1代表当前位置,2代表文件最末尾位置。 
            """
            f.seek(offset, 2)  # seek(offset, 2)表示文件指针：从文件末尾(2)开始向前50个字节(-50)
            lines = f.readlines()  # 读取文件指针范围内所有行
            if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                last_line = lines[-1]  # 取最后一行
                break
            # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
            # 所以off翻倍重新运行，直到readlines不止一行
            offset *= 2
    
    return first_line.decode(), last_line.decode()


'''
获取文件里从第2行到文件末尾中的随机一行
'''
def get_random_line_from_file(file_name):
    random_line = None

    # 判断文件不存在则返回
    if not os.path.exists(file_name):
        return random_line
    
    with open(file_name, 'r', encoding='utf-8') as f:
        # 获取文件总行数
        total_lines = sum(1 for _ in f)
        
        # 随机生成一个整数，表示要获取的行数
        line_num = random.randint(1, total_lines)

        # 重置文件指针
        f.seek(0)
        # 遍历文件，获取指定行的内容
        for i, line in enumerate(f):
            if i == line_num - 1:
                random_line = line.strip()
        
    # 如果文件为空或者n大于文件总行数，返回空字符串
    return random_line.decode()

'''
获取文件最后2行
'''
def get_last_two_line_from_file(file_name):
    last_two_line = None

    # 判断文件不存在则返回
    if not os.path.exists(file_name):
        return last_two_line

    with open(file_name, 'rb') as f:  # 打开文件
        f.seek(0, 0)
        lines = f.readlines()
        if len(lines)<3:
            # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
            offset = -50  # 设置偏移量
            while True:
                """
                file.seek(off, whence=0), 从文件中移动off个操作标记(文件指针)，正往结束方向移动，负往开始方向移动。
                如果设定了whence参数,就以whence设定的起始位为准,0代表从头开始,1代表当前位置,2代表文件最末尾位置。 
                """
                f.seek(offset, 2)  # seek(offset, 2)表示文件指针：从文件末尾(2)开始向前50个字节(-50)
                lines = f.readlines()  # 读取文件指针范围内所有行
                if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                    last_two_line = lines[-1]  # 取最后一行
                    break
                # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
                # 所以off翻倍重新运行，直到readlines不止一行
                offset *= 2
            # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
            offset = -50  # 设置偏移量
            
        else:
            last_two_line = lines[-2]  # 取倒数第二行   
    return last_two_line.decode()

# 加载旧模型需要修改checkpoint文件
def fix_checkpoint_file(ckpt_file, checkpoint_id):
    with open(ckpt_file, 'rb') as f: 
        lines = f.readlines()
    if len(lines)<=2:
        return
    
    os.remove(ckpt_file)
    with open(ckpt_file, 'wb') as f:
        for i,line in enumerate(lines):
            line = line.decode()
            if i ==0:
                old_checkpoint_id = line.split(f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1]
                old_checkpoint_id = re.findall(r'\d+\.?\d*', old_checkpoint_id)[0]
                
                line = line.replace(old_checkpoint_id, checkpoint_id)
                
            elif i == len(lines)-1:
                break
            f.write(line.encode())

'''
tar包压缩
'''
def make_tar_file(output_file_name, source_dir):
    if not output_file_name or not source_dir:
        return
    
    with tarfile.open(output_file_name, 'w') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

'''
tar包解压, 注意参数传递, 比如有zip则参数为r:zg
'''
def tar_flie_extract(input_file_name, destination_dir):
    if not input_file_name or not destination_dir:
        return
    
    try:
        tar = tarfile.open(input_file_name, 'r')
        file_names = tar.getnames()
        for file in file_names:
            tar.extract(file, destination_dir)
        tar.close()
    except Exception as e:
        raise e

'''
清空某个文件夹, 采用先删除, 再新增文件夹的方法
'''
def clean_dir(dir_path):
    if not dir_path:
        return
    
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)


'''
创建文件夹, 目录存在则跳过
'''
def make_single_dir(dir_path):
    if not dir_path:
        return

    folder = os.path.exists(dir_path)
    if not folder:
        os.makedirs(dir_path)

'''
根据函数名, 注意传递的是函数不是字符串
返回函数内容
'''
def get_fun_content_by_name(fun_name):
    import inspect

    if not fun_name or not inspect.isfunction(fun_name):
        print('func_name is None or not function')
        return
    
    return inspect.getsource(fun_name)


'''
在特定的字符串前或者后增加字符串
old_str: 原有的字符串
to_insert_str: 需要插入的字符串
to_find_str: 查找的字符串
before_or_after: 在查找字符串to_find_str前或者后插入to_insert_str
'''
def insert_any_string(old_str, to_insert_str, to_find_str, before_or_after):
    if not old_str or not to_insert_str or not to_find_str:
        return
    
    idx = old_str.find(to_find_str)
    if before_or_after == 'before':
        final_string = old_str[:idx] + to_insert_str + old_str[idx:]
    else:
        pass

    return final_string


'''
python 获取本机IP
'''
def get_host_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    
    return IP

'''
判断2个list是否相等, 主要是先排序再判断
'''
def is_list_eq(listA, listB):
    if not listA or not listB:
        return False

    listA.sort()
    listB.sort()

    return listA == listB

'''
2个list的差集
返回的数据包括list_A_have_B_not_have, list_B_have_A_not_have
'''
def list_diff(listA, listB):
    if not listA or not listB:
        return []

    # listB中有但是listA中没有
    list_A_have_B_not_have = list(set(listB).difference(set(listA)))
    # listA中有但是listB中没有
    list_B_have_A_not_have = list(set(listA).difference(set(listB)))

    return list_A_have_B_not_have, list_B_have_A_not_have

'''
获取当前时间的前后多少秒, 多少分钟, 前后多少小时, 前后多少天
ways: 支持天、小时、分钟、秒
delta: 差量, 如果是正则是当前时间往后, 如果是负则当前时间往前, 如果是0则就是当前时间
'''
def get_any_time_from_now(delta, ways='day'):
    now_time = datetime.datetime.now()
    end_time = None

    if 'day' == ways:
        end_time = now_time + datetime.timedelta(days=delta)
    elif 'hour' == ways:
        end_time = now_time + datetime.timedelta(hours=delta)
    elif 'min' == ways:
        end_time = now_time + datetime.timedelta(minutes=delta)
    elif 'second' == ways:
        end_time = now_time + datetime.timedelta(seconds=delta)
    else:
        pass
    
    return end_time


'''
python 执行shell语句命令
0表示执行成功, 非0表示执行失败
'''
def python_exec_shell(shell_content):
    if not shell_content:
        return 1, None

    result_code, result_str = subprocess.getstatusoutput(shell_content)
    return result_code, result_str

'''
actror/learner增加来自aisrv的TCP连接数目统计
'''
def actor_learner_aisrv_count(host, srv_name):
    if not srv_name or not host:
        return 0
    
    port = 0
    if KaiwuDRLDefine.SERVER_ACTOR == srv_name:
        port = CONFIG.zmq_server_port
    elif KaiwuDRLDefine.SERVER_LEARNER == srv_name:
        port = CONFIG.reverb_svr_port
    elif KaiwuDRLDefine.SERVER_AISRV == srv_name:
        port = CONFIG.aisrv_server_port
    else:
        pass

    # 建立连接的TCP数目
    cmd = f'ss -ano  | grep {host}:{port} | grep ESTAB | wc -l'

    result_code, result_str = python_exec_shell(cmd)
    if result_code != 0:
        return 0
    
    return int(result_str)

'''
获取GPU机器的型号, 支持异构GPU

shell命令返回的结果形如:
GPU 0: GRID T4-8C (UUID: GPU-cdd63ee4-0be7-11ed-9562-0c1a84aad0c2)

函数返回的结果形如:
GRID T4
'''
def get_gpu_machine_type_by_shell():
    cmd = 'nvidia-smi -L'

    result_code, result_str = python_exec_shell(cmd)

    # 解析结果
    if result_code or not result_str:
        return None
    
    try:
        gpu_machine_type = result_str.split('GPU 0:')[1].split('(')[0]
    except Exception as e:
        return None
    
    return gpu_machine_type

'''
1. get_gpu_machine_type_by_shell返回空时当CPU处理
2. 其余看属于哪个GPU场景
'''
def get_gpu_machine_type():
    gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_CPU

    gpu_machine_type_shell = get_gpu_machine_type_by_shell()
    if not gpu_machine_type_shell:
        pass
    else:
        if KaiwuDRLDefine.GPU_MACHINE_A100 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_A100
        elif KaiwuDRLDefine.GPU_MACHINE_V100 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_V100
        elif KaiwuDRLDefine.GPU_MACHINE_T4 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_T4
        else:
            pass

    return gpu_machine_type

'''
压缩, 主要是为了扩展, 以后方便多种算法
流程:
1. 进行序列化, 可选方法有pickle/protobuf
2. 进行压缩, 可选方法有lz4/zstd
'''
def compress_data(data, serialize=True):
    
    if not data:
        return data

    if not CONFIG.use_compress_decompress:
        return data
    
    try:
        # 采用pickle序列化
        if serialize:
            if CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PICKLE:
                data = pickle.dumps(data)
            elif CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PROTOBUF:
                data = data.SerializeToString()

        if CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_LZ4:
            compress_msg = lz4.block.compress(data, mode='fast', store_size=False)
        elif CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_ZSTD:
            # 设置为1时, 压缩耗时较小
            compress_msg = zstd.compress(data, 1)
        else:
            compress_msg = data

    # 失败场景下, 返回原始数据
    except Exception as e:
        print(f'compress_data error {str(e)}')
        compress_msg = data

    return compress_msg


'''
解压缩, 主要是为了扩展, 以后方便多种算法
流程:
1. 进行反序列化, 可选方法有pickle/protobuf
2. 进行解压缩, 可选方法有lz4/zstd
'''
def decompress_data(data, serialize=True):
    if not data:
        return data

    if not CONFIG.use_compress_decompress:
        return data
    
    try:
        if CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_LZ4:
            decompress_msg = lz4.block.decompress(data, uncompressed_size=CONFIG.lz4_uncompressed_size)
        
        elif CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_ZSTD:
            decompress_msg = zstd.decompress(data)

        else:
            decompress_msg = data
        
        if serialize:
            # 采用pickle反序列化
            if CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PICKLE:
                decompress_msg = pickle.loads(decompress_msg, encoding='bytes')
            # 采用protobuf序列化
            elif CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PROTOBUF:
                decompress_msg = decompress_msg

    except Exception as e:
        print(f'decompress_data error {str(e)}')
        decompress_msg = data

    return decompress_msg

'''
CPU 绑核操作, 规避因为CPU调度引起的时延大问题
'''
def cpu_affinity(pid, cpu_idx):
    # 如果设置的idx小于0, 则默认绑定在CPU 核1上
    if not isinstance(cpu_idx, list):
        cpu_idx = [cpu_idx]
    
    p = psutil.Process(pid)
    if not p:
        return False
    
    # 如果CPU列表为空, 意味着绑定到所有可用核上
    p.cpu_affinity(cpu_idx)
    return True


'''
根据输入的list, 生成平均值和最大值
'''
def get_mean_and_max(data):
    mean_value = 0
    max_value = 0

    if not data:
        return mean_value, max_value

    max_value = max(data)
    mean_value = sum(data)/len(data)
    
    return mean_value, max_value


'''
获取文件大小
'''
def get_file_size(file_path):
    if not file_path or not os.path.exists(file_path):
        return 0
    
    return os.path.getsize(file_path)

'''
将一个字典写入文件
'''
def write_json_to_file(data_dict, file_name, target_dir):
    with open(target_dir + '/' + file_name + '.json', 'w') as json_file:
        json_file.write(json.dumps(data_dict))

'''
对某个目录下的文件按照创建时间升序或者降序排序
dir_path: 文件夹
reverse: reverse = True 降序, reverse = False 升序（默认）
'''
def get_sort_file_list(dir_path, reverse):
    dir_list = os.listdir(dir_path)
    if not dir_list:
        return []
    
    dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=reverse)

    return dir_list

'''
按照进程启动命令形式停掉进程
'''
def stop_process_by_cmdline(cmdlines, not_to_kill_pid=-1):
    if not cmdlines:
        return
    
    # 按照进程启动命令形式, 采用遍历方式
    processes = psutil.process_iter()
    for process in processes:
        try:
            # 获取进程的命令行参数
            cmdline = process.cmdline()

            # 判断进程是否为目标进程
            if cmdlines in cmdline and process.pid != not_to_kill_pid:
                # 找到目标进程，杀死进程
                os.kill(process.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略异常进程
            pass

'''
按照进程名字停掉进程
'''
def stop_process_by_name(process_name, not_to_kill_pid=-1):
    if not process_name:
        return
    
    # 根据进程名获取进程ID, 采用遍历方式
    pids = psutil.process_iter()
    for pid in pids:
        if pid.name() == process_name and pid.pid != not_to_kill_pid:
            try:
                os.kill(pid.pid, signal.SIGKILL)
            except OSError as e:
                print(f'process_name {process_name} pid {pid} not exist')


'''
按照进程启动命令形式停掉进程
'''
def stop_process_by_cmdline(cmdlines, not_to_kill_pid=-1):
    if not cmdlines:
        return
    
    # 按照进程启动命令形式, 采用遍历方式
    processes = psutil.process_iter()
    for process in processes:
        try:
            # 获取进程的命令行参数
            cmdline = process.cmdline()

            # 判断进程是否为目标进程
            if cmdlines in cmdline and process.pid != not_to_kill_pid:
                # 找到目标进程，杀死进程
                os.kill(process.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略异常进程
            pass


'''
按照进程ID停掉进程
'''
def stop_process_by_pid(pid_list):
    if not pid_list or not len(pid_list):
        return
    
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            print(f' pid {pid} not exist')

'''
替换文件内容, 步骤:
1. 读取文件内容
2. 替换文本
3. 写回文件
'''
def replace_file_content(file, old_content, new_content):
    if not file:
        return
    
    # 读文件内容
    def read_file(file):
        with open(file, 'r', encoding='UTF-8') as f:
            read_all = f.read()

        return read_all

    # 写内容到文件
    def rewrite_file(file, data):
        with open(file, 'w', encoding='UTF-8') as f:
            f.write(data)
    
    content = read_file(file)
    content = content.replace(old_content, new_content)
    rewrite_file(file, content)

'''
随机生成某个区间的值
'''
def get_random(start_idx, end_idx):
    return random.randint(start_idx, end_idx)
