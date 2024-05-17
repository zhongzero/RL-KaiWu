# 下面是安装环境的具体步骤, 在CPU和GPU机器上验证可以使用的

# ------------------- 下面是采用docker容器方法, 建议 -------------------------------
# 步骤1, 拉取镜像, 使用方法useage: sh deploy.sh dev|cpu|gpu
# 注意镜像的生成日期
sh tools/deploy.sh 

# 步骤2, 进入各个容器启动进程, useage: sh start.sh all|actor|learner|aisrv
# 进程启动前请注意conf/framework下的配置文件, 主要是configure.toml, aisrv.toml, learner.toml, actor.toml里参数配置
# app conf和algo conf需要按照实际情况配置
sh tools/start.sh 

# 步骤3, 确认进程启动成功, 启动自带的client, 注意参数需要正确 python app/sgame/tools/start_multi_game.py 需要运行的轮数 配置文件
python3 app/sgame/tools/start_multi_game.py 1 /data/projects/kaiwu-fwk/conf/framework/client.toml

# 步骤4, 确认进程运行正常

# 步骤5, 接入业务的battlesrv, 注意配置文件里的内容

# ------------------- 下面是采用conda方法, 不建议 -------------------------------

# 步骤1, 安装anaconda, 解决版本冲突问题
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
chmod +x Anaconda3-5.3.0-Linux-x86_64.sh
./Anaconda3-5.3.0-Linux-x86_64.sh
# 安装Anaconda时, 按照屏幕提示进行操作, 比如按Enter键, 输入yes等, 注意安装完成后, 需要打开新的窗口才生效, 执行下面操作生成环境, 注意按照屏幕提示操作
conda -V
conda create -n kaiwu python=3.7 
conda activate kaiwu

# 步骤2, 拉取KaiWuDRL代码, 输入用户名和密码
cd /data/
git clone https://git.woa.com/king-kaiwu/kaiwu-fwk.git
git checkout college

<<'COMMENT' 步骤3, 安装KaiWuDRL环境, 注意看屏幕输出, 如果出现版本找不到, 可以按照需要调整, 选择提供版本中最新的进行安装, 
特别注意:
1.reverb, https://github.com/deepmind/reverb上的安装方法:pip install dm-reverb[tensorflow]
2.horovod, 需要gcc 版本7.3.0, 需要升级gcc版本
COMMENT

cd /data/kaiwu-fwk
pip uninstall -y dist/*.whl
sh build_wheel.sh debug
pip install dist/*.whl

#该步骤请检查GPU机器上的cuda, cuDNN, Driver Version, tensorflow, reverb, horovod的版本是否匹配, 参见：https://www.tensorflow.org/install/source#tested_build_configurations

# 步骤3, 启动第三方组件modepool
cd thirdparty/model_pool_go/op
start start.sh gpu

# 步骤4, 启动进程验证, 注意配置文件的地址

<<'COMMENT'
下面是可能遇见的问题解决方法:
1. ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
解决方法: find /root -name libpython3.7m.so.1.0找到对应的目录文件, cp到/usr/lib/和/usr/lib64/里即可

2. libzmq.so.5: cannot open shared object file: No such file or directory
解决方法: find / -name libzmq.so.5找到该路径, 然后在vim  /etc/ld.so.conf 增加项, 再执行ldconfig

3. 特征值抽取的interface.so相关问题
解决方法: 需要到项目根目录执行下export PYTHONPATH=`pwd`, 
ImportError: libboost_python3.so.1.53.0, 需要执行yum install boost-devel, 注意版本匹配情况

4. 找不到libzmqop的问题
解决方法: 需要重新编译libzmqop, 注意pybind11, python的配置正确才行, 再export PYTHONPATH

5. horovod报与tensorflow的版本不匹配问题
解决方法: pip install --no-cache-dir horovod具体版本, 强制重新编译安装

COMMENT
