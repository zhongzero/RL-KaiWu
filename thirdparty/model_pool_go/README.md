## model pool使用文档

### 部署与启动方式

#### server部署
该运行包可直接在开悟环境中使用，使用方法如下
- 直接将model_pool包放在GPU和CPU环境中
- CPU环境使用如下命令，启动运行
```shell
cd model_pool/op && bash start.sh actor
```
- GPU环境使用如下命令，启动运行
```shell
cd model_pool/op && bash start.sh learner
```

#### model_pool_api更新
需要将model_pool api切换到2.0版本
```
pip3 uninstall sail.model-pool
pip3 install --upgrade sail.model-pool==2.0 --index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --extra-index-url=https://mirrors.tencent.com/pypi/simple
```

### 注意事项
#### 实验启动
新版本的model_pool不需要CPU_model_pool的节点，因此实验启动过程中需要将CPU_model_pool的容器数目设置成0。

#### GPU
GPU段的配置文件为 model_pool/config/trpc_go.yaml.gpu,需要注意的配置项有以下两点
- 模型的存储路径，注意model_pool存储的模型会进行重命名，采取自定义的唯一标识fid作为文件名存储
- 模型存储的容量限制
```yaml
modelpool:
  role: master
  fileSavePath: files #模型存储路径，默认存储在model_pool/files目录下
  cluster: 
  ip: __MODELPOOL_IP_HERE__
  name: 
  maxStorage: 16GB #模型存储容量，单位可选 MB GB TB,业务需要根据环境磁盘容量进行配置
  statisticsBufferSize: 500
```

#### CPU
由于大多数业务的CPU容器使用的是CVM资源，采取的内存盘方案，因此对模型的存储容量以及模型的存储路径需要额外关注。
CPU段的model_pool默认会带一个负责进行解压的proxy，因此模型的存储路径会有两个
- modelpool从远端master分发过来的压缩的模型路径，修改model_pool/config/trpc_go.yaml.cpu
```yaml
modelpool:
  role: master
  fileSavePath: files #模型存储路径，默认存储在model_pool/files目录下
```
- modelpool_proxy解压的模型存储路径，修改model_pool/op/start.sh
```shell
## -fileSavePath=* 用于修改存储模型解压的路径
cd ../bin && nohup ./modelpool_proxy -fileSavePath=model > ../log/proxy.log 2>&1 &
```
模型的存储容量的设置是修改model_pool/config/trpc_go.yaml.cpu中的maxStorage字段
```yaml
modelpool:
  role: slave
  fileSavePath: ./files
  cluster: __MODELPOOL_CLUSTER_HERE__
  ip: __MODELPOOL_IP_HERE__
  name: __MODELPOOL_NAME_HERE__
  maxStorage: 1GB #单位可选 MB GB TB
  statisticsBufferSize: 0
```
新版本的model_pool自带解压功能，因此CPU镜像中可以将之前的解压和删除进程的启动去除