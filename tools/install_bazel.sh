#!/bin/bash


# 容器环境里安装使用bazel

chmod +x tools/common.sh
. tools/common.sh

# 安装java相关
sudo yum install java-11-openjdk -y
sudo yum install java-11-openjdk-devel -y

# 下载bazel
wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
mv vbatts-bazel-epel-7.repo /etc/yum.repos.d

# 安装bazel
yum install bazel3 -y

# 使用, 带上/usr/bin前缀
/usr/bin/bazel version