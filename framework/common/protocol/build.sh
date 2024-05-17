#!/bin/bash

# protobuf根据协议生成python/cpp文件

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh build.sh python|cpp \033[0m"
    exit -1
fi

language=$1

# 按照需要删除文件
for file in `ls `
do
    if [ $language == "python" ];
    then
        if [[ $file =~ \.o$ ]] || [[ $file =~ \.py$ ]];
        then
            rm -rf $file
        fi
    elif [ $language == "cpp" ];
    then
        if [[ $file =~ \.h$ ]] || [[ $file =~ \.cc$ ]] || [[ $file =~ \.o$ ]];
        then
            rm -rf $file
        fi
    else
        echo -e "\033[31m useage: sh build.sh python|cpp \033[0m"
        exit -1
    fi
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/protobuf/lib:/usr/local/lib

if [ $language == "python" ];
then
    protoc *.proto --python_out=.
elif [ $language == "cpp" ];
then
    protoc *.proto --cpp_out=.
else
    echo -e "\033[31m useage: sh build.sh python|cpp \033[0m"
    exit -1
fi

echo -e "\033[32m build.sh $language success \033[0m"
