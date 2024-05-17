#!/usr/bin/env bash

# pybind11, 主要是生成libzmqop

g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/local/python-3.7/include/python3.7m/ \
   -I/usr/local/python-3.7/lib/python3.7/site-packages/pybind11/include/ -I/usr/local/python-3.7/lib/python3.7/site-packages/tensorflow/include/ \
   libzmqop.cc -o libzmqop`python3-config --extension-suffix`



