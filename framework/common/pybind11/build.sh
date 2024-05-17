
#!/usr/bin/env bash

# pybind11
g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 `python3 -m pybind11 --includes` kaiwu.cc -o kaiwu`python3-config --extension-suffix`

