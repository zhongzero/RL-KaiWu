#include <pybind11/options.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int Add(int i, int j)
{
   return i + j;
}

int Max(int i, int j)
{
    if(i > j)
    {
        return i;
    }
    else
    {
        return j;
    }
}

PYBIND11_MODULE(kaiwu_test, m) {
    m.doc() = "kaiwu pybind11  pybind11";
    m.def("Add", &Add, "A function which adds two numbers");
    m.def("Max", &Max, "A function which max two numbers");
}
