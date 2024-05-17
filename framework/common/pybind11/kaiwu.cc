#include <pybind11/options.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "kaiwu.h"

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

int g_head_bufflen = 8200;
char g_head_buff [8200] = {0};

int Pack(int idx, int magic, int data_len, std::string &data)
{
    if(0 >= data_len)
    {
        return -1;
    }

    char* remain_buff = g_head_buff;
    int remain_bufflen = g_head_bufflen;

    memcpy(remain_buff, &magic, sizeof(magic));
    remain_buff += sizeof(magic);
    remain_bufflen -= sizeof(magic);

    memcpy(remain_buff, &data_len, sizeof(data_len));
    remain_buff += sizeof(data_len);
    remain_bufflen -= sizeof(data_len);

    if(data_len > remain_bufflen)
    {
        printf("data_len %d > remain_bufflen %d", data_len, remain_bufflen);
        return -2;
    }

    memcpy(remain_buff, &data, data_len);
    remain_buff += data_len;
    remain_bufflen -= data_len;

    //printf("remain_buff: %s", g_head_buff);

    return 0;
}

int UnPack()
{
    const uint8_t* remain_buff = reinterpret_cast<const uint8_t*>(g_head_buff);
    int32_t remain_bufflen = g_head_bufflen;

    const int32_t* magic = reinterpret_cast<const int32_t*>(remain_buff);
    remain_buff += sizeof(int32_t);
    remain_bufflen -= sizeof(int32_t);
    //printf("magic: %x \n", *magic);

    const int32_t* data_len = reinterpret_cast<const int32_t*>(remain_buff);
    remain_buff += sizeof(int32_t);
    remain_bufflen -= sizeof(int32_t);
    //printf("data_len: %d \n", *data_len);

    std::string data = reinterpret_cast<const char*>(remain_buff);
    //printf("data: %x \n", data.c_str());

    return 0;
}

namespace kaiwu
{
    KaiWu::KaiWu(const std::string &name)
    {
        m_name = name;
    }

   void KaiWu::setName(const std::string &name)
    {
        m_name = name;
    }
}

int list2Vector(std::vector<float> &vec, int batchSize)
{
    printf("checpoint1");

    return 0;
}

int numpy2Array(py::array_t<float> &input, int batchSize)
{
    printf("checpoint2");
    
    return 0;
}

PYBIND11_MODULE(kaiwu, m) {
    pybind11::class_<kaiwu::KaiWu>(m, "KaiWu")
        .def(pybind11::init<const std::string &>())
        .def("setName", &kaiwu::KaiWu::setName)
        .def("getName", &kaiwu::KaiWu::getName);

    m.doc() = "kaiwu pybind11  pybind11";
    m.def("Add", &Add, "A function which adds two numbers");
    m.def("Pack", &Pack, "A function which Pack Data");
    m.def("UnPack", &UnPack, "A function which UnPack Data");
    m.def("list2Vector", &list2Vector, "A function which list2Vector");
    m.def("numpy2Array", &numpy2Array, "A function which numpy2Array");

}
