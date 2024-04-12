#include <pybind11/pybind11.h>

namespace py = pybind11;

struct A
{
    int value;
    A(int value) : value(value) {}
    int get() const { return value; }
    void set(int value_) { value = value_; }
};

int add(int i = 1, int j = 2)
{
    return i + j;
}

PYBIND11_MODULE(my_extension, m)
{
    m.attr("hello_world") = py::cast("Hello, World!");
    m.attr("answer") = 42;

    m.doc() = "pybind11 example plugin";   
    m.def("add", &add, "A function that adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2);

    py::class_<A>(m, "A")
        .def(py::init<int>())
        .def("get", &A::get)
        .def("set", &A::set);
}