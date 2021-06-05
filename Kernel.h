#ifndef KERNEL_H
#define KERNEL_H

#include <cassert>
#include "DataTypeHandling.h"

using namespace std;

class Kernel {//ºËº¯Êý»ùÀà

public:
    virtual inline data_t operator()(const vector<data_t>& x1, const vector<data_t>& x2) const = 0;

    virtual shared_ptr<Kernel> clone() const = 0;

    virtual ~Kernel() {}
};


class KernelWrapper : public Kernel {
protected:
    function<data_t(vector<data_t> const&, vector<data_t> const&)> f;

public:

    KernelWrapper(function<data_t(vector<data_t> const&, vector<data_t> const&)> f) : f(f) {}

    inline data_t operator()(const vector<data_t>& x1, const vector<data_t>& x2) const override {
        return f(x1, x2);
    }

    shared_ptr<Kernel> clone() const override {
        return shared_ptr<Kernel>(new KernelWrapper(f));
    }

};

#endif // RBF_KERNEL_H
