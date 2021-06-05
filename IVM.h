#ifndef INFORMATIVE_VECTOR_MACHINE_H
#define INFORMATIVE_VECTOR_MACHINE_H

#include <mutex>
#include <vector>
#include <functional>
#include <math.h>
#include <cassert>
#include "DataTypeHandling.h"
#include "SubmodularFunction.h"
#include "Kernel.h"
#include "Matrix.h"

using namespace std;

/*
* IVM��֤���ǵ�����ģ�������̳�SubmodularFunction�ࡣ
*/
class IVM : public SubmodularFunction {
protected:
    /*
    * ����˾���
    * X�Ǵ���Ĵ𰸼���ͨ��kernel->operator()����Ԫ�ؼ�����ƶȣ����ת��Ϊ�˾��󣨶Գ�������
    * pow(1.0,2.0)=1�������趨�Ĳ�����1��
    * ���ﻹ����һ����������������һ����λ���󡣣�i==jʱ����˸�1��
    */
    inline Matrix compute_kernel(vector<vector<data_t>> const& X) const {
        unsigned int K = X.size();
        Matrix mat(K);

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel->operator()(X[i], X[j]);
                if (i == j) {
                    mat(i, j) = 1.0 + kval / pow(1.0, 2.0);
                }
                else {
                    mat(i, j) = kval / pow(1.0, 2.0);
                    mat(j, i) = kval / pow(1.0, 2.0);
                }
            }
        }

        // TODO CHECK IF THIS USES MOVE
        return mat;
    }

    shared_ptr<Kernel> kernel;//�����õĵĺ˺���
    data_t sigma;

public:
    IVM(Kernel const& kernel, data_t sigma) : kernel(kernel.clone()), sigma(sigma) {}

    IVM(function<data_t(vector<data_t> const&, vector<data_t> const&)> kernel, data_t sigma)
        : kernel(unique_ptr<Kernel>(new KernelWrapper(kernel))), sigma(sigma) {
    }

    data_t peek(vector<vector<data_t>> const& cur_solution, vector<data_t> const& x, unsigned int pos) override {
        vector<vector<data_t>> tmp(cur_solution);

        if (pos >= cur_solution.size()) {
            tmp.push_back(x);
        }
        else {
            tmp[pos] = x;
        }

        data_t ftmp = this->operator()(tmp);
        return ftmp;
    }

    void update(vector<vector<data_t>> const& cur_solution, vector<data_t> const& x, unsigned int pos) override {}

    data_t operator()(vector<vector<data_t>> const& X) const override {
        // This is the most basic implementations which recomputes everything with each call
        // I would not use this for any real-world problems. 

        Matrix kernel_mat = compute_kernel(X);
        return log_det(kernel_mat);
    }

    shared_ptr<SubmodularFunction> clone() const override {
        return make_shared<IVM>(*kernel, sigma);
    }

    ~IVM() {}
};

#endif // INFORMATIVE_VECTOR_MACHINE_H

