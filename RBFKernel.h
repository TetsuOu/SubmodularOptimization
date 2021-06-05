#ifndef RBF_KERNEL_H
#define RBF_KERNEL_H

#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include  <cmath>
#include <memory>

#include "DataTypeHandling.h"
#include "Kernel.h"

using namespace std;

class RBFKernel : public Kernel {
private:
    data_t sigma = 1.0;
    data_t scale = 1.0;

public:
    /*
    * Ĭ�Ϲ��캯����sigma��ֵΪ1.0�����뺯��Ϊƽ��ŷʽ���롣
    */
    RBFKernel() = default;

    /*
    * ���ݸ�����sigmaʵ����RBFKernel��
    */
    explicit RBFKernel(data_t sigma) : RBFKernel(sigma, 1.0) {
    }

    /**
     * Instantiates a RBF Kernel object with given sigma and a arbitrarily chosen 
        distance function.
     * @param sigma Kernel sigma value.
     * @param l A scaling value.
     */
    /*
    * ���ݸ�����sigma��scaleʵ����RBFKernel��
    */
    RBFKernel(data_t sigma, data_t scale) : sigma(sigma), scale(scale) {
        assert(("The scale of an RBF Kernel should be greater than 0!", scale > 0));
        assert(("The sigma value of an RBF Kernel should be greater than  0!", sigma > 0));
    };

    /*
    * ���������()�������º���������RBF�˺���ֵ
    * ԭע����Ĺ�ʽΪ k(x_1, x_2) = _l^2 \exp(- \frac{\|x_1 - x_2 \|_2^2}{2\sigma^2})
    * �����ݴ���ʵ������ӦΪ k(x_1,x_2)= scale * \exp(-\frac{||x_1-x_2||_2^2}{\sigma})
    * ��ʽ�еķ������ܻᱻ����һ�����뺯������ö���Ĳ�ͬ�������滻
    * Ȼ����Ĭ�ϵ�ʵ����ƽ����L2��������ʵ���������������������ƽ��ŷʽ����
    */
    inline data_t operator()(const vector<data_t>& x1, const vector<data_t>& x2) const override {
        data_t distance = 0;
        if (x1 != x2) {
            distance = inner_product(x1.begin(), x1.end(), x2.begin(), data_t(0),
                plus<data_t>(), [](data_t x, data_t y) {return (y - x) * (y - x); }
            );
            
            distance /= sigma;
        }
        return scale * exp(-distance);
    }

    shared_ptr<Kernel> clone() const override {
        return shared_ptr<Kernel>(new RBFKernel(sigma, scale));
    }
};

#endif // RBF_KERNEL_H
