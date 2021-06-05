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
    * 默认构造函数。sigma的值为1.0，距离函数为平方欧式距离。
    */
    RBFKernel() = default;

    /*
    * 根据给定的sigma实例化RBFKernel。
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
    * 根据给定的sigma和scale实例化RBFKernel。
    */
    RBFKernel(data_t sigma, data_t scale) : sigma(sigma), scale(scale) {
        assert(("The scale of an RBF Kernel should be greater than 0!", scale > 0));
        assert(("The sigma value of an RBF Kernel should be greater than  0!", sigma > 0));
    };

    /*
    * 重载运算符()，当作仿函数，返回RBF核函数值
    * 原注释里的公式为 k(x_1, x_2) = _l^2 \exp(- \frac{\|x_1 - x_2 \|_2^2}{2\sigma^2})
    * 但根据代码实际内容应为 k(x_1,x_2)= scale * \exp(-\frac{||x_1-x_2||_2^2}{\sigma})
    * 等式中的范数可能会被用另一个距离函数构造该对象的不同函数所替换
    * 然而，默认的实现是平方的L2范数，这实际上类似于输入向量间的平方欧式距离
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
