#ifndef GREEDY_H
#define GREEDY_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;

/**
 * @brief  The Greedy optimizer for submodular functions. It rates the marginal gain of each element and picks that element with the largest gain. This process is repeated until it K elements have been selected:
 *  - Stream:  No
 *  - Solution: 1 - exp(1)
 *  - Runtime: O(N * K)
 *  - Memory: O(K)
 *  - Function Queries per Element: O(1)
 *  - Function Types: nonnegative submodular functions
 *
 * See also :
 *   - Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions-I. Mathematical Programming, 14(1), 265–294. https://doi.org/10.1007/BF01588971
 * @note
 */
class Greedy : public SubmodularOptimizer {
public:

    /**
     * @brief Construct a new Greedy object
     *
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.
     */
    Greedy(unsigned int K, SubmodularFunction& f) : SubmodularOptimizer(K, f) {}


    /**
     * @brief Construct a new Greedy object
     *
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     */
    Greedy(unsigned int K, function<data_t(vector<vector<data_t>> const&)> f) : SubmodularOptimizer(K, f) {}


    /*
    * @brief 在整个数据集中挑选拥有最大边际增益的元素。一直重复直到选择完K个元素。
    *        调用'get_solution'得到结果。
    * @param X 整个数据集的常引用。
    * @param iterations：其实没什么用，贪心算法在任何情况下在整个数据集上迭代K次。
    */
    void fit(vector<vector<data_t>> const& X, vector<idx_t> const& ids,
        unsigned int iterations = 1) {
       
        vector<unsigned int> remaining(X.size());//数据集中剩下未被选择的元素序号
        iota(remaining.begin(), remaining.end(), 0);//0,1,2，...,X.size()-1
        data_t fcur = 0;

        
        while (solution.size() < K && remaining.size() > 0) {//K个元素未选择完、数据集中还剩余有元素
            vector<data_t> fvals;//每个剩余未被选择元素被添加进解后的函数值，每次选取一个最大值
            fvals.reserve(remaining.size());//共remaining.size()个，每次循环值会减小

            /*
            * 贪心算法挑选具有最大边际增益的元素。等价于挑选导致最大函数值的元素。
            * 没必要再显式地再计算这个增长值。
            * 由于pos = solution.size()，即pos>=solution.size()，
            * 故ftmp就是假设将X[i]添加至当前解后的函数值。保存至数组fvals中。
            */
            for (auto i : remaining) {
                data_t ftmp = f->peek(solution, X[i], solution.size());
                fvals.push_back(ftmp);
            }

            /*
            * max_ele为fvals中拥有最大函数值的元素的下标，范围为[0,remaining.size()-1]，（remaining.size()==fvals.size()）
            * fcur为当前的最大函数值，这个函数值导致了最大边际增益，即fcur-fcur(上一轮)=最大边际增益
            * max_idx为X中的元素序号，范围为[0,X.size()-1]
            */
            
            unsigned int max_ele = distance(fvals.begin(), max_element(fvals.begin(), fvals.end()));
            fcur = fvals[max_ele];
            unsigned int max_idx = remaining[max_ele];

            /*
            * 得出了最大值，贪心选择该元素。故将序号为max_idx的元素添加进当前解，
            * 同时，次模函数进行更新。
            */
            f->update(solution, X[max_idx], solution.size());
            solution.push_back(X[max_idx]);

            /*
            * this->ids序列保存每一次被选择的元素序号
            */
            if (ids.size() >= max_idx) {
                this->ids.push_back(max_idx);
            }
            /*
            * 挑选完一个元素，将其从remaining中移去
            */
            remaining.erase(remaining.begin() + max_ele);

        }

        fval = fcur;//最后一步的函数值
        is_fitted = true;
    }

    void fit(vector<vector<data_t>> const& X, unsigned int iterations = 1) {
        vector<idx_t> ids;
        fit(X, ids, iterations);
    }


    /**
     * @brief Throws an exception when called. Greedy does not support streaming!
     *
     * @param x A constant reference to the next object on the stream.
     */
     /*
     * 调用时抛出异常。因为这里贪心算法不支持流数据。
     */
    void next(vector<data_t> const& x, optional<idx_t> id = nullopt) {
        throw runtime_error("Greedy does not support streaming data, please use fit().");
    }

};

#endif // GREEDY_H
