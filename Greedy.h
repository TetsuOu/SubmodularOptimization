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
 *   - Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions-I. Mathematical Programming, 14(1), 265�C294. https://doi.org/10.1007/BF01588971
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
    * @brief ���������ݼ�����ѡӵ�����߼������Ԫ�ء�һֱ�ظ�ֱ��ѡ����K��Ԫ�ء�
    *        ����'get_solution'�õ������
    * @param X �������ݼ��ĳ����á�
    * @param iterations����ʵûʲô�ã�̰���㷨���κ���������������ݼ��ϵ���K�Ρ�
    */
    void fit(vector<vector<data_t>> const& X, vector<idx_t> const& ids,
        unsigned int iterations = 1) {
       
        vector<unsigned int> remaining(X.size());//���ݼ���ʣ��δ��ѡ���Ԫ�����
        iota(remaining.begin(), remaining.end(), 0);//0,1,2��...,X.size()-1
        data_t fcur = 0;

        
        while (solution.size() < K && remaining.size() > 0) {//K��Ԫ��δѡ���ꡢ���ݼ��л�ʣ����Ԫ��
            vector<data_t> fvals;//ÿ��ʣ��δ��ѡ��Ԫ�ر���ӽ����ĺ���ֵ��ÿ��ѡȡһ�����ֵ
            fvals.reserve(remaining.size());//��remaining.size()����ÿ��ѭ��ֵ���С

            /*
            * ̰���㷨��ѡ�������߼������Ԫ�ء��ȼ�����ѡ���������ֵ��Ԫ�ء�
            * û��Ҫ����ʽ���ټ����������ֵ��
            * ����pos = solution.size()����pos>=solution.size()��
            * ��ftmp���Ǽ��轫X[i]�������ǰ���ĺ���ֵ������������fvals�С�
            */
            for (auto i : remaining) {
                data_t ftmp = f->peek(solution, X[i], solution.size());
                fvals.push_back(ftmp);
            }

            /*
            * max_eleΪfvals��ӵ�������ֵ��Ԫ�ص��±꣬��ΧΪ[0,remaining.size()-1]����remaining.size()==fvals.size()��
            * fcurΪ��ǰ�������ֵ���������ֵ���������߼����棬��fcur-fcur(��һ��)=���߼�����
            * max_idxΪX�е�Ԫ����ţ���ΧΪ[0,X.size()-1]
            */
            
            unsigned int max_ele = distance(fvals.begin(), max_element(fvals.begin(), fvals.end()));
            fcur = fvals[max_ele];
            unsigned int max_idx = remaining[max_ele];

            /*
            * �ó������ֵ��̰��ѡ���Ԫ�ء��ʽ����Ϊmax_idx��Ԫ����ӽ���ǰ�⣬
            * ͬʱ����ģ�������и��¡�
            */
            f->update(solution, X[max_idx], solution.size());
            solution.push_back(X[max_idx]);

            /*
            * this->ids���б���ÿһ�α�ѡ���Ԫ�����
            */
            if (ids.size() >= max_idx) {
                this->ids.push_back(max_idx);
            }
            /*
            * ��ѡ��һ��Ԫ�أ������remaining����ȥ
            */
            remaining.erase(remaining.begin() + max_ele);

        }

        fval = fcur;//���һ���ĺ���ֵ
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
     * ����ʱ�׳��쳣����Ϊ����̰���㷨��֧�������ݡ�
     */
    void next(vector<data_t> const& x, optional<idx_t> id = nullopt) {
        throw runtime_error("Greedy does not support streaming data, please use fit().");
    }

};

#endif // GREEDY_H
