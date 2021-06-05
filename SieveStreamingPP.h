#ifndef SIEVESTREAMINGPP_H
#define SIEVESTREAMINGPP_H

#include "DataTypeHandling.h"
#include "SieveStreaming.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

/**
 * @brief The SieveStreaming optimizer for nonnegative, monotone submodular functions. This is an improved version of SieveStreaming which re-samples thresholds once a new (better) lower bound is detected.
 *  - Stream:  Yes
 *  - Solution: 1/2 - \varepsilon
 *  - Runtime: O(1)
 *  - Memory: O(K / \varepsilon)
 *  - Function Queries per Element: O(log(K) / \varepsilon)
 *  - Function Types: nonnegative, monotone submodular functions
 *
 * See also:
 *   - Kazemi, E., Mitrovic, M., Zadimoghaddam, M., Lattanzi, S., & Karbasi, A. (2019). Submodular streaming in all its glory: Tight approximation, minimum memory and low adaptive complexity. 36th International Conference on Machine Learning, ICML 2019, 2019-June, 5767�C5784. Retrieved from http://proceedings.mlr.press/v97/kazemi19a/kazemi19a.pdf
*/
class SieveStreamingPP : public SubmodularOptimizer {
private:

    class Sieve : public SubmodularOptimizer {
    public:
        // ��ֵ
        data_t threshold;

        /**
         * @brief ����һ���µ�Sieve����
         *
         * @param K ��ģ�Ż�����Ļ���Լ����ҪѡȡԪ�صĸ���
         * @param f Ӧ�ñ���󻯵Ĵ�ģ������ע�⵽��'clone'��������������һ���µ�SubmodularFunction����
                    �������������ӵ�С����Ҫʵ��һ�����Ƶ�SubmodularFunction��Ҫȷ��������Ҫ��ʵ����
                    ���ܱ���¡/����
         * @param threshold ��ֵ
         */
        Sieve(unsigned int K, SubmodularFunction& f, data_t threshold) : SubmodularOptimizer(K, f), threshold(threshold) {}

        /**
         * @brief ����һ���µ�Sieve����
         *
         * @param K ��ģ�Ż�����Ļ���Լ����ҪѡȡԪ�صĸ���
         * @param f Ӧ�ñ���󻯵Ĵ�ģ������ע�⵽��'clone'��������������һ���µ�SubmodularFunction����
                    �������������ӵ�С����Ҫʵ��һ�����Ƶ�SubmodularFunction��Ҫȷ��������Ҫ��ʵ����
                    ���ܱ���¡/����
         * @param threshold ��ֵ
         */
        Sieve(unsigned int K, function<data_t(vector<vector<data_t>> const&)> f, data_t threshold) : SubmodularOptimizer(K, f), threshold(threshold) {
        }

       
        /*
        * @brief �׳��쳣����Ϊfit()������Ӧ�������ﱻֱ�ӵ��á�Sieve�����������Լ�ʹ�õģ�����
                 ͨ��SieveStreamingpp����
        */
        void fit(vector<vector<data_t>> const& X, unsigned int iterations = 1) {
            throw runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
        }

       
        /*
        * @brief �����������е���һ�����ݡ����ｫ�߼���������ֵ���бȽϣ������������ֵ��
                 �ͽ���ǰԪ����ӵ���ǰ�⼯��
        * @param x �������е�ǰ���ݵ�һ��������
        */
        void next(vector<data_t> const& x, optional<idx_t> const id = nullopt) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {//�������Լ��
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;//�߼�����

                if (fdelta >= threshold) {
                    f->update(solution, x, solution.size());
                    solution.push_back(x);
                    if (id.has_value()) ids.push_back(id.value());
                    fval += fdelta;
                }
            }
            is_fitted = true;
        }
    };


    data_t lower_bound;
    data_t m;
    data_t epsilon;

public:
    vector<unique_ptr<Sieve>> sieves;

    SieveStreamingPP(unsigned int K, SubmodularFunction& f, data_t m, data_t epsilon)
        : SubmodularOptimizer(K, f), lower_bound(0), m(m), epsilon(epsilon) {
        // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

        // for (auto t : ts) {
        //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
        // }
    }

    SieveStreamingPP(unsigned int K, function<data_t(vector<vector<data_t>> const&)> f, data_t m, data_t epsilon)
        : SubmodularOptimizer(K, f), lower_bound(0), m(m), epsilon(epsilon) {
        // std::vector<data_t> ts = thresholds(m/(1.0 + epsilon), K * m, epsilon);

        // for (auto t : ts) {
        //     sieves.push_back(std::make_unique<Sieve>(K, *this->f, t));
        // }
    }

    unsigned int get_num_candidate_solutions() const {
        return sieves.size();
    }

    unsigned long get_num_elements_stored() const {
        unsigned long num_elements = 0;
        for (auto const& s : sieves) {
            num_elements += s->get_solution().size();
        }

        return num_elements;
    }

    void next(vector<data_t> const& x, optional<idx_t> const id = nullopt) {
        if (lower_bound != fval || sieves.size() == 0) {
            lower_bound = fval;
            data_t tau_min = max(lower_bound, m) / static_cast<data_t>(2.0 * K);//������С��ֵ
            auto no_sieves_before = sieves.size();

            auto res = remove_if(sieves.begin(), sieves.end(),
                [tau_min](auto const& s) { return s->threshold < tau_min; }//ɾ��С����С��ֵ��ɸ��
            );
            sieves.erase(res, sieves.end());

            if (no_sieves_before > sieves.size() || no_sieves_before == 0) {
                vector<data_t> ts = thresholds(tau_min / (1.0 + epsilon), K * m, epsilon);

                for (auto t : ts) {
                    bool any = any_of(sieves.begin(), sieves.end(),
                        [t](auto const& s) { return s->threshold == t; }
                    );
                    if (!any) {
                        sieves.push_back(make_unique<Sieve>(K, *f, t));
                    }
                }
            }
        }

        // std::cout << sieves.size() << std::endl;
        for (auto& s : sieves) {
            s->next(x, id);
            if (s->get_fval() > fval) {
                fval = s->get_fval();
                // TODO THIS IS A COPY AT THE MOMENT
                solution = s->solution;
                ids = s->ids;//������ţ��������Ǽ���
            }
        }
        is_fitted = true;
    };
};

#endif