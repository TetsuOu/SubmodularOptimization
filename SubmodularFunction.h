#ifndef SUBMODULARFUNCTION_H
#define SUBMODULARFUNCTION_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>

#include "DataTypeHandling.h"

using namespace std;

/*
* ÿ����ģ����Ӧ��ʵ�ֵĽӿ��ࡣ���е��Ż�������Ҫ��
* ����ӿ��ṩ��һ����ݵķ�ʽ��ʵ����״̬�Ĵ�ģ������ÿ����ģ�����������ṩ�ĸ�������
* operater(),peek,update,clone��SubmodularOptimizers���ۺ�ʱ����һ������ֵʱ����'peek'
* ���������ۺ�ʱһ����Ԫ��������⼯ʱ����'update'������
*/
class SubmodularFunction {
public:
    
    virtual data_t operator()(vector<vector<data_t>> const& cur_solution) const = 0;

     /*
     * @brief ���x���ڵ�ǰ���λ��pos����ӽ���Ļ������غ���ֵ��
     *        ���pos�ȵ�ǰ���е�Ԫ�ظ�����Ļ�����x��ӽ���ǰ�⡣������x�滻pos����Ԫ�ء�    
     * @param cur_solution: ��ǰ��
     * @param x: ����Ҫ��ӽ����Ԫ��
     * @param pos: Ҫ��ӽ�x��λ�á�0 <= pos < K
     * @retval ��x��ӽ���ǰ���pos���ĺ���ֵ
     */
    virtual data_t peek(vector<vector<data_t>> const& cur_solution,
        vector<data_t> const& x, unsigned int pos) = 0;

    /**
     * @brief  �����"pos"������x����ǰ��Ļ����º��������pos�ȵ�ǰ���е�Ԫ�ش�Ļ�����x��ӽ�
     *         ��ǰ�⡣���򣬽�pos���Ľ���x�����滻��
     * @note
     * @param  cur_solution: ��ǰ��.
     * @param  x: ��Ҫ��ӽ��⼯��Ԫ��.
     * @param  pos: ���x��λ�ã�ע�⵽��ΧΪ 0 <= pos < K
     * @retval None
     */
    virtual void update(vector<vector<data_t>> const& cur_solution,
        vector<data_t> const& x, unsigned int pos) = 0;

    /**
     * @brief  This function returns a clone of this Submodular function.
               Make sure, that the new objet is a valid clone which behaves like a
               new object and does not reference any members of this object. Some
               algorithms like SieveStreaming(++) or Salsa utilize multiple optimizers
               in parallel each with their own unique SubmodularFunction. Moreover,
               to make for efficient PyBind bindings, we use clone() to give the C++
               side more control over the memory.
     * @note
     * @retval
     */
    virtual shared_ptr<SubmodularFunction> clone() const = 0;

    /**
     * @brief  Destroys this object
     * @note
     * @retval None
     */
    virtual ~SubmodularFunction() {}
};

/**
 * @brief  A simple Wrapper class which takse a std::function and uses it to implement
        the SubmodularFunction interface. This is used as a convience class for the
        SubmodularOptimizer interface. This wrapper is meant for stateless functions
        so that the std::function __should not__ have / change / maintain an internal
        state which depends on the order of function calls. The main reason for this
        is, that the given std::function is likely to be moved into the member object
        (and not copied) which makes for very efficient code. However, some optimizers
        require multiple copies of the same function such as SieveStreaming(++) for
        multiple sub-optimizers. In this case, _all_ (sub-) optimizers reference the
        same object, which works fine if the function is stateless but probably breaks
        for stateful functions. If your submodular function requires some internal
        states which e.g. depend on the order of items added please consider to
        implement a `proper' SubmodularFunction.
 * @note
 * @retval None
 */
class SubmodularFunctionWrapper : public SubmodularFunction {
protected:
    // The std::function which implements the actual submodular function
    function<data_t(vector<vector<data_t>> const&)> f;

public:

    /**
     * @brief  Creates a new SubmodularFunction from a given std::function object.
     * @note
     * @param  f: The (stateless) function which implements the actual submodular
            function
     * @retval
     */
    SubmodularFunctionWrapper(function<data_t(vector<vector<data_t>>
        const&)> f) : f(f) {}

    /**
     * @brief  Implements the () operator by simply delegating the call to the
            underlying std::function.
     * @note
     * @param  &cur_solution:
     * @retval
     */
    data_t operator()(vector<vector<data_t>> const& cur_solution) const {
        return f(cur_solution);
    }

    /**
     * @brief  Implements the peek method. This copies the current solution vector to
            a new one, adds x at the appropriate positon and calls the ()-operator.
            In most cases the copy is probably not necessary (e.g. if we only append
            x to the current solution) which makes this code slightly inefficient.
     * @note
     * @param  &cur_solution:
     * @param  &x:
     * @param  pos:
     * @retval
     */
    data_t peek(vector<vector<data_t>> const& cur_solution,
        vector<data_t> const& x, unsigned int pos) {
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

    /**
     * @brief  Implements the update method. This class only wraps an std::function
            so it is state-less and the std::function would have to deal with any
            stateful behaviour. Thus, we don't do anything here.
     * @note
     * @param  &cur_solution:
     * @param  &x:
     * @param  pos:
     * @retval None
     */
    void update(vector<vector<data_t>> const& cur_solution,
        vector<data_t> const& x, unsigned int pos) {}

    /**
     * @brief  Implements the clone method. Note, that it is very likely that the
            std::function `f' has been moved into this object and similarly, we
            will move it into the clone as-well. This is okay, as long as `f' is a
            stateless function. However, if `f' has some internal state, then the
            other optimizers will use the __same__ function with the shared statte
            which will probably lead to weird side-effects. In this case consider
            implementing a proper SubmodularFunction.
     * @note
     * @retval
     */
    shared_ptr<SubmodularFunction> clone() const {
        return shared_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f));
    }

    ~SubmodularFunctionWrapper() {}
};

#endif // SUBMODULARFUNCTION_H

