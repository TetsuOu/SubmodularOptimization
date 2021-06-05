#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string> 
#include <cmath>  

#include "DataTypeHandling.h"

using namespace std;

class Matrix {//这里是对称正定矩阵
private:
    unsigned int N;//行/列的大小
    // 使用std::vector替代原生指针主要有以下两个原因
    // （1）std::vector更具现代C++风格，原生指针有点被遗弃
    // （2）原生指针使得实现合适的复制/移动构造函数十分困难
    vector<data_t> data;//一维向量存储矩阵

public:

    /*
    * 构造函数1，复制矩阵other的左上角N_sub*N_sub的子矩阵至新矩阵。
    * 需保证N_sub<=other.size()。
    */
    Matrix(Matrix const& other, unsigned int N_sub) : N(N_sub), data(N_sub* N_sub) {
        for (unsigned int i = 0; i < N_sub; ++i) {
            for (unsigned int j = 0; j < N_sub; ++j) {
                this->operator()(i, j) = other(i, j);
            }
        }
    }
    /*
    * 构造函数2，默认为_size行_size列的零矩阵
    */
    Matrix(unsigned int _size) : N(_size), data(_size* _size, 0) {}

    ~Matrix() { }

    /*
    * 返回行/列的大小
    */
    inline unsigned int size() const { return N; }

    /*
    * 函数命名来看是用*x替换矩阵原第row行的内容。
    * 但实际上却是替换了第row列的内容。
    * x是指向类型为data_t的常变量的常指针。
    */
    void replace_row(unsigned int row, data_t const* const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(i, row) = x[i];
        }
    }
     
    void replace_column(unsigned int col, data_t const* const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(col, i) = x[i];
        }
    }
    /*
    * 秩一校正。
    */
    void rank_one_update(unsigned int row, data_t const* const x) {
        for (unsigned int i = 0; i < N; ++i) {
            if (row == i) {
                this->operator()(i, i) += x[i];
            }
            else {
                this->operator()(i, row) += x[i];
                this->operator()(row, i) += x[i];
            }
        }
    }
    /*
    * 重载运算符[]
    */
    data_t& operator [](int i) { return  data[i * N]; }
    data_t operator [](int i) const { return data[i * N]; }

    /*
    * 重载运算符()
    */
    data_t& operator()(int i, int j) { return data[i * N + j]; }
    data_t operator()(int i, int j) const { return data[i * N + j]; }
};

/*
* 重载函数to_string。
* 字符式返回mat的左上角N_sub*N_sub大小的子矩阵。
*/
inline string to_string(Matrix const& mat, unsigned int N_sub) {
    string s = "[";

    for (unsigned int i = 0; i < N_sub; ++i) {
        s += "[";
        for (unsigned int j = 0; j < N_sub; ++j) {
            if (j < N_sub - 1) {
                s += to_string(mat(i, j)) + ",";
            }
            else {
                s += to_string(mat(i, j));
            }
        }

        if (i < N_sub - 1) {
            s += "],\n";
        }
        else {
            s += "]";
        }
    }

    return s + "]";
}
/*
* 重载函数to_string。
* 字符式返回矩阵mat。
*/
inline string to_string(Matrix const& mat) {
    return to_string(mat, mat.size());
}

/*
* cholesky分解：A = L*L^T。
* 这里返回的L理论上应该是下三角矩阵，上三角元素都为0，
* 但是实际上后面的计算只需要用到对角线的元素来求行列式，
* 所以这里的L实际上的上三角元素并不会为0，而是原来的值
*（即上三角元素未进行处理，只处理了下三角元素及对角元素）
*/
inline Matrix cholesky(Matrix const& in, unsigned int N_sub) {
    Matrix L(in, N_sub);

    for (unsigned int j = 0; j < N_sub; ++j) {
        data_t sum = 0.0;

        for (unsigned int k = 0; k < j; ++k) {
            sum += L(j, k) * L(j, k);
        }

        L(j, j) = sqrt(in(j, j) - sum);
        
        for (unsigned int i = j + 1; i < N_sub; ++i) {
            data_t sum = 0.0;

            for (unsigned int k = 0; k < j; ++k) {
                sum += L(i, k) * L(j, k);
            }
            L(i, j) = (in(i, j) - sum) / L(j, j);
            
        }
    }
    return L;
}

inline Matrix cholesky(Matrix const& in) { return cholesky(in, in.size()); }

/*
* A=L*L^T，矩阵乘积的行列式等于矩阵行列式的乘积。
* 故 |A|=|L|*|L^T|。由于L理论上是下三角矩阵，即 |L| = L(0,0)*L(1,1)*...*L(n-1,n-1)。
* 又根据对数运算规则，有log(|L|) = log(L(0,0))+...+log(L(n-1,n-1))。
* 又L和L^T的对角线元素相同，那么log(|A|)=2*log(|L|)
*/
inline data_t log_det_from_cholesky(Matrix const& L) {
    data_t det = 0;

    for (size_t i = 0; i < L.size(); ++i) {
        det += log(L(i, i));
    }

    return 2 * det;
}
/*
* 计算矩阵mat左上角N_sub*N_sub大小的子矩阵的对数行列式
*/
inline data_t log_det(Matrix const& mat, unsigned int N_sub) {
    Matrix L = cholesky(mat, N_sub);
    return log_det_from_cholesky(L);
}

inline data_t log_det(Matrix const& mat) {
    return log_det(mat, mat.size());
}

#endif