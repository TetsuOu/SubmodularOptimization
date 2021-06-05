#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string> 
#include <cmath>  

#include "DataTypeHandling.h"

using namespace std;

class Matrix {//�����ǶԳ���������
private:
    unsigned int N;//��/�еĴ�С
    // ʹ��std::vector���ԭ��ָ����Ҫ����������ԭ��
    // ��1��std::vector�����ִ�C++���ԭ��ָ���е㱻����
    // ��2��ԭ��ָ��ʹ��ʵ�ֺ��ʵĸ���/�ƶ����캯��ʮ������
    vector<data_t> data;//һά�����洢����

public:

    /*
    * ���캯��1�����ƾ���other�����Ͻ�N_sub*N_sub���Ӿ������¾���
    * �豣֤N_sub<=other.size()��
    */
    Matrix(Matrix const& other, unsigned int N_sub) : N(N_sub), data(N_sub* N_sub) {
        for (unsigned int i = 0; i < N_sub; ++i) {
            for (unsigned int j = 0; j < N_sub; ++j) {
                this->operator()(i, j) = other(i, j);
            }
        }
    }
    /*
    * ���캯��2��Ĭ��Ϊ_size��_size�е������
    */
    Matrix(unsigned int _size) : N(_size), data(_size* _size, 0) {}

    ~Matrix() { }

    /*
    * ������/�еĴ�С
    */
    inline unsigned int size() const { return N; }

    /*
    * ����������������*x�滻����ԭ��row�е����ݡ�
    * ��ʵ����ȴ���滻�˵�row�е����ݡ�
    * x��ָ������Ϊdata_t�ĳ������ĳ�ָ�롣
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
    * ��һУ����
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
    * ���������[]
    */
    data_t& operator [](int i) { return  data[i * N]; }
    data_t operator [](int i) const { return data[i * N]; }

    /*
    * ���������()
    */
    data_t& operator()(int i, int j) { return data[i * N + j]; }
    data_t operator()(int i, int j) const { return data[i * N + j]; }
};

/*
* ���غ���to_string��
* �ַ�ʽ����mat�����Ͻ�N_sub*N_sub��С���Ӿ���
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
* ���غ���to_string��
* �ַ�ʽ���ؾ���mat��
*/
inline string to_string(Matrix const& mat) {
    return to_string(mat, mat.size());
}

/*
* cholesky�ֽ⣺A = L*L^T��
* ���ﷵ�ص�L������Ӧ���������Ǿ���������Ԫ�ض�Ϊ0��
* ����ʵ���Ϻ���ļ���ֻ��Ҫ�õ��Խ��ߵ�Ԫ����������ʽ��
* ���������Lʵ���ϵ�������Ԫ�ز�����Ϊ0������ԭ����ֵ
*����������Ԫ��δ���д���ֻ������������Ԫ�ؼ��Խ�Ԫ�أ�
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
* A=L*L^T������˻�������ʽ���ھ�������ʽ�ĳ˻���
* �� |A|=|L|*|L^T|������L�������������Ǿ��󣬼� |L| = L(0,0)*L(1,1)*...*L(n-1,n-1)��
* �ָ��ݶ������������log(|L|) = log(L(0,0))+...+log(L(n-1,n-1))��
* ��L��L^T�ĶԽ���Ԫ����ͬ����ôlog(|A|)=2*log(|L|)
*/
inline data_t log_det_from_cholesky(Matrix const& L) {
    data_t det = 0;

    for (size_t i = 0; i < L.size(); ++i) {
        det += log(L(i, i));
    }

    return 2 * det;
}
/*
* �������mat���Ͻ�N_sub*N_sub��С���Ӿ���Ķ�������ʽ
*/
inline data_t log_det(Matrix const& mat, unsigned int N_sub) {
    Matrix L = cholesky(mat, N_sub);
    return log_det_from_cholesky(L);
}

inline data_t log_det(Matrix const& mat) {
    return log_det(mat, mat.size());
}

#endif