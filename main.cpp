#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <chrono>

#include "FastIVM.h"
#include "RBFKernel.h"
#include "Greedy.h"
#include "Random.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"

#include "DataTypeHandling.h"

using namespace std;

vector<vector<data_t>> read_arff(string const& path) {
    vector<vector<data_t>> X;

    string line;
    ifstream file(path);

    if (file.is_open()) {
        while (getline(file, line)) {
            // Skip every meta information
            if (line.size() > 0 && line[0] != '@' && line != "\r") {
                vector<data_t> x;
                stringstream ss(line);
                string entry;
                // All entries are float, but the last one (the label, string) and the second to last(the id, integer). Skip both.
                while (getline(ss, entry, ',') && x.size() < 41) {
                    if (entry.size() > 0) { //&& entry[0] != '\''
                        x.push_back(static_cast<float>(atof(entry.c_str())));
                    }
                }
                if (X.size() > 0 && x.size() != X[0].size()) {
                    cout << "Size mismatch detected. Ignoring line." << std::endl;
                }
                else {
                    X.push_back(x);
                }
            }
        }
        file.close();
    }

    return X;
}

auto evaluate_optimizer(SubmodularOptimizer& opt, vector<vector<data_t>>& X) {
    auto start = chrono::steady_clock::now();
    opt.fit(X);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> runtime_seconds = end - start;
    auto fval = opt.get_fval();
    cout << "Selected " << opt.get_solution().size() << endl;
    //最终的函数值；运行时间；存储元素个数；候选解集的个数
    return make_tuple(fval, runtime_seconds.count(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions());
}


//greedy
auto evaluate_optimizer_ids(SubmodularOptimizer& opt, vector<vector<data_t>>& X, vector<idx_t> ids) {
    auto start = chrono::steady_clock::now();
    opt.fit(X,ids);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> runtime_seconds = end - start;
    auto fval = opt.get_fval();
    cout << "Selected " << opt.get_solution().size() << endl;
    //最终的函数值；运行时间；存储元素个数；候选解集的个数
    return make_tuple(fval, runtime_seconds.count(), opt.get_num_elements_stored(), opt.get_num_candidate_solutions());
}


string to_string(vector<vector<data_t>> const& solution) {
    string s;

    for (auto& x : solution) {
        for (auto xi : x) {
            s += to_string(xi) + " ";
        }
        s += "\n";
    }

    return s;
}

int main() {
    cout << "Reading data" << endl;
    auto data = read_arff("./KDDCup99/KDDCup99_withoutdupl_norm_1ofn.arff");
    //https://www.kaggle.com/isaikumar/creditcardfraud
    //auto data = read_arff("./Creditcard/test_dim29.arff");
    cout << "dataset size: " << data.size() << "; dimensions: " << data[0].size() << endl;
   

    vector<idx_t> ids;
    ids.resize(data.size());
    iota(ids.begin(), ids.end(), 0);
    
    unsigned int K = 5;

    FastIVM fastIVM(K, RBFKernel(sqrt(data[0].size()), 1.0), 1.0);
    tuple<data_t, double, unsigned long, unsigned int> res;

    cout << "Selecting " << K << " representatives via fast IVM with Greedy" << endl;
    Greedy fastGreedy(K, fastIVM);
    res = evaluate_optimizer_ids(fastGreedy, data, ids);
    cout << "\t fval:\t\t" << get<0>(res) << "\n\t runtime:\t" << get<1>(res) << "s\n\t memory:\t" << get<2>(res) << "\n\t num_sieves:\t" << get<3>(res) << "\n\n" << endl;
    /*vector<idx_t> solution_ids = fastGreedy.get_ids();

    for (auto x : solution_ids) {
        cout << x << ' ';
    }
    cout << endl;*/

    iota(ids.begin(), ids.end(), 0);
    cout << "Selecting " << K << " representatives via Random with seed = 0" << endl;
    Random random0(K, fastIVM, 0);
    res = evaluate_optimizer_ids(random0, data, ids);
    cout << "\t fval:\t\t" << get<0>(res) << "\n\t runtime:\t" << get<1>(res) << "s\n\t memory:\t" << get<2>(res) << "\n\t num_sieves:\t" << get<3>(res) << "\n\n" << endl;

    /*solution_ids = random0.get_ids();
    for (auto x : solution_ids) {
        cout << x << ' ';
    }
    cout << endl;*/

    auto eps = { 0.01, 0.02, 0.05, 0.1 };
    for (auto e : eps) {
        cout << "Selecting " << K << " representatives via SieveStreaming with eps = " << e << std::endl;
        SieveStreaming sieve(K, fastIVM, 1.0, e);
        iota(ids.begin(), ids.end(), 0);
        res = evaluate_optimizer_ids(sieve, data,  ids);
        cout << "\t fval:\t\t" << get<0>(res) << "\n\t runtime:\t" << get<1>(res) << "s\n\t memory:\t" << get<2>(res) << "\n\t num_sieves:\t" << get<3>(res) << "\n\n" << endl;
        
        /*for (auto x : sieve.get_ids()) {
            cout << x << ' ';
        }cout << endl;*/

        iota(ids.begin(), ids.end(), 0);
        cout << "Selecting " << K << " representatives via SieveStreaming++ with eps = " << e << std::endl;
        SieveStreamingPP sievepp(K, fastIVM, 1.0, e);
        res = evaluate_optimizer_ids(sievepp, data, ids);
        cout << "\t fval:\t\t" << get<0>(res) << "\n\t runtime:\t" << get<1>(res) << "s\n\t memory:\t" << get<2>(res) << "\n\t num_sieves:\t" << get<3>(res) << "\n\n" << endl;
        /*for (auto x : sievepp.get_ids()) {
            cout << x << ' ';
        }cout << endl;*/
    }

    
}