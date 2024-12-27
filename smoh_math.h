#ifndef SMOH_MATH_H
#define SMOH_MATH_H
#include <bits/stdc++.h>
using namespace std;

namespace smoh_array{
    void print_array(vector<int> vec);  
}


namespace smoh_matrix
{
    // List of the functions
    vector<vector<float>> input_matrix(float num_row, float num_col);
    vector<vector<float>> gen_random_matrix(float num_row, float num_col);
    void print_matrix(vector<vector<float>> vec);
    void print_header(vector<vector<float>> vec,int a);
    vector<vector<float>> add_matrix(vector<vector<float>> vec1, vector<vector<float>> vec2);
    vector<vector<float>> sub_matrix(const vector<vector<float>>& vec1, const vector<vector<float>>& vec2);
    vector<vector<float>> transpose_matrix(const vector<vector<float>>& vec);
    vector<vector<float>> scalar_mul(const vector<vector<float>>& vec, float scalar);
    long long sizeofmatrix(vector<vector<float>> vec);
    void shape_of_matrix(vector<vector<float>> vec);
    vector<vector<float>> zeros(float row, float col);
    vector<vector<float>> ones(float row, float col);
    void is_square(vector<vector<float>> vec);
    vector<vector<float>> get_row(const vector<vector<float>>& vec, int num_row);
    vector<vector<float>> get_col(const vector<vector<float>>& vec, int num_col);
    float determinant(vector<vector<float>> vec);
    vector<vector<float>> hadamard_product(const vector<vector<float>>& vec1, const vector<vector<float>>& vec2);
    float trace(vector<vector<float>> vec);
    void swap_rows(vector<vector<float>>& vec, int row1, int row2);
    void swap_cols(vector<vector<float>>& vec, int col1, int col2);
    void has_a_shapeof(vector<vector<float>> vec);
    vector<vector<float>> make_unit_matrix(float num_row, float num_col);
    vector<float> all_diagonal_element(vector<vector<float>> vec);
    vector<vector<float>> multiplyMatrices(const vector<vector<float>>& A, const vector<vector<float>>& B);
    vector<vector<float>> rowWiseSum(const vector<vector<float>>& matrix);
} // namespace smoh_matrix

namespace smoh_statistic
{
    float mean(vector<float> vec);
    float median(vector<float> vec);
    float mode(vector<float> vec);
    float variance(vector<float> vec);
    float standard_deviation(vector<float> vec);
} // namespace smoh_statistic

namespace smoh_advmath
{
    void gauess_elimination(vector<vector<float>>& vec);
    double integration(double (*func)(double), double lower, double upper);
} // namespace smoh_advmath


namespace smoh_ml
{
    vector<vector<float>> ReLU(vector<vector<float>>& vec);
    vector<vector<float>> Softmax(vector<vector<float>>& vec);
 vector<vector<float>> one_hot(const vector<float>& labels);
    vector<vector<float>> avg_bias(vector<vector<float>> vec,float num);
    std::vector<std::vector<float>> relu_derivative_2d(const std::vector<std::vector<float>>& input);
    vector<vector<float>> addMatrixWithBias(const vector<vector<float>>& matrix, const vector<vector<float>>& bias);

} // namespace smoh_ml


#endif // SMOH_MATH_
