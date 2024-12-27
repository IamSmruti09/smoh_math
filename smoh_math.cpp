#include <bits/stdc++.h>
#include "smoh_math.h"
#include<algorithm>
#include <cstdlib>
#include <cmath>
using namespace std;

namespace smoh_array
{
    void print_array(vector<int> vec){
        for (int i = 0; i < vec.size(); i++)
        {
            cout<<vec[i]<<endl;
        }
        
    }
} // namespace smoh_array


namespace smoh_matrix
{
    // Input and store a 2D matrix
vector<vector<float>> input_matrix(float num_row, float num_col) {
    if (num_row <= 0 || num_col <= 0) {
        cout << "Error: Matrix dimensions must be positive integers!" << endl;
        return {};
    }

    vector<vector<float>> vec(num_row, vector<float>(num_col));

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            while (true) {
                cout << "Enter element for position (" << i + 1 << "," << j + 1 << "): ";
                cin >> vec[i][j];
                
                if (cin.fail()) {
                    cin.clear();  
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');  
                    cout << "Invalid input. Please enter a valid number." << endl;
                } else {
                    break;
                }
            }
        }
    }

    return vec;
}

// generating random number matrix
vector<vector<float>> gen_random_matrix(float num_row, float num_col) {
    if (num_row <= 0 || num_col <= 0) {
        cout << "Error: Matrix dimensions must be positive integers!" << endl;
        return {};
    }

    vector<vector<float>> vec(num_row, vector<float>(num_col));

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            while (true) {
                float a = static_cast<float>(rand()) / RAND_MAX;
                vec[i][j]=a;
                
                if (cin.fail()) {
                    cin.clear();  
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');  
                    cout << "Invalid input. Please enter a valid number." << endl;
                } else {
                    break;
                }
            }
        }
    }

    return vec;
}



void print_matrix(vector<vector<float>> vec) {
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return;
    }

    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[0].size(); j++) {
            cout << vec[i][j] << " ";
        }
        cout << endl;
    }
}

void print_header(vector<vector<float>> vec,int a) {
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return;
    }

    for (int i = 0; i < a; i++) {
        for (int j = 0; j < vec[0].size(); j++) {
            cout << vec[i][j] << " ";
        }
        cout << endl;
    }
}


vector<vector<float>> add_matrix(vector<vector<float>> vec1, vector<vector<float>> vec2) {
    if (vec1.size() != vec2.size() || vec1[0].size() != vec2[0].size()) {
        cout << "Error: Matrices have different dimensions, cannot add." << endl;
        return {};
    }

    vector<vector<float>> res_vec(vec1.size(), vector<float>(vec1[0].size()));

    for (int i = 0; i < vec1.size(); i++) {
        for (int j = 0; j < vec1[0].size(); j++) {
            res_vec[i][j] = vec1[i][j] + vec2[i][j];
        }
    }

    return res_vec;
}


vector<vector<float>> sub_matrix(const vector<vector<float>>& vec1, const vector<vector<float>>& vec2) {
    if (vec1.empty() || vec2.empty()) {
        throw runtime_error("Error: Empty matrix in subtraction");
    }
    if (vec1.size() != vec2.size() || vec1[0].size() != vec2[0].size()) {
        throw runtime_error("Error: Matrix dimension mismatch in subtraction: (" + 
                          to_string(vec1.size()) + "," + to_string(vec1[0].size()) + ") vs (" +
                          to_string(vec2.size()) + "," + to_string(vec2[0].size()) + ")");
    }
    
    vector<vector<float>> res_vec(vec1.size(), vector<float>(vec1[0].size()));
    for (size_t i = 0; i < vec1.size(); i++) {
        for (size_t j = 0; j < vec1[0].size(); j++) {
            res_vec[i][j] = vec1[i][j] - vec2[i][j];
        }
    }
    return res_vec;
}

vector<vector<float>> multiplyMatrices(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty()) {
        throw runtime_error("Error: Empty matrix in multiplication");
    }

    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();

    if (A_cols != B_rows) {
        throw runtime_error("Error: Invalid dimensions for matrix multiplication: (" + 
                          to_string(A_rows) + "," + to_string(A_cols) + ") x (" +
                          to_string(B_rows) + "," + to_string(B_cols) + ")");
    }

    vector<vector<float>> result(A_rows, vector<float>(B_cols, 0));
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}


vector<vector<float>> transpose_matrix(const vector<vector<float>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        throw runtime_error("Error: Empty matrix in transpose");
    }

    vector<vector<float>> res_vec(vec[0].size(), vector<float>(vec.size()));
    for (size_t i = 0; i < vec.size(); i++) {
        for (size_t j = 0; j < vec[0].size(); j++) {
            res_vec[j][i] = vec[i][j];
        }
    }
    return res_vec;
}

vector<vector<float>> scalar_mul(const vector<vector<float>>& vec, float scalar) {
    if (vec.empty() || vec[0].empty()) {
        throw runtime_error("Error: Empty matrix in scalar multiplication");
    }

    vector<vector<float>> res_vec(vec.size(), vector<float>(vec[0].size()));
    for (size_t i = 0; i < vec.size(); i++) {
        for (size_t j = 0; j < vec[0].size(); j++) {
            res_vec[i][j] = scalar * vec[i][j];
        }
    }
    return res_vec;
}


long long sizeofmatrix(vector<vector<float>> vec) {
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return 0;
    }
    long long a = vec.size();
    long long b = vec[0].size();
    return a * b;
}

void shape_of_matrix(vector<vector<float>> vec) {
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return;
    }
    cout << vec.size() << "x" << vec[0].size() << endl;
}


vector<vector<float>> ones(float row, float col) {
    if (row <= 0 || col <= 0) {
        cout << "Error: Matrix dimensions must be positive." << endl;
        return {};
    }

    vector<vector<float>> res_vec(row, vector<float>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            res_vec[i][j] = 1;
        }
    }

    return res_vec;
}

vector<vector<float>> zeros(float row, float col) {
    if (row <= 0 || col <= 0) {
        cout << "Error: Matrix dimensions must be positive." << endl;
        return {};
    }

    vector<vector<float>> res_vec(row, vector<float>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            res_vec[i][j] = 0;
        }
    }

    return res_vec;
}

void is_square(vector<vector<float>> vec) {
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return;
    }

    if (vec.size() == vec[0].size()) {
        cout << "yes" << endl;
    } else {
        cout << "no" << endl;
    }
}

vector<vector<float>> get_row(const vector<vector<float>>& vec, int num_row) {
    if (vec.empty() || vec[0].empty()) {
        throw runtime_error("Error: The matrix is empty.");
    }

    if (num_row < 1 || num_row > vec.size()) {
        throw out_of_range("Error: Invalid row number.");
    }

    return {vec[num_row - 1]}; // Return row wrapped in a 2D vector
}

vector<vector<float>> get_col(const vector<vector<float>>& vec, int num_col) {
    if (vec.empty() || vec[0].empty()) {
        throw runtime_error("Error: The matrix is empty.");
    }

    if (num_col < 1 || num_col > vec[0].size()) {
        throw out_of_range("Error: Invalid column number.");
    }

    vector<vector<float>> col(vec.size(), vector<float>(1));
    for (int i = 0; i < vec.size(); i++) {
        col[i][0] = vec[i][num_col - 1];
    }

    return col;
}




float determinant(vector<vector<float>> vec) {
    int n = vec.size();

    // Handle empty matrix
    if (n == 0 || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return 0;
    }

    // Check if the matrix is square
    if (n != vec[0].size()) {
        cout << "Matrix must be square to find determinant!" << endl;
        return 0;
    }
    
    // Base case for 1x1 matrix
    if (n == 1) {
        return vec[0][0];
    }

    // Base case for 2x2 matrix
    if (n == 2) {
        return vec[0][0] * vec[1][1] - vec[0][1] * vec[1][0];
    }

    float det = 0;
    for (int i = 0; i < n; i++) {
        vector<vector<float>> sub_matrix;
        for (int j = 1; j < n; j++) {
            vector<float> row;
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    row.push_back(vec[j][k]);
                }
            }
            sub_matrix.push_back(row);
        }
        det += ((i % 2 == 0 ? 1 : -1) * vec[0][i] * determinant(sub_matrix));
    }

    return det;
}



vector<vector<float>> hadamard_product(const vector<vector<float>>& vec1, const vector<vector<float>>& vec2) {
    if (vec1.empty() || vec2.empty() || vec1[0].empty() || vec2[0].empty()) {
        throw runtime_error("Error: Empty matrix in Hadamard product");
    }

    if (vec1.size() != vec2.size() || vec1[0].size() != vec2[0].size()) {
        throw runtime_error("Error: Matrix dimension mismatch in Hadamard product: (" + 
                          to_string(vec1.size()) + "," + to_string(vec1[0].size()) + ") vs (" +
                          to_string(vec2.size()) + "," + to_string(vec2[0].size()) + ")");
    }

    vector<vector<float>> res_vec(vec1.size(), vector<float>(vec1[0].size()));
    for (size_t i = 0; i < vec1.size(); i++) {
        for (size_t j = 0; j < vec1[0].size(); j++) {
            res_vec[i][j] = vec1[i][j] * vec2[i][j];
        }
    }
    return res_vec;
}

float trace(vector<vector<float>> vec) {
    // Check if matrix is empty
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return 0;
    }

    if (vec.size() != vec[0].size()) {
        cout << "Matrix must be square to calculate trace!" << endl;
        return 0;
    }

    float trace_value = 0;
    for (int i = 0; i < vec.size(); i++) {
        trace_value += vec[i][i];
    }
    return trace_value;
}


void swap_rows(vector<vector<float>>& vec, int row1, int row2) {
    // Validate row indices
    if (row1 < 0 || row1 >= vec.size() || row2 < 0 || row2 >= vec.size()) {
        cout << "Invalid row indices!" << endl;
        return;
    }
    swap(vec[row1], vec[row2]);
}

void swap_cols(vector<vector<float>>& vec, int col1, int col2) {
    // Validate column indices
    if (col1 < 0 || col1 >= vec[0].size() || col2 < 0 || col2 >= vec[0].size()) {
        cout << "Invalid column indices!" << endl;
        return;
    }
    for (int i = 0; i < vec.size(); i++) {
        swap(vec[i][col1], vec[i][col2]);
    }
}

void has_a_shapeof(vector<vector<float>> vec) {
    // Check if the matrix is empty
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: The matrix is empty." << endl;
        return;
    }

    bool is_upper = true;
    bool is_lower = true;

    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[0].size(); j++) {

            if (i > j && vec[i][j] != 0) {
                is_upper = false;
            }

            if (i < j && vec[i][j] != 0) {
                is_lower = false;
            }
        }
    }

    // Determine the matrix type based on the flags
    if (is_upper && is_lower) {
        cout << "It is a diagonal matrix." << endl;
    } else if (is_upper) {
        cout << "It is an upper triangular matrix." << endl;
    } else if (is_lower) {
        cout << "It is a lower triangular matrix." << endl;
    } else {
        cout << "It is neither an upper nor a lower triangular matrix." << endl;
    }
}

vector<vector<float>> make_unit_matrix(float num_row, float num_col) {

    if (num_row != num_col) {
        cout << "Error: Unit matrix must be square (same number of rows and columns)." << endl;
        return {};  
    }

    vector<vector<float>> vec(num_row, vector<float>(num_col));

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            if (i == j) {
                vec[i][j] = 1;  
            } else {
                vec[i][j] = 0; 
            }
        }
    }

    return vec;  
}


vector<float> all_diagonal_element(vector<vector<float>> vec) {
    vector<float> res;

    
    if (vec.empty() || vec[0].empty()) {
        cout << "Error: Matrix is empty." << endl;
        return res; 
    }


    for (int i = 0; i < vec.size() && i < vec[0].size(); i++) {
        res.push_back(vec[i][i]);
    }
    
    return res;
}

vector<vector<float>> rowWiseSum(const vector<vector<float>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        throw runtime_error("Error: Empty matrix in rowWiseSum");
    }

    vector<vector<float>> rowSums(matrix.size(), vector<float>(1, 0.0f));
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            rowSums[i][0] += matrix[i][j];
        }
    }
    return rowSums;
}





} // namespace smohmath

namespace smoh_advmath
{
void gauess_elimination(vector<vector<float>> &vec) {
    vector<float> ans(vec.size());
    int n = vec.size();  // Get the number of rows

    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(vec[k][i]) > abs(vec[max_row][i])) {
                max_row = k;
            }
        }

        if (i != max_row) {
            swap(vec[i], vec[max_row]);
        }

        // Check for singular matrix
        if (vec[i][i] == 0) {
            cout << "Matrix is singular, cannot solve!" << endl;
            return;
        }

        for (int j = i + 1; j < n; j++) {
            float ratio = vec[j][i] / vec[i][i];
            for (int k = i; k < n + 1; k++) {
                vec[j][k] -= ratio * vec[i][k];
            }
        }
    }

    // Backward Substitution
    for (int i = n - 1; i >= 0; i--) {
        ans[i] = vec[i][n] / vec[i][i];
        for (int j = i - 1; j >= 0; j--) {
            vec[j][n] -= vec[j][i] * ans[i];
        }
    }

    // Print solution
    for (int i = 0; i < ans.size(); i++) {
        cout << ans[i] << " ";
    }
    cout << endl;
}

double integration(double (*func)(double), double lower, double upper) {
    double h = (upper - lower) / 999999;
    double ans = func(lower) + func(upper);
    double ans1 = 0;
    double ans2 = 0;

    for (int i = 1; i < 999998; i++) {
        lower += h;
        if (i % 3 == 0) {
            ans1 += func(lower);
        } else {
            ans2 += func(lower);
        }
    }

    double result = ((3 * h) / 8) * (ans + (3 * ans2) + (2 * ans1));
    return result;
}


} // namespace smoh_advmath

namespace smoh_statistic
{
    // mean
    float mean(vector<float> vec){
        float sum=0;
        for (int i = 0; i < vec.size(); i++)
        {
            sum+=vec[i];
        }
        return sum/vec.size();
    }

    // median
    float median(vector<float> vec){
        sort(vec.begin(),vec.end());
        if(vec.size()%2==0){
            return (vec[vec.size()/2]+vec[(vec.size()-1)/2])/2;
        }
        else{
            return vec[vec.size()/2];
        }
    }

    // mode:
    float mode(vector<float> vec){
        map<float,int> mpp;
        for (int i = 0; i < vec.size(); i++)
        {
            mpp[vec[i]]++;
        }
        int maxOccurrences = 0;
        int elementWithMaxOccurrences = 0;

        for (const auto& mpp : mpp) {
            if (mpp.second > maxOccurrences) {
                maxOccurrences = mpp.second;
                elementWithMaxOccurrences = mpp.first;  
            }
        }
        return elementWithMaxOccurrences;
    }

    float variance(vector<float> vec){
        float a= smoh_statistic::mean(vec);
        float res=0;
        for (int i = 0; i < vec.size(); i++)
        {
            res+=((vec[i]-a)*(vec[i]-a));
        }
        return res/vec.size();
    }

    float sqrt_approx(float num) {
        if (num < 0) {
            cout << "Invalid Input: Negative number doesn't have a real square root." << endl;
            return -1;
        }

        float guess = num / 2.0;
        float tolerance = 0.00001;
        
        while (true) {
            float new_guess = 0.5 * (guess + num / guess);
            if (abs(new_guess - guess) < tolerance) {
                break;
            }
            guess = new_guess;
        }
        
        return guess;
    }

    float standard_deviation(vector<float> vec){
        float a = smoh_statistic::variance(vec);
        return sqrt_approx(a);
    }



}

namespace smoh_ml {

vector<vector<float>> ReLU(vector<vector<float>>& vec) {
    if (vec.empty()) return {}; 

    vector<vector<float>> res_vec(vec.size(), vector<float>(vec[0].size()));  

    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++) {
            res_vec[i][j] = (vec[i][j] > 0) ? vec[i][j] : 0;  
        }
    }

    return res_vec;
}

vector<vector<float>> Softmax(vector<vector<float>>& vec) {
    if (vec.empty()) return {}; 

    vector<vector<float>> res_vec(vec.size(), vector<float>(vec[0].size()));

    for (int j = 0; j < vec[0].size(); ++j) {  
        float max_val = vec[0][j];
        for (int i = 1; i < vec.size(); ++i) {
            max_val = max(max_val, vec[i][j]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < vec.size(); ++i) {
            sum_exp += exp(vec[i][j] - max_val);
        }

        for (int i = 0; i < vec.size(); ++i) {
            res_vec[i][j] = exp(vec[i][j] - max_val) / sum_exp;
        }
    }

    return res_vec;
}

vector<vector<float>> one_hot(const vector<float>& labels) {
    int num_classes = 10;
   
    vector<vector<float>> one_hot_matrix(num_classes, vector<float>(labels.size(), 0));
    for(size_t i = 0; i < labels.size(); i++) {
        one_hot_matrix[static_cast<int>(labels[i])][i] = 1;
    }
    return one_hot_matrix;
}

vector<vector<float>> relu_derivative_2d(const vector<vector<float>>& input) {
    if (input.empty()) return {}; 

    vector<vector<float>> derivative(input.size(), vector<float>(input[0].size()));

    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            derivative[i][j] = (input[i][j] > 0) ? 1.0f : 0.0f;
        }
    }

    return derivative;
}

vector<vector<float>> addMatrixWithBias(const vector<vector<float>>& matrix, const vector<vector<float>>& bias) {
    if (bias.size() != matrix.size() || bias[0].size() != 1) {
        throw runtime_error("Bias matrix must have same number of rows as input matrix and one column.");
    }

    vector<vector<float>> result = matrix;

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i][j] += bias[i][0];
        }
    }

    return result;
}

} // namespace smoh_ml