# SMOH Math Library

A powerful C++ mathematical library that combines NumPy-like functionality with machine learning capabilities, advanced mathematical operations, and statistical analysis tools. This header-only library provides efficient implementations for scientific computing, machine learning, and data analysis tasks, with robust error handling and input validation.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Examples](#examples)
5. [API Reference](#api-reference)
6. [Error Handling](#error-handling)
7. [Contributing](#contributing)

## Features

### Matrix Operations (`smoh_matrix`)
- Matrix creation (random, zeros, ones, identity)
- Basic operations (addition, subtraction, multiplication)
- Advanced operations (determinant, trace, transpose)
- Matrix analysis and manipulation
- Shape verification and transformation

### Machine Learning Functions (`smoh_ml`)
- Activation functions (ReLU, Softmax)
- One-hot encoding
- Bias operations
- Derivative computations

### Statistical Operations (`smoh_statistic`)
- Basic statistics (mean, median, mode)
- Advanced statistics (variance, standard deviation)
- Data analysis tools

### Advanced Mathematics (`smoh_advmath`)
- Gaussian elimination
- Numerical integration

## Installation

### Project Structure
```
your_project/
│
├── smoh_math.h     # Main library header
├── main.cpp        # Your application
└── README.md
```

### Requirements
- C++ compiler with C++11 support
- Standard Template Library (STL)

### Compilation Steps

1. Clone or download the repository
2. Copy `smoh_math.h` to your project directory
3. Include the header in your source files:
```cpp
#include "smoh_math.h"
```

## Getting Started

### Basic Example
Create a file named `main.cpp`:

```cpp
#include <iostream>
#include "smoh_math.h"

using namespace std;
using namespace smoh_matrix;

int main() {
    // Create a 3x3 random matrix
    auto matrix = gen_random_matrix(3, 3);
    
    cout << "Random Matrix:" << endl;
    print_matrix(matrix);
    
    // Create identity matrix
    auto identity = make_unit_matrix(3, 3);
    
    cout << "\nIdentity Matrix:" << endl;
    print_matrix(identity);
    
    // Multiply matrices
    auto result = multiplyMatrices(matrix, identity);
    
    cout << "\nResult of multiplication:" << endl;
    print_matrix(result);
    
    return 0;
}
```

### Compilation and Running

```bash
# Compile
g++ main.cpp -o main

# Run
./main
```

For development with debugging:
```bash
g++ -g main.cpp -o main
```

For optimized release:
```bash
g++ -O3 main.cpp -o main
```

## Examples

### Matrix Operations
```cpp
using namespace smoh_matrix;

// Create matrices
auto mat1 = gen_random_matrix(3, 3);
auto mat2 = zeros(3, 3);
auto mat3 = ones(3, 3);

// Basic operations
auto sum = add_matrix(mat1, mat2);
auto product = multiplyMatrices(mat1, mat3);
auto transposed = transpose_matrix(mat1);

// Analysis
float det = determinant(mat1);
float tr = trace(mat1);
shape_of_matrix(mat1);
```

### Machine Learning Operations
```cpp
using namespace smoh_ml;

// Create input layer
auto input = smoh_matrix::gen_random_matrix(3, 3);

// Apply activations
auto activated = ReLU(input);
auto probabilities = Softmax(activated);

// One-hot encoding
vector<float> labels = {0, 1, 2};
auto encoded = one_hot(labels);

// Add bias
vector<vector<float>> bias = {{0.1}, {0.2}, {0.3}};
auto with_bias = addMatrixWithBias(activated, bias);
```

### Statistical Analysis
```cpp
using namespace smoh_statistic;

vector<float> dataset = {12.5, 14.2, 15.8, 12.5, 13.7};

float mean_val = mean(dataset);
float median_val = median(dataset);
float mode_val = mode(dataset);
float variance_val = variance(dataset);
float std_dev = standard_deviation(dataset);
```

### Advanced Mathematics
```cpp
using namespace smoh_advmath;

// Numerical integration
double result = integration([](double x) { return x * x; }, 0, 1);

// System of linear equations
vector<vector<float>> equations = {
    {2, 1, -1, 8},
    {-3, -1, 2, -11},
    {-2, 1, 2, -3}
};
gauess_elimination(equations);
```

## API Reference

### Matrix Operations
- `gen_random_matrix(rows, cols)`: Generate random matrix
- `input_matrix(rows, cols)`: Create matrix with user input
- `print_matrix(matrix)`: Display matrix
- `multiplyMatrices(matrix1, matrix2)`: Matrix multiplication
- `transpose_matrix(matrix)`: Matrix transposition
- `determinant(matrix)`: Calculate determinant
- `trace(matrix)`: Calculate trace

### Machine Learning
- `ReLU(matrix)`: Apply ReLU activation
- `Softmax(matrix)`: Apply Softmax activation
- `one_hot(labels)`: Create one-hot encoding
- `relu_derivative_2d(matrix)`: Calculate ReLU derivative

### Statistics
- `mean(vector)`: Calculate mean
- `median(vector)`: Calculate median
- `mode(vector)`: Calculate mode
- `variance(vector)`: Calculate variance
- `standard_deviation(vector)`: Calculate standard deviation

## Error Handling

The library includes comprehensive error checking:
- Matrix dimension validation
- Empty matrix detection
- Invalid index handling
- Numerical stability checks

Example:
```cpp
try {
    auto result = multiplyMatrices(matrix1, matrix2);
} catch (const runtime_error& e) {
    cerr << "Error: " << e.what() << endl;
}
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

### Code Style
- Use consistent indentation
- Add comments for complex operations
- Include error handling
- Write unit tests for new features

## Version
1.0.0
