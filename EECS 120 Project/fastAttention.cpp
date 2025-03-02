
#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <chrono>

// Function declarations
void matTrans(torch::Tensor AT, torch::Tensor A);
void matMul(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void naiveAttentionImpl(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask, torch::Tensor output);

// Function to transpose a tensor
torch::Tensor transpose(torch::Tensor A) {
    torch::Tensor AT = torch::zeros_like(A);
    matTrans(AT, A);
    return AT;
}

// Function to perform matrix multiplication
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    torch::Tensor C = torch::zeros({ A.size(0), B.size(1) },
        torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    matMul(A, B, C);
    return C;
}

// Function to perform naive attention
torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask) {
    torch::Tensor output = torch::zeros_like(V);
    naiveAttentionImpl(Q, K, V, mask, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose functions to Python
    m.def("naive_transpose", &transpose, "naive transpose");
    m.def("naive_matmul", &matmul, "naive matrix multiplication");
    m.def("naive_attention", &naiveAttention, "naive attention");
    // m.def("fused_attention", &fusedAttention, "fused attention");
    // m.def("tc_fused_attention", &tcFusedAttention, "fused attention with tensor cores");
    // m.def("sparse_tc_fused_attention", &sparseTcFusedAttention, "sparse fused attention with tensor cores");
    // add more here if you have more variants to test
}