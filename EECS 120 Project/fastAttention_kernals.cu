#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
using namespace nvcuda;
#define NEGINFINITY -1e20f


__global__ void matTransKernel(float* AT, float* A, int rows, int cols);

void matTrans(torch::Tensor AT, torch::Tensor A) {

    assert(AT.size(0) == A.size(1));
    assert(AT.size(1) == A.size(0));

    int rows = A.size(0);
    int cols = A.size(1);


    int threadsPerBlock = 512;
    int blocks = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    matTransKernel << <blocks, threadsPerBlock >> > (
        AT.data_ptr<float>(),
        A.data_ptr<float>(),
        rows,
        cols);
}

__global__ void matTransKernel(float* AT, float* A, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    for (int i = tid; i < total_elements; i += blockDim.x * gridDim.x) {
        int row = i / cols;
        int col = i % cols;
        AT[col * rows + row] = A[row * cols + col];
    }
}


__global__ void navSoftmax(float* input, float* S, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows && idx < 1) {
        float maxValue = input[idy * cols];
        //find the max value in the row
        for (int i = 1; i < cols; i++) {
            if (input[idy * cols + i] > maxValue) {
                maxValue = input[idy * cols + i];
            }
        }
        //compute the softmax
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += exp(input[idy * cols + i] - maxValue);
        }
        for (int i = 0; i < cols; i++) {
            S[idy * cols + i] = exp(input[idy * cols + i] - maxValue) / sum;
        }
    }
}


__global__ void navMatMul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
		    float sum = 0.0f;
		    for (int i = 0; i < K; i++) {
			    sum += A[row * K + i] * B[i * N + col];
		    }
		    C[row * N + col] = sum;
	}
}


__global__ void navMasking(float* Out, float* S, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;

    if (idx < total) {
        int row = idx / seq_len;  // Row index
        int col = idx % seq_len;  // Column index

        if (col > row) {
            Out[idx] = -1e9; 
        }
        else {
            Out[idx] = S[idx]; //unmasked values
        }
    }
}



void naive_attention(torch::Tensor O, torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int seq_len = Q.size(0);
    int embed_dim = Q.size(1);

    torch::Tensor KT = torch::zeros({ embed_dim, seq_len }, torch::TensorOptions().dtype(K.dtype()).device(K.device()));
    matTrans(KT, K);

    torch::Tensor QKT = torch::zeros({ seq_len, seq_len }, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
    navMatMul<<<numBlocks, threadsPerBlock>>>(Q.data_ptr<float>(), KT.data_ptr<float>(), QKT.data_ptr<float>(), seq_len, seq_len, embed_dim);
    torch::Tensor S = torch::zeros_like(QKT);

    dim3 SBlocks(1, (seq_len + 31) / 32);
    dim3 SThreads(32, 32);

    navSoftmax <<<SBlocks, SThreads >>> (
        QKT.data_ptr<float>(),
        S.data_ptr<float>(),
        seq_len, seq_len
        );

    //torch::Tensor S_masked = torch::zeros_like(S);

	torch::Tensor O = torch::zeros_like(Q);

    dim3 outputBlocks((seq_len + 15) / 16, (embed_dim + 15) / 16);
    dim3 outputThreads(16, 16);

    navMatMul <<<outputBlocks, outputThreads >>> (
		S.data_ptr<float>(),
		V.data_ptr<float>(),
		O.data_ptr<float>(),
		seq_len, embed_dim, seq_len
		);

}




#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
using namespace nvcuda;
#define NEGINFINITY -1e20f
#define TILE_SIZE 32
#define NUM_WARP_PER_BLOCK 8

__global__ void matTransKernel(float* AT, float* A, int rows, int cc
    ols);

void matTrans(torch::Tensor AT, torch::Tensor A) {

    assert(AT.size(0) == A.size(1));
    assert(AT.size(1) == A.size(0));

    int rows = A.size(0);
    int cols = A.size(1);


    int threadsPerBlock = 512;
    int blocks = (rows * cols + threadsPerBlock - 1) / threadsPerBlocc
        k;

    matTransKernel << <blocks, threadsPerBlock >> > (
        AT.data_ptr<float>(),
        A.data_ptr<float>(),
        rows,
        cols);
}

__global__ void matTransKernel(float* AT, float* A, int rows, int cc
    ols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    for (int i = tid; i < total_elements; i += blockDim.x * gridDim.x))
    {
        int row = i / cols;
        int col = i % cols;
        AT[col * rows + row] = A[row * cols + col];
  }
}


__global__ void navSoftmax(float* input, float* S, int rows, int coo
    ls) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows && idx < 1) {
        float maxValue = input[idy * cols];
        //find the max value in the row
        for (int i = 1; i < cols; i++) {
            if (input[idy * cols + i] > maxValue) {
                maxValue = input[idy * cols + i];
            }
        }
        //compute the softmax
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += exp(input[idy * cols + i] - maxValuee
            );
        }
        for (int i = 0; i < cols; i++) {
            S[idy * cols + i] = exp(input[idy * cols +
                i] - maxValue) / sum;
        }
    }
}

__global__ void navMatMul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void naive_Attention(torch::Tensor O, torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int seq_len = Q.size(0);
    int embed_dim = Q.size(1);

    torch::Tensor KT = torch::zeros({ embed_dim, seq_len }, torch::TensorOptions().dtype(K.dtype()).device(K.device()));
    matTrans(KT, K); // take the transpose of K

    //compute QKT
    torch::Tensor QKT = torch::zeros({ seq_len, seq_len }, torch::
        :TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPP
        erBlock.x,
        (seq_len + threadsPerBlock.y - 1) / threadss
        PerBlock.y);


    //torch::Tensor half_Q = Q.to(torch::kHalf);
    //torch::Tensor half_KT = KT.to(torch::kHalf);

    navMatMul << <numBlocks, threadsPerBlock >> > (
        Q.data_ptr<float>(),
        KT.data_ptr<float>(),
        QKT.data_ptr<float>(),
        seq_len, seq_len, embed_dim
        );

    torch::Tensor S = torch::zeros_like(QKT);

    dim3 SBlocks(1, (seq_len + 31) / 32);
    dim3 SThreads(32, 32);

    navSoftmax <<<SBlocks, SThreads >>> (
        QKT.data_ptr<float>(),
        S.data_ptr<float>(),
        seq_len, seq_len
        );

    // Final matrix multiply S * V
    torch::Tensor half_S = S.to(torch::kHalf);
    torch::Tensor half_V = V.to(torch::kHalf);

    dim3 outputBlocks((seq_len + 15) / 16, (embed_dim + 15) / 16);

    navMatMul <<<outputBlocks, threadsPerBlock >>> (
        S.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        seq_len, embed_dim, seq_len
        );
}
