#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define BLK_M 16
#define BLK_N 16
#define BLK_K 16
#define NEGINFINITY -1e20f
#define BLOCKSIZE 256

// Kernel for transposing a matrix
__global__ void matTransKernel(float* AT, float* A, int N) {
    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < N * N; i += blockDim.x * gridDim.x * blockDim.y) {
        int row = i / N;
        int col = i % N;
        AT[col * N + row] = A[i];
    }
}

__global__ void naive_softmax(float* input, float* softmaxVec, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows && idx < 1) { //one thread per row
        float maxValue = input[idy * cols];
        // Find max value in the row
        for (int i = 1; i < cols; i++) {
            if (input[idy * cols + i] > maxValue) {
                maxValue = input[idy * cols + i];
            }
        }

        // Compute softmax
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += exp(input[idy * cols + i] - maxValue);
        }

        for (int i = 0; i < cols; i++) {
            softmaxVec[idy * cols + i] = exp(input[idy * cols + i] - maxValue) / sum;
        }
    }
}

__global__ void naive_masking(float* premaskedMat, float* mask, float* maskedMat, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows && idx < cols) {
        int index = idy * cols + idx;
        // Apply mask - if mask is 0, set to NEGINFINITY
        if (mask[index] == 0.0f) {
            maskedMat[index] = NEGINFINITY;
        }
        else {
            maskedMat[index] = premaskedMat[index];
        }
    }
}

// Device function for GEMM
__device__ void wmmaGemm(half* A, half* B, half* C, int M, int N, int K, int tileWidth) {
    int wid = threadIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    for (int i = 0; i < K / BLK_K; i++) {
        wmma::load_matrix_sync(a_frag, A + i * BLK_K, K);
        wmma::load_matrix_sync(b_frag, B + i * BLK_K * tileWidth + wid * BLK_N, tileWidth);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + wid * BLK_N, c_frag, tileWidth, wmma::mem_row_major);
}

// GEMM kernel using tensor cores
__global__ void gemm(half* A, half* B, half* C, int M, int N, int K, int tileWidth, int tileHeight) {
    int bid = blockIdx.x; // block id
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nTilesPerRow = N / tileWidth;
    int nTilesPerCol = M / tileHeight;
    int nTilesTotal = nTilesPerRow * nTilesPerCol;
    extern __shared__ half sharedMem[];
    int aShmSize = tileHeight * K;
    int bShmSize = K * tileWidth;
    int cShmSize = tileHeight * tileWidth;
    half* aShm = sharedMem;
    half* bShm = aShm + aShmSize;
    half* cShm = bShm + bShmSize;
    for (int i = bid; i < nTilesTotal; i += gridDim.x) {
        int tileRow = i / nTilesPerRow;
        int tileCol = i % nTilesPerRow;
        for (int j = tid; j < aShmSize; j += blockDim.x * blockDim.y) {
            int row = j / K;
            int col = j % K;
            aShm[j] = A[tileRow * tileHeight * K + row * K + col];
        }
        for (int j = tid; j < bShmSize; j += blockDim.x * blockDim.y) {
            int row = j / tileWidth;
            int col = j % tileWidth;
            bShm[j] = B[row * N + tileCol * tileWidth + col];
        }
        __syncthreads();
        //launch wmmaGemm
        wmmaGemm(aShm, bShm, cShm, M, N, K, tileWidth);
        __syncthreads();
        //store C
        for (int j = tid; j < cShmSize; j += blockDim.x * blockDim.y) {
            int row = j / tileWidth;
            int col = j % tileWidth;
            C[tileRow * tileHeight * N + tileCol * tileWidth + row * N + col] = cShm[j];
        }
    }
}


// Matrix transpose function
void matTrans(torch::Tensor AT, torch::Tensor A) {
    assert(AT.size(0) == A.size(1));
    assert(AT.size(1) == A.size(0));

    dim3 block(16, 16);
    dim3 grid((A.size(0) + block.x - 1) / block.x, (A.size(1) + block.y - 1) / block.y);

    matTransKernel << <grid, block >> > (
        AT.data_ptr<float>(),
        A.data_ptr<float>(),
        A.size(0)
        );
}

// Matrix multiplication 
void matMul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    assert(A.dim() == 2);
    assert(B.dim() == 2);
    assert(C.dim() == 2);
    assert(A.size(1) == B.size(0));
    assert(A.size(0) == C.size(0));
    assert(B.size(1) == C.size(1));

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    if (A.dtype() == torch::kHalf) {
        assert(M % BLK_M == 0);
        assert(N % BLK_N == 0);
        assert(K % BLK_K == 0);

        dim3 block(BLK_M, BLK_N);
        dim3 grid((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);

        int shared_mem_size = 3 * BLK_M * BLK_N * sizeof(half);

        gemm << <grid, block, shared_mem_size >> > (
            A.data_ptr<half>(),
            B.data_ptr<half>(),
            C.data_ptr<half>(),
            M, N, K, BLK_N, BLK_M
            );
    }
    else {

        C.copy_(torch::matmul(A, B));
    }
}

// Naive attention 
void naiveAttentionImpl(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask, torch::Tensor output) {
    assert(Q.dim() == 3);
    assert(K.dim() == 3);
    assert(V.dim() == 3);
    assert(mask.dim() == 2);
    assert(Q.size(0) == K.size(0));
    assert(Q.size(0) == V.size(0));
    assert(Q.size(1) == mask.size(0));
    assert(K.size(1) == mask.size(1));
    assert(Q.size(2) == K.size(2));

    int batch_size = Q.size(0);
    int seq_len_q = Q.size(1);
    int seq_len_k = K.size(1);
    int embed_dim = Q.size(2);

    torch::Tensor KT = torch::zeros({ batch_size, K.size(2), K.size(1) },
        torch::TensorOptions().device(K.device()).dtype(K.dtype()));

    for (int b = 0; b < batch_size; b++) {
        torch::Tensor K_b = K[b];
        torch::Tensor KT_b = KT[b];
        matTrans(KT_b, K_b);
    }

    torch::Tensor scores = torch::zeros({ batch_size, seq_len_q, seq_len_k },
        torch::TensorOptions().device(Q.device()).dtype(Q.dtype()));

    for (int b = 0; b < batch_size; b++) {
        torch::Tensor Q_b = Q[b];
        torch::Tensor KT_b = KT[b];
        torch::Tensor scores_b = scores[b];

        matMul(Q_b, KT_b, scores_b);
    }

    scores = scores / sqrt(static_cast<float>(embed_dim));

    torch::Tensor attention_weights = torch::zeros_like(scores);

    dim3 softmax_block(32, 32);
    dim3 softmax_grid((seq_len_k + softmax_block.x - 1) / softmax_block.x,
        (seq_len_q + softmax_block.y - 1) / softmax_block.y);

    for (int b = 0; b < batch_size; b++) {
        naive_softmax << <softmax_grid, softmax_block >> > (
            scores[b].data_ptr<float>(),
            attention_weights[b].data_ptr<float>(),
            seq_len_q,
            seq_len_k
            );
    }

    torch::Tensor masked_weights = torch::zeros_like(attention_weights);

    dim3 mask_block(32, 32);
    dim3 mask_grid((seq_len_k + mask_block.x - 1) / mask_block.x,
        (seq_len_q + mask_block.y - 1) / mask_block.y);

    for (int b = 0; b < batch_size; b++) {
        naive_masking << <mask_grid, mask_block >> > (
            attention_weights[b].data_ptr<float>(),
            mask.data_ptr<float>(),
            masked_weights[b].data_ptr<float>(),
            seq_len_q,
            seq_len_k
            );
    }

    for (int b = 0; b < batch_size; b++) {
        torch::Tensor masked_weights_b = masked_weights[b];
        torch::Tensor V_b = V[b];
        torch::Tensor output_b = output[b];

        matMul(masked_weights_b, V_b, output_b);
    }
}