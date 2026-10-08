// The predictor's forward pass as a MUSA kernel: a batch of feature vectors in,
// one logit out each.
//
// Why this exists, on a project whose bottleneck is not loops: it is the honest
// answer to "did you use the GPU". Scoring 30k scenarios through torch on the MTT
// S4000 takes about 13 seconds, so a hand-written kernel buys no wall clock at
// all. What it buys is a second, independent implementation of the same
// arithmetic, in another language, on another device -- a way of checking the
// model rather than a way of speeding it up. The README says exactly that and
// does not dress it up as a performance result.
//
// Layout: the layer weights live in one contiguous buffer with an absolute index
// per layer, so a layer is  out[r] = b[bstart[L] + r] + sum_c W[start[L] + r*nin + c] * in[c].
// One block per row, walking all layers with the activations in shared memory.
// HIDDEN is at most 256 wide, so shared memory costs nothing.
//
// Built with mcc from MUSA 3.1.0 (clang-14), target MUSA_ARCH set by musa/build.sh.

#include <musa_runtime.h>

#define MAX_DIM 512

__device__ __forceinline__ float silu_f(float x) {
    // silu(x) = x * sigmoid(x). The expf is the device fast intrinsic; the parity
    // check against the Python reference uses tolerance 2e-4 rather than exact
    // equality precisely because this is a fast-math exponential, and it is
    // measured rather than assumed.
    return x / (1.0f + __expf(-x));
}

extern "C" __global__ void mlp_forward(
    const float* __restrict__ x,        // [n_rows, in_dim]
    const float* __restrict__ wpack,    // all weight blocks, row-major
    const float* __restrict__ bpack,    // all bias blocks
    const int*   __restrict__ start,    // absolute float index of each weight block
    const int*   __restrict__ bstart,   // absolute float index of each bias block
    const int*   __restrict__ l_in,     // input width per layer
    const int*   __restrict__ l_out,    // output width per layer
    const int    n_layers,
    const int    in_dim,
    float* __restrict__ out)            // [n_rows], one logit per row
{
    __shared__ float buf_a[MAX_DIM];
    __shared__ float buf_b[MAX_DIM];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int c = tid; c < in_dim; c += nthreads) {
        buf_a[c] = x[row * in_dim + c];
    }
    __syncthreads();

    for (int L = 0; L < n_layers; ++L) {
        const int nin = l_in[L];
        const int nout = l_out[L];
        const int wo = start[L];
        const int bo = bstart[L];
        for (int r = tid; r < nout; r += nthreads) {
            float acc = bpack[bo + r];
            const float* wr = wpack + wo + r * nin;
            for (int c = 0; c < nin; ++c) {
                acc += wr[c] * buf_a[c];
            }
            // SiLU on every layer but the last, which is the raw logit
            buf_b[r] = (L == n_layers - 1) ? acc : silu_f(acc);
        }
        __syncthreads();
        for (int c = tid; c < nout; c += nthreads) {
            buf_a[c] = buf_b[c];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = buf_a[0];
    }
}


// Host-side launcher, compiled by mcc inside this .mu translation unit, where the
// kernel symbol and the <<<>>> syntax are both in scope. The driver -- an ordinary
// .cpp compiled by g++ -- calls this instead.
extern "C" void mlp_forward_kernel_launch(
    const float* x, const float* wpack, const float* bpack,
    const int* start, const int* bstart, const int* l_in, const int* l_out,
    int n_layers, int in_dim, float* out, int n_rows)
{
    // 256 threads per block: HIDDEN is 256 wide, so one block covers an entire
    // layer's output in a single step for the widest layer in this model.
    mlp_forward<<<n_rows, 256>>>(x, wpack, bpack, start, bstart, l_in, l_out,
                                 n_layers, in_dim, out);
}
