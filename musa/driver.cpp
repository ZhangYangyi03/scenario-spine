// Host driver for the MUSA predictor kernel.
//
//   ./predict <state_dir> <x.f32> <n_rows> <in_dim> <out.f32>
//
// state_dir holds the flattened model written by tools/export_state.py:
// shapes.txt (one "<out> <in>" per linear layer), weights.f32, biases.f32.
//
// No framework anywhere in this file: musaMalloc and musaMemcpy only, so the
// kernel is exercised by code that shares nothing with torch and the parity check
// between them means something.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <musa_runtime.h>

#define MUSA_CHECK(x) do { musaError_t e_ = (x); if (e_ != musaSuccess) { \
    fprintf(stderr, "MUSA error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(2); } } while (0)

// Launched through the runtime API rather than with <<<>>> from a plain g++
// translation unit: the triple-chevron form is a language extension and only
// parses in a .mu file compiled by mcc. Passing a host function pointer with the
// argument list is the portable form, and it keeps the driver compilable by an
// ordinary C++ compiler.
extern "C" void mlp_forward_kernel_launch(
    const float* x, const float* wpack, const float* bpack,
    const int* start, const int* bstart, const int* l_in, const int* l_out,
    int n_layers, int in_dim, float* out, int n_rows);

static std::vector<float> read_f32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <state_dir> <x.f32> <n_rows> <in_dim> <out.f32>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const int n_rows = atoi(argv[3]);
    const int in_dim = atoi(argv[4]);

    std::ifstream shapes(dir + "/shapes.txt");
    if (!shapes) { fprintf(stderr, "cannot read %s/shapes.txt\n", dir.c_str()); return 2; }
    std::vector<int> l_in, l_out, start, bstart;
    {
        std::string line;
        int woff = 0, boff = 0;
        while (std::getline(shapes, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            int o = 0, i = 0;
            ss >> o >> i;
            l_out.push_back(o);
            l_in.push_back(i);
            start.push_back(woff);
            bstart.push_back(boff);
            woff += o * i;
            boff += o;
        }
    }
    const int n_layers = (int)l_out.size();
    if (n_layers == 0) { fprintf(stderr, "no layers in shapes.txt\n"); return 2; }

    std::vector<float> w = read_f32(dir + "/weights.f32");
    std::vector<float> b = read_f32(dir + "/biases.f32");
    std::vector<float> x = read_f32(argv[2]);
    if ((int)x.size() != n_rows * in_dim) {
        fprintf(stderr, "x has %zu floats, expected %d\n", x.size(), n_rows * in_dim);
        return 2;
    }

    float *dw = nullptr, *db = nullptr, *dx = nullptr, *dy = nullptr;
    int *d_start = nullptr, *d_bstart = nullptr, *d_lin = nullptr, *d_lout = nullptr;
    MUSA_CHECK(musaMalloc(&dw, w.size() * sizeof(float)));
    MUSA_CHECK(musaMalloc(&db, b.size() * sizeof(float)));
    MUSA_CHECK(musaMalloc(&dx, (size_t)n_rows * in_dim * sizeof(float)));
    MUSA_CHECK(musaMalloc(&dy, (size_t)n_rows * sizeof(float)));
    MUSA_CHECK(musaMalloc(&d_start, n_layers * sizeof(int)));
    MUSA_CHECK(musaMalloc(&d_bstart, n_layers * sizeof(int)));
    MUSA_CHECK(musaMalloc(&d_lin, n_layers * sizeof(int)));
    MUSA_CHECK(musaMalloc(&d_lout, n_layers * sizeof(int)));

    MUSA_CHECK(musaMemcpy(dw, w.data(), w.size() * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(db, b.data(), b.size() * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(dx, x.data(), (size_t)n_rows * in_dim * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_start, start.data(), n_layers * sizeof(int), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_bstart, bstart.data(), n_layers * sizeof(int), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_lin, l_in.data(), n_layers * sizeof(int), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_lout, l_out.data(), n_layers * sizeof(int), musaMemcpyHostToDevice));

    mlp_forward_kernel_launch(dx, dw, db, d_start, d_bstart, d_lin, d_lout,
                              n_layers, in_dim, dy, n_rows);
    MUSA_CHECK(musaGetLastError());
    MUSA_CHECK(musaDeviceSynchronize());

    std::vector<float> y(n_rows);
    MUSA_CHECK(musaMemcpy(y.data(), dy, (size_t)n_rows * sizeof(float), musaMemcpyDeviceToHost));

    std::ofstream o(argv[5], std::ios::binary);
    o.write(reinterpret_cast<char*>(y.data()), (size_t)n_rows * sizeof(float));
    // Event-timed repeats, because the wall clock of this process is dominated by
    // the process start and the file I/O, not by the kernel: a 30k-row call takes
    // 766 ms end to end of which the kernel itself is a small fraction. Reporting
    // only the wall clock would invite the reading that the kernel is 600x slower
    // than torch, which is a statement about subprocess startup and not about the
    // kernel. The event timer measures the launch only.
    const int reps = (argc >= 7) ? atoi(argv[6]) : 50;
    musaEvent_t e0, e1;
    MUSA_CHECK(musaEventCreate(&e0));
    MUSA_CHECK(musaEventCreate(&e1));
    // warm-up, because the first launch pays for module load
    mlp_forward_kernel_launch(dx, dw, db, d_start, d_bstart, d_lin, d_lout,
                              n_layers, in_dim, dy, n_rows);
    MUSA_CHECK(musaDeviceSynchronize());
    MUSA_CHECK(musaEventRecord(e0, 0));
    for (int r = 0; r < reps; ++r) {
        mlp_forward_kernel_launch(dx, dw, db, d_start, d_bstart, d_lin, d_lout,
                                  n_layers, in_dim, dy, n_rows);
    }
    MUSA_CHECK(musaEventRecord(e1, 0));
    MUSA_CHECK(musaDeviceSynchronize());
    float ms = 0.0f;
    MUSA_CHECK(musaEventElapsedTime(&ms, e0, e1));
    const double per = (double)ms / reps;
    printf("ok rows=%d in_dim=%d layers=%d\nkernel_ms_per_call=%.4f reps=%d "
           "rows_per_sec=%.1f\n",
           n_rows, in_dim, n_layers, per, reps, n_rows / (per / 1000.0));
    return 0;
}
