#include <x86intrin.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

// Runtime evaluation for squared Eucliden distance functions
// - fvec_L2_sqr_ref: naive reference impl from Faiss
// - fvec_L2_sqr_sse: SSE impl from Faiss
// - fvec_L2_sqr_avx: AVX impl from Faiss
// - fvec_L2_sqr_avx512: AVX512 impl

// Note that fvec_L2_sqr_{ref, sse, avx} are from Faiss:
// https://github.com/facebookresearch/faiss/blob/v1.2.1/utils.cpp
// https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp

// Compile:
//   $ g++ -O3 -Wall --std=c++14 -march=native -o main main.cpp

// Result on intel mac pro:
// ref: 1168 msec
// sse : 281 msec
// avx : 177 msec
// avx512 : 145 msec

const uint uint32_max = 4294967295;
std::mt19937 mt(123);

std::vector<std::vector<float>> gen_random_fvectors(int N, int D) {
    std::vector<std::vector<float>> vecs(N, std::vector<float>(D));
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            vecs[n][d] = (float)mt() / uint32_max;
        }
    }
    return vecs;
}

float fvec_L2sqr_ref(const float *x, const float *y, size_t d) {
    size_t i;
    float res_ = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res_ += tmp * tmp;
    }
    return res_;
}

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float *x) {
    assert(0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

// SSE implementation
float fvec_L2sqr_sse(const float *x, const float *y, size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

// reads 0 <= d < 8 floats as __m256
static inline __m256 masked_read_8(int d, const float *x) {
    assert(0 <= d && d < 8);
    if (d < 4) {
        __m256 res = _mm256_setzero_ps();
        res = _mm256_insertf128_ps(res, masked_read(d, x), 0);
        return res;
    } else {
        __m256 res = _mm256_setzero_ps();
        res = _mm256_insertf128_ps(res, _mm_loadu_ps(x), 0);
        res = _mm256_insertf128_ps(res, masked_read(d - 4, x + 4), 1);
        return res;
    }
}

float fvec_L2sqr_avx(const float *x, const float *y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

#ifdef __AVX512F__
// reads 0 <= d < 16 floats as __m512
static inline __m512 masked_read_16(int d, const float *x) {
    assert(0 <= d && d < 16);
    if (d < 8) {
        __m512 res = _mm512_setzero_ps();
        res = _mm512_insertf32x8(res, masked_read_8(d, x), 0);
        return res;
    } else {
        __m512 res = _mm512_setzero_ps();
        res = _mm512_insertf32x8(res, _mm256_loadu_ps(x), 0);
        res = _mm512_insertf32x8(res, masked_read_8(d - 8, x + 8), 1);
        return res;
    }
}

float fvec_L2sqr_avx512(const float *x, const float *y, size_t d) {
    __m512 msum1 = _mm512_setzero_ps();

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps(x);
        x += 16;
        __m512 my = _mm512_loadu_ps(y);
        y += 16;
        const __m512 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 16;
    }

    __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
    msum2 += _mm512_extractf32x8_ps(msum1, 0);

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
    msum3 += _mm256_extractf128_ps(msum2, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = mx - my;
        msum3 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = mx - my;
        msum3 += a_m_b1 * a_m_b1;
    }

    msum3 = _mm_hadd_ps(msum3, msum3);
    msum3 = _mm_hadd_ps(msum3, msum3);
    return _mm_cvtss_f32(msum3);
}
#endif

int main() {
    size_t N = 10000;
    size_t Nq = 1000;
    size_t D = 128;

    std::vector<std::vector<float>> queries = gen_random_fvectors(Nq, D);
    std::vector<std::vector<float>> vecs = gen_random_fvectors(N, D);

    // ============== Value check ===============
    for (const auto &q : queries) {
        for (size_t n = 0; n < N; ++n) {
            float d1 = fvec_L2sqr_ref(q.data(), vecs[n].data(), D);
            float d2 = fvec_L2sqr_sse(q.data(), vecs[n].data(), D);
            float d3 = fvec_L2sqr_avx(q.data(), vecs[n].data(), D);
            assert(std::abs(d1 - d2) < 0.001);
            assert(std::abs(d1 - d3) < 0.001);
#ifdef __AVX512F__
            float d4 = fvec_L2sqr_avx512(q.data(), vecs[n].data(), D);
            assert(std::abs(d1 - d4) < 0.001);
#endif
        }
    }

    // ===================== Runtime evaluation ======================

    auto eval = [&](std::string method, auto fvec_L2sqr) {
        std::vector<float> rets(N);
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();
        for (const auto &q : queries) {
            for (size_t n = 0; n < N; ++n) {
                rets[n] = fvec_L2sqr(q.data(), vecs[n].data(), D);
            }
        }
        end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << method << ": " << elapsed << " msec" << std::endl;
    };

    eval("ref", fvec_L2sqr_ref);
    eval("sse", fvec_L2sqr_sse);
    eval("avx", fvec_L2sqr_avx);
#ifdef __AVX512F__
    eval("avx512", fvec_L2sqr_avx512);
#endif

    return 0;
}
