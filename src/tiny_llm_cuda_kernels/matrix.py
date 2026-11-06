# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Week 2 Day 3 quantized_matmul matrix path (M > 8): tiled W4A16 GEMM on tensor cores.

Metal uses simdgroup_matrix 8x8 tiles; here each block computes a 64x64 output tile with
4 warps (2x2, 32x32 each) issuing mma.sync.m16n8k16 (f32 accumulate) through inline PTX, so no
CUDA headers are needed under NVRTC. Weights are dequantized to T (f16/bf16) into shared memory,
32 inner elements per stage (one quantization group covers the whole stage).

Layout as in quantized.py: a [M, N], b [K, N/8] uint32, scales/biases [K, N/128]; out [M, K].
use_split_k only changes Metal's scheduling, not the result, so it is accepted and ignored.
"""

from .common import kernel, launch

_TILED_SRC = r"""
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BKI = 32;
    constexpr int LDS = BKI + 8;
    __shared__ __align__(16) unsigned short As[BM * LDS];
    __shared__ __align__(16) unsigned short Bs[BN * LDS];
    const int M = a_shape[0];
    const int N = a_shape[1];
    const int K = b_shape[0];
    const int groups_per_row = N / 128;
    const int packed_cols = N / 8;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp >> 1) * 32;
    const int wn = (warp & 1) * 32;
    const int g = lane >> 2;
    const int t = lane & 3;
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;
    const unsigned short* a16 = reinterpret_cast<const unsigned short*>(a);
    for (int n0 = 0; n0 < N; n0 += BKI) {
        for (int idx = tid; idx < BM * (BKI / 2); idx += 128) {
            const int r = idx / (BKI / 2);
            const int c = (idx - r * (BKI / 2)) * 2;
            unsigned int v = 0u;
            if (row0 + r < M) {
                const unsigned short* p = a16 + (long)(row0 + r) * N + n0 + c;
                v = (unsigned int)p[0] | ((unsigned int)p[1] << 16);
            }
            *reinterpret_cast<unsigned int*>(&As[r * LDS + c]) = v;
        }
        for (int idx = tid; idx < BN * (BKI / 8); idx += 128) {
            const int c = idx / (BKI / 8);
            const int w = idx - c * (BKI / 8);
            const int col = col0 + c;
            unsigned short* dst = &Bs[c * LDS + w * 8];
            if (col < K) {
                const unsigned int word = b[(long)col * packed_cols + n0 / 8 + w];
                const int pi = col * groups_per_row + n0 / 128;
                const float s = static_cast<float>(scales[pi]);
                const float bi = static_cast<float>(biases[pi]);
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    T x = static_cast<T>(static_cast<float>((word >> (4 * v)) & 0xfu) * s + bi);
                    dst[v] = *reinterpret_cast<unsigned short*>(&x);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 8; ++v) dst[v] = 0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BKI; kk += 16) {
            unsigned int af[2][4];
            unsigned int bf[4][2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm + mi * 16 + g;
                af[mi][0] = *reinterpret_cast<const unsigned int*>(&As[r * LDS + kk + t * 2]);
                af[mi][1] = *reinterpret_cast<const unsigned int*>(&As[(r + 8) * LDS + kk + t * 2]);
                af[mi][2] = *reinterpret_cast<const unsigned int*>(&As[r * LDS + kk + t * 2 + 8]);
                af[mi][3] = *reinterpret_cast<const unsigned int*>(&As[(r + 8) * LDS + kk + t * 2 + 8]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int n = wn + ni * 8 + g;
                bf[ni][0] = *reinterpret_cast<const unsigned int*>(&Bs[n * LDS + kk + t * 2]);
                bf[ni][1] = *reinterpret_cast<const unsigned int*>(&Bs[n * LDS + kk + t * 2 + 8]);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    float* c = acc[mi][ni];
                    if (IS_BF16) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                              "r"(bf[ni][0]), "r"(bf[ni][1]));
                    } else {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]),
                              "r"(bf[ni][0]), "r"(bf[ni][1]));
                    }
                }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r = row0 + wm + mi * 16 + g;
            const int c = col0 + wn + ni * 8 + t * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rr = r + h * 8;
                if (rr >= M) continue;
                if (c < K) out[(long)rr * K + c] = static_cast<T>(acc[mi][ni][h * 2]);
                if (c + 1 < K) out[(long)rr * K + c + 1] = static_cast<T>(acc[mi][ni][h * 2 + 1]);
            }
        }
"""


def quantized_matmul_tiled(scales, biases, a, b, use_split_k=False, stream=None):
    import mlx.core as mx

    M = a.shape[0]
    K = b.shape[0]
    return launch(
        kernel("tl_qmm_mma_w4a16_g128", ["scales", "biases", "a", "b"], ["out"], _TILED_SRC),
        [scales, biases, a, b], ((K + 63) // 64, (M + 63) // 64), (128,),
        [(M, K)], [a.dtype], template=[("T", a.dtype), ("IS_BF16", int(a.dtype == mx.bfloat16))],
        stream=stream,
    )[0]


__all__ = ["quantized_matmul_tiled"]
