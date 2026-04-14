/**
 * Int8 HMX matmul for stage-wise quantization in DLLM.
 *
 * Early diffusion steps: Q4_0 weight expanded to int8 (no dequant to F16),
 * F32 activation quantized to uint8, then int8 × int8 HMX matmul.
 *
 * Compared to the F16 path:
 *   F16: Q4_0 → dequant → F16 tiles (32×32) → HMX F16×F16
 *   Int8: Q4_0 → expand → int8 tiles (32×128) → HMX uint8×int8
 *
 * Int8 tiles are 4× larger per tile (4096 bytes vs 2048 bytes), so the
 * reduction dimension is consumed 4× faster per HMX operation.
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_internal.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"

// Expand Q4_0 quants to signed int8.
// Each Q4_0 group has 32 elements packed as 16 bytes (2 elements per byte).
// The 4-bit values are in range [0, 15], we subtract 8 to get [-8, 7] (signed).
//
// Input:  my_block_q4_0 super block (8 groups, 256 elements total)
// Output: 256 × int8 values (no scale applied, raw integer)
// Also outputs: 8 × fp16 scales (for post-matmul rescaling)
static void expand_q4_0_super_block_to_int8(int8_t *restrict out_quants,
                                            __fp16 *restrict out_scales,
                                            const my_block_q4_0 *restrict src) {
    // Copy scales for later rescaling
    memcpy(out_scales, src->scales, 8 * sizeof(__fp16));

    const uint8_t *packed = src->quants;

    for (int g = 0; g < 8; g++) {
        // Each group: 16 bytes = 32 elements (lo 4-bit, hi 4-bit per byte)
        for (int j = 0; j < 16; j++) {
            uint8_t byte = packed[g * 16 + j];
            // Low nibble: bits [3:0], mapped to [-8, 7]
            out_quants[g * 32 + j]      = (int8_t)((byte & 0x0F) - 8);
            // High nibble: bits [7:4], mapped to [-8, 7]
            out_quants[g * 32 + j + 16] = (int8_t)((byte >> 4) - 8);
        }
    }
}

// Quantize F32 activation row to uint8 with per-row scale.
// Returns the scale factor (F32 → uint8 mapping).
//
// x_f32[k] → x_uint8[k] = clamp(round(x_f32[k] / scale + 128), 0, 255)
// scale = max(abs(x_f32)) / 127
static float quantize_f32_row_to_uint8(uint8_t *restrict out, const float *restrict src, int k) {
    // Find max absolute value
    float amax = 0.0f;
    for (int i = 0; i < k; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }

    if (amax < 1e-10f) {
        memset(out, 128, k);  // zero → 128 (offset binary)
        return 0.0f;
    }

    float scale = amax / 127.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < k; i++) {
        int v = (int)roundf(src[i] * inv_scale) + 128;
        out[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    return scale;
}

/**
 * Int8 HMX matmul: output[m, n] = activation[m, k] × weight^T[k, n]
 *
 * activation: F32 [m, k] — will be quantized to uint8 per row
 * weight: Q4_0 (HMX repacked) [n, k] — will be expanded to int8
 * output: F32 [m, n]
 *
 * The int8 matmul result is rescaled:
 *   output[i, j] = act_scale[i] × weight_scale[j/32] × hmx_result[i, j]
 *
 * This function is a simplified prototype. For production, the per-row
 * quantization and int8 tile formatting should be vectorized with HVX.
 */
int hmx_mat_mul_int8_q4_0(float *restrict dst, const float *restrict activation,
                           const uint8_t *restrict permuted_weight, int m, int k, int n,
                           enum ggml_type weight_type) {
    if (!dst || !activation || !permuted_weight || !m || !n || !k) return -1;
    if (k % 256 != 0 || n % 32 != 0) return -1;  // Q4_0 super block alignment
    if (weight_type != GGML_TYPE_Q4_0) return -1;  // only Q4_0 for now

    // For now, fall back to scalar computation as a correctness prototype.
    // TODO: implement proper int8 HMX tile operations once tile layout is validated.

    // Step 1: Expand all weight super blocks to int8 + collect scales
    const int n_super_blocks_per_col = k / 256;  // QK_K = 256
    const size_t super_block_size = sizeof(my_block_q4_0);

    // Allocate temporary buffers (should use VTCM in production)
    int8_t  *w_int8  = (int8_t *)  __builtin_alloca(n * k * sizeof(int8_t));
    __fp16  *w_scales = (__fp16 *) __builtin_alloca(n * (k / 32) * sizeof(__fp16));

    for (int col = 0; col < n; col++) {
        const my_block_q4_0 *sb = (const my_block_q4_0 *)(permuted_weight + col * n_super_blocks_per_col * super_block_size);
        for (int sb_idx = 0; sb_idx < n_super_blocks_per_col; sb_idx++) {
            expand_q4_0_super_block_to_int8(
                w_int8 + col * k + sb_idx * 256,
                w_scales + col * (k / 32) + sb_idx * 8,
                sb + sb_idx);
        }
    }

    // Step 2: Quantize activation rows to uint8
    uint8_t *a_uint8  = (uint8_t *) __builtin_alloca(m * k * sizeof(uint8_t));
    float   *a_scales = (float *)   __builtin_alloca(m * sizeof(float));

    for (int row = 0; row < m; row++) {
        a_scales[row] = quantize_f32_row_to_uint8(
            a_uint8 + row * k,
            activation + row * k,
            k);
    }

    // Step 3: Integer matmul (scalar reference)
    // TODO: Replace with HMX int8 tile operations
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; kk++) {
                // a_uint8 is offset binary: actual_value = (a_uint8 - 128) * a_scale
                // w_int8 is signed: actual_value = w_int8 * w_scale
                int a_val = (int)a_uint8[i * k + kk] - 128;
                int w_val = (int)w_int8[j * k + kk];
                acc += a_val * w_val;
            }

            // Rescale: acc * act_scale * weight_scales (per group of 32)
            // Simplified: use average weight scale for the whole column
            // TODO: per-group rescaling for accuracy
            float w_scale_avg = 0.0f;
            for (int g = 0; g < k / 32; g++) {
                w_scale_avg += (float)w_scales[j * (k / 32) + g];
            }
            w_scale_avg /= (k / 32);

            dst[i * n + j] = (float)acc * a_scales[i] * w_scale_avg;
        }
    }

    return 0;
}
