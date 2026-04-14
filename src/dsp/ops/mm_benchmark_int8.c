/**
 * Int8 vs F16 HMX micro-benchmark v2.
 * Measures: (1) tile load only (2) consume only (3) combined
 * Also tests different accumulator conversion modes for INT8.
 */

#include <stdint.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/utils.h"

#include "HAP_perf.h"

int benchmark_int8_vs_f16(float *result_buf, int K, int n_iters) {
    if (K < 128 || K % 128 != 0 || n_iters <= 0) return -1;

    uint8_t *vtcm = (uint8_t *)vtcm_manager_get_vtcm_base();
    uint8_t *ptr = vtcm;

    int n_fp16_tiles = K / 32;
    int n_int8_tiles = K / 128;

    // F16 buffers
    __fp16  *act_fp16 = (__fp16 *)vtcm_seq_alloc(&ptr, n_fp16_tiles * HMX_FP16_TILE_SIZE);
    __fp16  *wt_fp16  = (__fp16 *)vtcm_seq_alloc(&ptr, n_fp16_tiles * HMX_FP16_TILE_SIZE);
    __fp16  *out_fp16 = (__fp16 *)vtcm_seq_alloc(&ptr, HMX_FP16_TILE_SIZE);

    // Int8 buffers
    uint8_t *act_u8  = (uint8_t *)vtcm_seq_alloc(&ptr, n_int8_tiles * HMX_INT8_TILE_SIZE);
    int8_t  *wt_i8   = (int8_t  *)vtcm_seq_alloc(&ptr, n_int8_tiles * HMX_INT8_TILE_SIZE);
    __fp16  *out_i8  = (__fp16  *)vtcm_seq_alloc(&ptr, HMX_FP16_TILE_SIZE);
    // Extra output buffers for different acc modes
    __fp16  *out_i8_m0 = (__fp16 *)vtcm_seq_alloc(&ptr, HMX_FP16_TILE_SIZE);

    uint8_t *scales = (uint8_t *)vtcm_seq_alloc(&ptr, 256);

    // Fill with non-zero data
    memset(act_fp16, 0x3c, n_fp16_tiles * HMX_FP16_TILE_SIZE);
    memset(wt_fp16,  0x3c, n_fp16_tiles * HMX_FP16_TILE_SIZE);
    memset(act_u8,   64,   n_int8_tiles * HMX_INT8_TILE_SIZE);
    memset(wt_i8,    1,    n_int8_tiles * HMX_INT8_TILE_SIZE);

    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));  // 1.0

    hmx_manager_enable_execution();
    hmx_set_output_scales(scales);

    // Warmup
    for (int i = 0; i < 50; i++) {
        hmx_load_tiles_fp16(act_fp16, wt_fp16, n_fp16_tiles);
        hmx_consume_accumulator_fp16(out_fp16);
    }
    for (int i = 0; i < 50; i++) {
        hmx_load_tiles_int8(act_u8, wt_i8, n_int8_tiles);
        hmx_consume_accumulator_int8_to_fp16(out_i8);
    }

    // Test 1: FP16 combined (load + consume)
    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int i = 0; i < n_iters; i++) {
        hmx_load_tiles_fp16(act_fp16, wt_fp16, n_fp16_tiles);
        hmx_consume_accumulator_fp16(out_fp16);
    }
    int64_t t1 = HAP_perf_get_qtimer_count();
    int64_t fp16_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    // Test 2: INT8 combined (load + consume with acc mode 2)
    int64_t t2 = HAP_perf_get_qtimer_count();
    for (int i = 0; i < n_iters; i++) {
        hmx_load_tiles_int8(act_u8, wt_i8, n_int8_tiles);
        hmx_consume_accumulator_int8_to_fp16(out_i8);
    }
    int64_t t3 = HAP_perf_get_qtimer_count();
    int64_t int8_us = HAP_perf_qtimer_count_to_us(t3 - t2);

    // Test 3: FP16 load-only (no consume, accumulate multiple times)
    int64_t t4 = HAP_perf_get_qtimer_count();
    for (int i = 0; i < n_iters; i++) {
        hmx_load_tiles_fp16(act_fp16, wt_fp16, n_fp16_tiles);
    }
    int64_t t5 = HAP_perf_get_qtimer_count();
    hmx_consume_accumulator_fp16(out_fp16);  // consume once at end
    int64_t fp16_load_us = HAP_perf_qtimer_count_to_us(t5 - t4);

    // Test 4: INT8 load-only
    int64_t t6 = HAP_perf_get_qtimer_count();
    for (int i = 0; i < n_iters; i++) {
        hmx_load_tiles_int8(act_u8, wt_i8, n_int8_tiles);
    }
    int64_t t7 = HAP_perf_get_qtimer_count();
    hmx_consume_accumulator_int8_to_fp16(out_i8);
    int64_t int8_load_us = HAP_perf_qtimer_count_to_us(t7 - t6);

    // Test 5: INT8 with acc mode 0 (might be different conversion)
    int64_t t8 = HAP_perf_get_qtimer_count();
    for (int i = 0; i < n_iters; i++) {
        hmx_load_tiles_int8(act_u8, wt_i8, n_int8_tiles);
        // Try acc mode 0 instead of 2
        asm volatile(
            "cvt.hf = acc(%0)\n"
            "mxmem(%1, %2) = cvt\n" ::"r"(0),
            "r"(out_i8_m0), "r"(0)
            : "memory");
    }
    int64_t t9 = HAP_perf_get_qtimer_count();
    int64_t int8_m0_us = HAP_perf_qtimer_count_to_us(t9 - t8);

    hmx_manager_disable_execution();

    // Results: 8 floats
    result_buf[0] = (float)fp16_us;       // FP16 combined
    result_buf[1] = (float)int8_us;       // INT8 combined (acc mode 2)
    result_buf[2] = (int8_us > 0) ? (float)fp16_us / (float)int8_us : 0.0f;  // speedup
    result_buf[3] = (float)K;
    result_buf[4] = (float)fp16_load_us;  // FP16 load-only
    result_buf[5] = (float)int8_load_us;  // INT8 load-only
    result_buf[6] = (float)int8_m0_us;    // INT8 with acc mode 0
    result_buf[7] = (float)n_iters;

    return 0;
}
