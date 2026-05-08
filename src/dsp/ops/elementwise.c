#include "dsp/hvx_internal.h"

int hvx_mul_f32(float *restrict dst, const float *restrict src0, const float *restrict src1, int ne0, int ne1,
                int src1_broadcast) {
  if (!dst || !src0 || !src1 || ne0 <= 0 || ne1 <= 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src0, VLEN) || !is_aligned(src1, VLEN)) {
    return -1;
  }

  const int n_vecs = ne0 / 32;
  const int tail   = ne0 & 31;

  for (int row = 0; row < ne1; ++row) {
    const float *row0 = src0 + row * ne0;
    const float *row1 = src1_broadcast ? src1 : src1 + row * ne0;
    float       *out  = dst + row * ne0;

    const HVX_Vector *v0 = (const HVX_Vector *) row0;
    const HVX_Vector *v1 = (const HVX_Vector *) row1;
    HVX_Vector       *vo = (HVX_Vector *) out;

    for (int i = 0; i < n_vecs; ++i) {
      vo[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0[i], v1[i]));
    }

    for (int i = ne0 - tail; i < ne0; ++i) {
      out[i] = row0[i] * row1[i];
    }
  }

  return 0;
}
