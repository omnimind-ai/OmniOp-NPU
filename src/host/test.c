#include <math.h>
#include <float.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "host/session.h"
#include "htp_ops.h"  // auto-generated
#include "message.h"
#include "op_reg.h"
#include "dsp/quants.h"

static inline int64_t get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

static inline int align_up(size_t size, size_t align) {
  return (size + align - 1) / align * align;
}

static inline double rand_01() {
  return ((double) rand()) / RAND_MAX;
}

static inline float rand_centered(float scale) {
  return (float) ((rand_01() * 2.0 - 1.0) * scale);
}

static inline size_t hmx_permuted_index(int row_k, int col_n, int k) {
  const int i0 = row_k / 32;
  const int i1 = row_k % 32;
  const int j0 = col_n / 32;
  const int j1 = col_n % 32;
  const int tile_idx = j0 * (k / 32) + i0;
  return (size_t) tile_idx * 1024 + (size_t) (i1 & ~1) * 32 + (size_t) j1 * 2 + (i1 & 1);
}

static void quantize_permuted_q4_0(const float *src, my_block_q4_0 *dst, float *deq, int n_elements) {
  const int n_blocks = n_elements / QK4_0;
  const int n_super  = n_blocks / 8;

  for (int s = 0; s < n_super; ++s) {
    uint8_t quants_unpacked[8 * QK4_0];

    for (int g = 0; g < 8; ++g) {
      const int base = (s * 8 + g) * QK4_0;
      float amax = 0.0f;
      for (int i = 0; i < QK4_0; ++i) {
        const float v = src[base + i];
        if (fabsf(v) > amax) {
          amax = fabsf(v);
        }
      }

      const float d = amax > 0.0f ? amax / 7.0f : 0.0f;
      const float id = d > 0.0f ? 1.0f / d : 0.0f;
      dst[s].scales[g] = (__fp16) d;

      for (int i = 0; i < QK4_0; ++i) {
        int q = (int) roundf(src[base + i] * id) + 8;
        if (q < 0) {
          q = 0;
        } else if (q > 15) {
          q = 15;
        }
        quants_unpacked[g * QK4_0 + i] = (uint8_t) q;
        deq[base + i] = (float) ((__fp16) ((q - 8) * (float) dst[s].scales[g]));
      }
    }

    for (int i = 0; i < 64; ++i) {
      dst[s].quants[i * 2 + 0] = (uint8_t) ((quants_unpacked[i + 128] << 4) | quants_unpacked[i + 0]);
      dst[s].quants[i * 2 + 1] = (uint8_t) ((quants_unpacked[i + 192] << 4) | quants_unpacked[i + 64]);
    }
  }
}

// assert p_buf, p_fd and size are always valid
int alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size) {
  void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
  if (!buf) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_alloc failed\n");
    return -1;
  }

  int fd = rpcmem_to_fd(buf);
  if (fd < 0) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_to_fd failed\n");
    return -1;
  }

  // map buffer to the DSP
  int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf, 0, size, FASTRPC_MAP_FD);
  if (err) {
    fprintf(stderr, "alloc_shared_mem_buf: fastrpc_mmap failed, err: %d\n", err);
    return -1;
  }

  *p_buf = buf;
  *p_fd  = fd;
  return 0;
}

void free_shared_mem_buf(void *buf, int fd, size_t size) {
  fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf, size);
  rpcmem_free(buf);
}

static void rms_norm_f32_ref(float *dst, const float *src, int ne0, int ne1, float eps) {
  for (int j = 0; j < ne1; ++j) {
    const float *x = src + j * ne0;
    float       *y = dst + j * ne0;

    float sum = 0;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i] * x[i];
    }

    float mean  = sum / ne0;
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = x[i] * scale;
    }

    printf("%s: sum: %.5f mean: %.5f scale: %.5f\n", __func__, sum, mean, scale);
  }
}

static void fill_rms_norm_sensitive_input(float *src, int ne0) {
  for (int i = 0; i < ne0; ++i) {
    int centered = (i % 17) - 8;
    src[i] = centered * 1e-5f;
  }
}

static void test_rms_norm_f32_rpc(remote_handle64 handle, int ne0) {
  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;
  const float eps = 1e-6f;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  fill_rms_norm_sensitive_input(src, ne0);

  int64_t t0             = get_time_us();
  err                    = htp_ops_rms_norm_f32(handle, fd_dst, 0, fd_src, 0, ne0, 1, eps);
  int64_t rpc_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 RPC took %ld us\n", rpc_elapsed_us);

  if (err != 0) {
    fprintf(stderr, "%s: RPC failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1, eps);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
  return;
}

static void test_rms_norm_f32_chan(void *chan, int ne0) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;
  const float eps = 1e-6f;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  fill_rms_norm_sensitive_input(src, ne0);

  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    struct OpComputeRequest compute_req = {
      .op = HTP_OPS_RMS_NORM_F32,
    };
    struct RmsNormF32Params params = {
      .dst = { .fd = fd_dst, .offset = 0, },
      .src = { .fd = fd_src, .offset = 0, },
      .ne0 = ne0,
      .ne1 = 1,
      .eps = eps,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct OpComputeRequest *) p = compute_req;
    p += sizeof(struct OpComputeRequest);
    *(struct RmsNormF32Params *) p = params;
    p += sizeof(struct RmsNormF32Params);
  }

  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    // usleep(10);
  }
  int64_t chan_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 CHAN took %ld us\n", chan_elapsed_us);

  err = message_header_get_request_ptr(msg, 0)->state;
  if (err != 0) {
    fprintf(stderr, "%s: CHAN failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1, eps);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

  // extra test: trigger DSP-side mapping reclaimation
  // fprintf(stderr, "manually unmap fd %d, %d\n", fd_dst, fd_src);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_dst, NULL, 0);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_src, NULL, 0);
  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    struct RpcmemMapRequest map_req = {
      .n_puts = 2,
      .n_gets = 0,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(map_req) + 2 * sizeof(int);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct RpcmemMapRequest *) p = map_req;
    p += sizeof(struct RpcmemMapRequest);

    // fill in fd data
    *(int *) p = fd_dst;
    p += sizeof(int);
    *(int *) p = fd_src;
    p += sizeof(int);
  }

  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
}

static int test_mat_mul_chan_shape(void *chan, int m, int k, int n) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *activation = NULL, *output = NULL;
  __fp16 *weight = NULL;
  float *weight_ref = NULL, *output_ref = NULL, *output_mix = NULL;
  __fp16 *output_f16 = NULL;

  int output_fd = -1, activation_fd = -1, weight_fd = -1;

  int passed = 0;

  if (alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float))) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float))) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16))) {
    goto end;
  }

  weight_ref = (float *) malloc(n * k * sizeof(float));
  output_ref = (float *) malloc(m * n * sizeof(float));
  output_f16 = (__fp16 *) malloc(m * n * sizeof(__fp16));
  output_mix = (float *) malloc(m * n * sizeof(float));
  if (!weight_ref || !output_ref || !output_f16 || !output_mix) {
    fprintf(stderr, "%s: host malloc failed for m=%d k=%d n=%d\n", __func__, m, k, n);
    goto end;
  }

  memset(output_ref, 0, m * n * sizeof(float));
  memset(output_f16, 0, m * n * sizeof(__fp16));
  memset(output_mix, 0, m * n * sizeof(float));

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = rand_01();
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = rand_01();

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
      weight_ref[i * n + j] = x;
    }
  }

  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  };
  struct MatMulParams params = {
    .output     = { .fd = output_fd,     .offset = 0, },
    .activation = { .fd = activation_fd, .offset = 0, },
    .weight     = { .fd = weight_fd,     .offset = 0, },
    .m          = m,
    .k          = k,
    .n          = n,
    .skip_scale = 0,
  };

  struct RequestHeader map_req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_RPCMEM_MAP,
  };
  struct RpcmemMapRequest map_req = {
    .n_puts = 3,
    .n_gets = 0,
  };

  size_t op_req_size  = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
  size_t map_req_size = sizeof(map_req_hdr) + sizeof(map_req) + 3 * sizeof(int32_t);
  msg->state.d        = 0;
  msg->n_reqs         = 2;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + op_req_size;
  msg->req_offsets[2] = msg->req_offsets[1] + map_req_size;

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(struct OpComputeRequest);
  *(struct MatMulParams *) p = params;

  p = (uint8_t *) message_header_get_request_ptr(msg, 1);
  *(struct RequestHeader *) p = map_req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct RpcmemMapRequest *) p = map_req;
  p += sizeof(struct RpcmemMapRequest);
  *(int32_t *) p = output_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = activation_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = weight_fd;

  __sync_synchronize();
  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(1);
  }
  int64_t elapsed_us = get_time_us() - t0;

  int err = message_header_get_request_ptr(msg, 0)->state;
  msg->state.d = 0;
  if (err != 0) {
    fprintf(stderr, "%s: channel op failed with %x for m=%d k=%d n=%d\n", __func__, err, m, k, n);
    goto end;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        output_ref[i * n + j] += activation[i * k + l] * weight_ref[l * n + j];
        output_f16[i * n + j] += (__fp16)(((__fp16) activation[i * k + l]) * ((__fp16) weight_ref[l * n + j]));
        output_mix[i * n + j] += (float)((__fp16) activation[i * k + l] * ((__fp16) weight_ref[l * n + j]));
      }
    }
  }

  int   n_failed = 0;
  float max_abs  = 0.0f;
  for (int i = 0; i < m * n; ++i) {
    float diff = fabsf(output[i] - output_mix[i]);
    if (diff > max_abs) {
      max_abs = diff;
    }
    if (diff > 2.0f) {
      if (n_failed < 16) {
        fprintf(stderr, "#%d hmx=%g mix=%g f32=%g diff=%g\n", i, output[i], output_mix[i], output_ref[i], diff);
      }
      n_failed++;
    }
  }
  passed = (n_failed == 0);
  fprintf(stderr, "mat_mul_w16a32_chan m=%d k=%d n=%d elapsed_us=%ld max_abs=%g failed=%d %s\n",
          m, k, n, elapsed_us, max_abs, n_failed,
          passed ? "passed" : "failed");

end:
  if (weight_ref) {
    free(weight_ref);
  }
  if (output_ref) {
    free(output_ref);
  }
  if (output_f16) {
    free(output_f16);
  }
  if (output_mix) {
    free(output_mix);
  }

  if (output) {
    free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  }
  if (activation) {
    free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  }
  if (weight) {
    free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
  }

  return passed ? 0 : 1;
}

static void test_mat_mul_chan(void *chan) {
  int failed = 0;

  failed += test_mat_mul_chan_shape(chan, 1, 1024, 1024);
  failed += test_mat_mul_chan_shape(chan, 1, 1536, 1024);
  failed += test_mat_mul_chan_shape(chan, 64, 1536, 1024);
  failed += test_mat_mul_chan_shape(chan, 1, 1536, 256);
  failed += test_mat_mul_chan_shape(chan, 64, 1536, 256);
  failed += test_mat_mul_chan_shape(chan, 1, 256, 1536);
  failed += test_mat_mul_chan_shape(chan, 64, 256, 1536);

  fprintf(stderr, failed == 0 ? "%s passed\n" : "%s failed: %d shapes failed\n", __func__, failed);
}

static void flash_attn_ref_f32(float *out, const float *q, const __fp16 *k, const __fp16 *v, const __fp16 *mask,
                               int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim, float scale,
                               int mask_stride) {
  const int gqa_factor = n_heads / n_kv_heads;
  const int q_stride   = n_heads * head_dim;
  const int kv_stride  = n_kv_heads * head_dim;

  for (int iq = 0; iq < qo_len; ++iq) {
    for (int ih = 0; ih < n_heads; ++ih) {
      const int ikh = ih / gqa_factor;

      float max_score = -FLT_MAX;
      for (int ikv = 0; ikv < kv_len; ++ikv) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score += q[iq * q_stride + ih * head_dim + d] *
                   (float) k[ikv * kv_stride + ikh * head_dim + d];
        }
        score *= scale;
        if (mask) {
          score += (float) mask[iq * mask_stride + ikv];
        }
        if (score > max_score) {
          max_score = score;
        }
      }

      float denom = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        out[iq * q_stride + ih * head_dim + d] = 0.0f;
      }

      for (int ikv = 0; ikv < kv_len; ++ikv) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score += q[iq * q_stride + ih * head_dim + d] *
                   (float) k[ikv * kv_stride + ikh * head_dim + d];
        }
        score *= scale;
        if (mask) {
          score += (float) mask[iq * mask_stride + ikv];
        }

        const float p = expf(score - max_score);
        denom += p;
        for (int d = 0; d < head_dim; ++d) {
          out[iq * q_stride + ih * head_dim + d] += p * (float) v[ikv * kv_stride + ikh * head_dim + d];
        }
      }

      const float inv_denom = 1.0f / denom;
      for (int d = 0; d < head_dim; ++d) {
        out[iq * q_stride + ih * head_dim + d] *= inv_denom;
      }
    }
  }
}

static int test_flash_attn_chan_shape(void *chan, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                      int use_mask) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float  *out = NULL, *q = NULL, *ref = NULL;
  __fp16 *k = NULL, *v = NULL, *mask = NULL;
  int out_fd = -1, q_fd = -1, k_fd = -1, v_fd = -1, mask_fd = -1;

  int passed = 0;
  const int mask_stride = align_up(kv_len, 64);
  const float scale = 1.0f / sqrtf((float) head_dim);

  const size_t qo_size = align_up((size_t) qo_len * n_heads * head_dim * sizeof(float), 128);
  const size_t kv_size = align_up((size_t) kv_len * n_kv_heads * head_dim * sizeof(__fp16), 128);
  const size_t mask_size = use_mask ? align_up((size_t) qo_len * mask_stride * sizeof(__fp16), 128) : 0;

  if (alloc_shared_mem_buf((void **) &out, &out_fd, qo_size)) goto end;
  if (alloc_shared_mem_buf((void **) &q, &q_fd, qo_size)) goto end;
  if (alloc_shared_mem_buf((void **) &k, &k_fd, kv_size)) goto end;
  if (alloc_shared_mem_buf((void **) &v, &v_fd, kv_size)) goto end;
  if (use_mask && alloc_shared_mem_buf((void **) &mask, &mask_fd, mask_size)) goto end;

  ref = (float *) malloc(qo_size);
  if (!ref) {
    fprintf(stderr, "%s: malloc failed\n", __func__);
    goto end;
  }

  memset(out, 0, qo_size);
  memset(ref, 0, qo_size);
  for (int i = 0; i < qo_len * n_heads * head_dim; ++i) {
    q[i] = rand_centered(0.2f);
  }
  for (int i = 0; i < kv_len * n_kv_heads * head_dim; ++i) {
    k[i] = (__fp16) rand_centered(0.2f);
    v[i] = (__fp16) rand_centered(0.2f);
  }
  const int causal_mask = getenv("HTP_TEST_FA_CAUSAL") ? atoi(getenv("HTP_TEST_FA_CAUSAL")) : 0;
  if (use_mask) {
    const int causal_base = kv_len > qo_len ? kv_len - qo_len : 0;
    for (int r = 0; r < qo_len; ++r) {
      for (int c = 0; c < mask_stride; ++c) {
        const int in_kv = c < kv_len;
        const int allowed = !causal_mask || c <= causal_base + r;
        mask[r * mask_stride + c] = (in_kv && allowed) ? (__fp16) 0.0f : (__fp16) -INFINITY;
      }
    }
  }

  flash_attn_ref_f32(ref, q, k, v, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, scale, mask_stride);

  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  };
  struct FlashAttnParams params = {
    .o          = { .fd = out_fd,  .offset = 0, },
    .q          = { .fd = q_fd,    .offset = 0, },
    .k          = { .fd = k_fd,    .offset = 0, },
    .v          = { .fd = v_fd,    .offset = 0, },
    .mask       = { .fd = use_mask ? mask_fd : -1, .offset = 0, },
    .scale      = scale,
    .mask_stride = mask_stride,
    .qo_len     = qo_len,
    .kv_len     = kv_len,
    .n_heads    = n_heads,
    .n_kv_heads = n_kv_heads,
    .head_dim   = head_dim,
  };

  struct RequestHeader map_req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_RPCMEM_MAP,
  };
  struct RpcmemMapRequest map_req = {
    .n_puts = use_mask ? 5 : 4,
    .n_gets = 0,
  };

  const size_t op_req_size = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
  const size_t map_req_size = sizeof(map_req_hdr) + sizeof(map_req) + map_req.n_puts * sizeof(int32_t);
  msg->state.d        = 0;
  msg->n_reqs         = 2;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + op_req_size;
  msg->req_offsets[2] = msg->req_offsets[1] + map_req_size;

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(struct OpComputeRequest);
  *(struct FlashAttnParams *) p = params;

  p = (uint8_t *) message_header_get_request_ptr(msg, 1);
  *(struct RequestHeader *) p = map_req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct RpcmemMapRequest *) p = map_req;
  p += sizeof(struct RpcmemMapRequest);
  *(int32_t *) p = out_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = q_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = k_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = v_fd;
  p += sizeof(int32_t);
  if (use_mask) {
    *(int32_t *) p = mask_fd;
  }

  __sync_synchronize();
  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    if (get_time_us() - t0 > 5000000) {
      fprintf(stderr, "%s: channel timeout for qo=%d kv=%d h=%d hk=%d d=%d mask=%d\n", __func__, qo_len, kv_len,
              n_heads, n_kv_heads, head_dim, use_mask);
      goto end;
    }
    usleep(1);
  }
  int64_t elapsed_us = get_time_us() - t0;

  int err = message_header_get_request_ptr(msg, 0)->state;
  msg->state.d = 0;
  if (err != 0) {
    if (!use_mask) {
      fprintf(stderr,
              "flash_attn_chan qo=%d kv=%d h=%d hk=%d d=%d mask=%d elapsed_us=%ld expected unsupported err=%x\n",
              qo_len, kv_len, n_heads, n_kv_heads, head_dim, use_mask, elapsed_us, err);
      passed = 1;
      goto end;
    }
    fprintf(stderr, "%s: channel op failed with %x for qo=%d kv=%d h=%d hk=%d d=%d mask=%d\n", __func__, err,
            qo_len, kv_len, n_heads, n_kv_heads, head_dim, use_mask);
    goto end;
  }

  double sum_sq = 0.0;
  float max_abs = 0.0f;
  int n_failed = 0;
  const int n = qo_len * n_heads * head_dim;
  for (int i = 0; i < n; ++i) {
    const float diff = fabsf(out[i] - ref[i]);
    sum_sq += (double) diff * diff;
    if (diff > max_abs) max_abs = diff;
    if (diff > 5e-2f) {
      if (n_failed < 16) {
        fprintf(stderr, "fa mismatch #%d dsp=%g ref=%g diff=%g\n", i, out[i], ref[i], diff);
      }
      n_failed++;
    }
  }
  const double rmse = sqrt(sum_sq / n);
  passed = n_failed == 0 && rmse < 1e-2 && max_abs < 5e-2f;
  fprintf(stderr,
          "flash_attn_chan qo=%d kv=%d h=%d hk=%d d=%d mask=%d elapsed_us=%ld rmse=%g max_abs=%g failed=%d %s\n",
          qo_len, kv_len, n_heads, n_kv_heads, head_dim, use_mask, elapsed_us, rmse, max_abs, n_failed,
          passed ? "passed" : "failed");

end:
  if (ref) free(ref);
  if (out) free_shared_mem_buf(out, out_fd, qo_size);
  if (q) free_shared_mem_buf(q, q_fd, qo_size);
  if (k) free_shared_mem_buf(k, k_fd, kv_size);
  if (v) free_shared_mem_buf(v, v_fd, kv_size);
  if (mask) free_shared_mem_buf(mask, mask_fd, mask_size);

  return passed ? 0 : 1;
}

static int test_flash_attn_chan(void *chan) {
  int failed = 0;

  const char *qo_env = getenv("HTP_TEST_FA_QO");
  const char *kv_env = getenv("HTP_TEST_FA_KV");
  if (qo_env || kv_env) {
    const int qo_len = qo_env ? atoi(qo_env) : 1;
    const int kv_len = kv_env ? atoi(kv_env) : 64;
    const int n_heads = getenv("HTP_TEST_FA_H") ? atoi(getenv("HTP_TEST_FA_H")) : 16;
    const int n_kv_heads = getenv("HTP_TEST_FA_HKV") ? atoi(getenv("HTP_TEST_FA_HKV")) : 4;
    const int head_dim = getenv("HTP_TEST_FA_D") ? atoi(getenv("HTP_TEST_FA_D")) : 256;
    const int use_mask = getenv("HTP_TEST_FA_MASK") ? atoi(getenv("HTP_TEST_FA_MASK")) : 1;
    failed += test_flash_attn_chan_shape(chan, qo_len, kv_len, n_heads, n_kv_heads, head_dim, use_mask);
    fprintf(stderr, failed == 0 ? "%s passed\n" : "%s failed: %d shapes failed\n", __func__, failed);
    return failed;
  }

  failed += test_flash_attn_chan_shape(chan, 1, 64, 16, 4, 256, 1);
  failed += test_flash_attn_chan_shape(chan, 8, 128, 16, 4, 256, 1);
  failed += test_flash_attn_chan_shape(chan, 64, 128, 16, 4, 256, 1);
  failed += test_flash_attn_chan_shape(chan, 8, 128, 16, 4, 256, 0);

  fprintf(stderr, failed == 0 ? "%s passed\n" : "%s failed: %d shapes failed\n", __func__, failed);
  return failed;
}

static int test_quant_mat_mul_chan_shape(void *chan, int m, int k, int n) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *activation = NULL, *output = NULL, *weight_perm = NULL, *weight_deq_perm = NULL, *output_ref = NULL;
  my_block_q4_0 *weight_q4 = NULL;
  int output_fd = -1, activation_fd = -1, weight_fd = -1;
  int passed = 0;

  if (k % 256 != 0 || n % 32 != 0) {
    fprintf(stderr, "%s: k must be multiple of 256 and n multiple of 32, got m=%d k=%d n=%d\n", __func__, m, k, n);
    goto end;
  }

  const size_t output_size = (size_t) m * n * sizeof(float);
  const size_t activation_size = (size_t) m * k * sizeof(float);
  const size_t permuted_weight_elems = (size_t) n * k;
  const size_t weight_size = permuted_weight_elems / QK_K * sizeof(my_block_q4_0);

  if (alloc_shared_mem_buf((void **) &output, &output_fd, output_size)) goto end;
  if (alloc_shared_mem_buf((void **) &activation, &activation_fd, activation_size)) goto end;
  if (alloc_shared_mem_buf((void **) &weight_q4, &weight_fd, weight_size)) goto end;

  weight_perm = (float *) malloc(permuted_weight_elems * sizeof(float));
  weight_deq_perm = (float *) malloc(permuted_weight_elems * sizeof(float));
  output_ref = (float *) calloc((size_t) m * n, sizeof(float));
  if (!weight_perm || !weight_deq_perm || !output_ref) {
    fprintf(stderr, "%s: host malloc failed for m=%d k=%d n=%d\n", __func__, m, k, n);
    goto end;
  }

  memset(output, 0, output_size);
  for (int i = 0; i < m * k; ++i) {
    activation[i] = rand_centered(0.25f);
  }
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < k; ++row) {
      const float v = rand_centered(0.2f);
      weight_perm[hmx_permuted_index(row, col, k)] = v;
    }
  }

  quantize_permuted_q4_0(weight_perm, weight_q4, weight_deq_perm, (int) permuted_weight_elems);

  for (int row_m = 0; row_m < m; ++row_m) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int row_k = 0; row_k < k; ++row_k) {
        acc += activation[row_m * k + row_k] * weight_deq_perm[hmx_permuted_index(row_k, col, k)];
      }
      output_ref[row_m * n + col] = acc;
    }
  }

  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  };
  struct MatMulParams params = {
    .output     = { .fd = output_fd,     .offset = 0, },
    .activation = { .fd = activation_fd, .offset = 0, },
    .weight     = { .fd = weight_fd,     .offset = 0, },
    .m          = m,
    .k          = k,
    .n          = n,
    .skip_scale = 0,
  };

  struct RequestHeader map_req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_RPCMEM_MAP,
  };
  struct RpcmemMapRequest map_req = {
    .n_puts = 3,
    .n_gets = 0,
  };

  const size_t op_req_size  = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
  const size_t map_req_size = sizeof(map_req_hdr) + sizeof(map_req) + 3 * sizeof(int32_t);
  msg->state.d        = 0;
  msg->n_reqs         = 2;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + op_req_size;
  msg->req_offsets[2] = msg->req_offsets[1] + map_req_size;

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(struct OpComputeRequest);
  *(struct MatMulParams *) p = params;

  p = (uint8_t *) message_header_get_request_ptr(msg, 1);
  *(struct RequestHeader *) p = map_req_hdr;
  p += sizeof(struct RequestHeader);
  *(struct RpcmemMapRequest *) p = map_req;
  p += sizeof(struct RpcmemMapRequest);
  *(int32_t *) p = output_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = activation_fd;
  p += sizeof(int32_t);
  *(int32_t *) p = weight_fd;

  __sync_synchronize();
  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    if (get_time_us() - t0 > 10000000) {
      fprintf(stderr, "%s: channel timeout for q4 m=%d k=%d n=%d\n", __func__, m, k, n);
      goto end;
    }
    usleep(1);
  }
  int64_t elapsed_us = get_time_us() - t0;

  int err = message_header_get_request_ptr(msg, 0)->state;
  msg->state.d = 0;
  if (err != 0) {
    fprintf(stderr, "%s: channel op failed with %x for q4 m=%d k=%d n=%d\n", __func__, err, m, k, n);
    goto end;
  }

  double sum_sq = 0.0;
  float max_abs = 0.0f;
  int n_failed = 0;
  for (int i = 0; i < m * n; ++i) {
    const float diff = fabsf(output[i] - output_ref[i]);
    sum_sq += (double) diff * diff;
    if (diff > max_abs) max_abs = diff;
    if (diff > 2.0f) {
      if (n_failed < 16) {
        fprintf(stderr, "q4 mm mismatch #%d dsp=%g ref=%g diff=%g\n", i, output[i], output_ref[i], diff);
      }
      n_failed++;
    }
  }
  const double rmse = sqrt(sum_sq / ((double) m * n));
  passed = n_failed == 0 && rmse < 0.25 && max_abs < 2.0f;
  fprintf(stderr,
          "quant_mat_mul_q4_chan m=%d k=%d n=%d elapsed_us=%ld rmse=%g max_abs=%g failed=%d %s\n",
          m, k, n, elapsed_us, rmse, max_abs, n_failed, passed ? "passed" : "failed");

end:
  if (weight_perm) free(weight_perm);
  if (weight_deq_perm) free(weight_deq_perm);
  if (output_ref) free(output_ref);
  if (output) free_shared_mem_buf(output, output_fd, output_size);
  if (activation) free_shared_mem_buf(activation, activation_fd, activation_size);
  if (weight_q4) free_shared_mem_buf(weight_q4, weight_fd, weight_size);

  return passed ? 0 : 1;
}

static int test_quant_mat_mul_chan(void *chan) {
  int failed = 0;

  const char *m_env = getenv("HTP_TEST_MM_M");
  const char *k_env = getenv("HTP_TEST_MM_K");
  const char *n_env = getenv("HTP_TEST_MM_N");
  if (m_env || k_env || n_env) {
    const int m = m_env ? atoi(m_env) : 128;
    const int k = k_env ? atoi(k_env) : 256;
    const int n = n_env ? atoi(n_env) : 1536;
    failed += test_quant_mat_mul_chan_shape(chan, m, k, n);
    fprintf(stderr, failed == 0 ? "%s passed\n" : "%s failed: %d shapes failed\n", __func__, failed);
    return failed;
  }

  failed += test_quant_mat_mul_chan_shape(chan, 96, 256, 1536);
  failed += test_quant_mat_mul_chan_shape(chan, 128, 256, 1536);
  failed += test_quant_mat_mul_chan_shape(chan, 160, 256, 1536);
  failed += test_quant_mat_mul_chan_shape(chan, 128, 1536, 1024);
  failed += test_quant_mat_mul_chan_shape(chan, 128, 1536, 4096);

  fprintf(stderr, failed == 0 ? "%s passed\n" : "%s failed: %d shapes failed\n", __func__, failed);
  return failed;
}

int main(int argc, char **argv) {
  int err = open_dsp_session(CDSP_DOMAIN_ID, 1);
  if (err != 0) {
    fprintf(stderr, "Open DSP session failed\n");
    return 1;
  }

  init_htp_backend();

  if (getenv("HTP_TEST_MATMUL")) {
    void        *chan;
    int          chan_fd;
    const size_t max_msg_size = 4096;

    err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
      close_dsp_session();
      return 1;
    }

    err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Create channel failed\n");
      free_shared_mem_buf(chan, chan_fd, max_msg_size);
      close_dsp_session();
      return 1;
    }

    int failed = 0;
    if (getenv("HTP_TEST_QUANT_MATMUL")) {
      failed = test_quant_mat_mul_chan(chan);
    } else {
      test_mat_mul_chan(chan);
    }

    htp_ops_destroy_channel(get_global_handle());
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
    close_dsp_session();
    return failed == 0 ? 0 : 1;
  }

  if (getenv("HTP_TEST_RMS_NORM")) {
    test_rms_norm_f32_rpc(get_global_handle(), 60000);

    void        *chan;
    int          chan_fd;
    const size_t max_msg_size = 4096;

    err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
      close_dsp_session();
      return 1;
    }

    err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Create channel failed\n");
      free_shared_mem_buf(chan, chan_fd, max_msg_size);
      close_dsp_session();
      return 1;
    }

    test_rms_norm_f32_chan(chan, 60000);

    htp_ops_destroy_channel(get_global_handle());
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
    close_dsp_session();
    return 0;
  }

  if (getenv("HTP_TEST_FLASH_ATTN")) {
    void        *chan;
    int          chan_fd;
    const size_t max_msg_size = 4096;

    err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
      close_dsp_session();
      return 1;
    }

    err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
    if (err) {
      fprintf(stderr, "Create channel failed\n");
      free_shared_mem_buf(chan, chan_fd, max_msg_size);
      close_dsp_session();
      return 1;
    }

    int failed = test_flash_attn_chan(chan);

    htp_ops_destroy_channel(get_global_handle());
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
    close_dsp_session();
    return failed == 0 ? 0 : 1;
  }

  htp_ops_test_ops(get_global_handle());

  close_dsp_session();
  return 0;
}
