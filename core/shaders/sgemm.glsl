// C[m,n] = A[m,k] * B[k,n]; mode 1 additionally applies bias[n] and ReLU (dense layer).
// WG2D x WG2D shared-memory tiling (default 16; the GLES backend lowers it on drivers that
// allow fewer than 256 invocations per workgroup), bounds-checked for arbitrary m,k,n.
#ifndef WG2D
#define WG2D 16
#endif
layout(local_size_x = WG2D, local_size_y = WG2D) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 1) readonly buffer BufB { float B[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };
layout(std430, binding = 3) readonly buffer BufBias { float bias[]; };
shared float As[WG2D][WG2D];
shared float Bs[WG2D][WG2D];

void main() {
  uint row = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  uint lr = gl_LocalInvocationID.y;
  uint lc = gl_LocalInvocationID.x;
  float acc = 0.0;
  uint tiles = (P_K + uint(WG2D) - 1u) / uint(WG2D);
  for (uint t = 0u; t < tiles; ++t) {
    uint ak = t * uint(WG2D) + lc;
    uint bk = t * uint(WG2D) + lr;
    As[lr][lc] = (row < P_M && ak < P_K) ? A[row * P_K + ak] : 0.0;
    Bs[lr][lc] = (bk < P_K && col < P_N) ? B[bk * P_N + col] : 0.0;
    barrier();
    for (uint i = 0u; i < uint(WG2D); ++i) acc += As[lr][i] * Bs[i][lc];
    barrier();
  }
  if (row < P_M && col < P_N) {
    float v = acc;
    if (P_MODE == 1u) v = max(v + bias[col], 0.0);
    C[row * P_N + col] = v;
  }
}
