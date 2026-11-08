// C[m,n] = A[m,k] * B[k,n]; mode 1 additionally applies bias[n] and ReLU (dense layer).
// 16x16 shared-memory tiling, bounds-checked for arbitrary m,k,n.
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 1) readonly buffer BufB { float B[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };
layout(std430, binding = 3) readonly buffer BufBias { float bias[]; };
shared float As[16][16];
shared float Bs[16][16];

void main() {
  uint row = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  uint lr = gl_LocalInvocationID.y;
  uint lc = gl_LocalInvocationID.x;
  float acc = 0.0;
  uint tiles = (P_K + 15u) / 16u;
  for (uint t = 0u; t < tiles; ++t) {
    uint ak = t * 16u + lc;
    uint bk = t * 16u + lr;
    As[lr][lc] = (row < P_M && ak < P_K) ? A[row * P_K + ak] : 0.0;
    Bs[lr][lc] = (bk < P_K && col < P_N) ? B[bk * P_N + col] : 0.0;
    barrier();
    for (uint i = 0u; i < 16u; ++i) acc += As[lr][i] * Bs[i][lc];
    barrier();
  }
  if (row < P_M && col < P_N) {
    float v = acc;
    if (P_MODE == 1u) v = max(v + bias[col], 0.0);
    C[row * P_N + col] = v;
  }
}
