// 3x3 correlation over an image of P_M rows x P_N cols. Border pixels copy the input.
#ifndef WG2D
#define WG2D 16
#endif
layout(local_size_x = WG2D, local_size_y = WG2D) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 1) readonly buffer BufB { float W[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };

void main() {
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;
  if (x >= P_N || y >= P_M) return;
  if (x == 0u || y == 0u || x == P_N - 1u || y == P_M - 1u) {
    C[y * P_N + x] = A[y * P_N + x];
    return;
  }
  float s = 0.0;
  for (uint dy = 0u; dy < 3u; ++dy)
    for (uint dx = 0u; dx < 3u; ++dx)
      s += A[(y + dy - 1u) * P_N + (x + dx - 1u)] * W[dy * 3u + dx];
  C[y * P_N + x] = s;
}
