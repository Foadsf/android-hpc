// STREAM triad: C[i] = A[i] + alpha * B[i]. Grid-stride so the dispatch stays within the
// guaranteed 65535 workgroup limit for any length.
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 1) readonly buffer BufB { float B[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };

void main() {
  uint stride = gl_NumWorkGroups.x * 256u;
  for (uint i = gl_GlobalInvocationID.x; i < P_N; i += stride) C[i] = A[i] + P_ALPHA * B[i];
}
