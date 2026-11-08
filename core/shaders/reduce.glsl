// One pass of a two-pass sum. Each workgroup grid-strides over its share of A[0..P_N) and
// writes one partial sum to C[workgroup]. Pass 2 re-runs this with a single workgroup over
// the partials. WG1D must be a power of two (default 256; the GLES backend may lower it).
#ifndef WG1D
#define WG1D 256
#endif
layout(local_size_x = WG1D) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };
shared float sm[WG1D];

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint stride = gl_NumWorkGroups.x * uint(WG1D);
  float s = 0.0;
  for (uint i = gl_GlobalInvocationID.x; i < P_N; i += stride) s += A[i];
  sm[tid] = s;
  barrier();
  for (uint o = uint(WG1D) / 2u; o > 0u; o >>= 1u) {
    if (tid < o) sm[tid] += sm[tid + o];
    barrier();
  }
  if (tid == 0u) C[gl_WorkGroupID.x] = sm[0];
}
