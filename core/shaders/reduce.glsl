// One pass of a two-pass sum. Each workgroup grid-strides over its share of A[0..P_N) and
// writes one partial sum to C[workgroup]. Pass 2 re-runs this with a single workgroup over
// the partials.
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer BufA { float A[]; };
layout(std430, binding = 2) writeonly buffer BufC { float C[]; };
shared float sm[256];

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint stride = gl_NumWorkGroups.x * 256u;
  float s = 0.0;
  for (uint i = gl_GlobalInvocationID.x; i < P_N; i += stride) s += A[i];
  sm[tid] = s;
  barrier();
  for (uint o = 128u; o > 0u; o >>= 1u) {
    if (tid < o) sm[tid] += sm[tid + o];
    barrier();
  }
  if (tid == 0u) C[gl_WorkGroupID.x] = sm[0];
}
