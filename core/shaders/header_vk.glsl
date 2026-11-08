#version 450
// Vulkan 1.0-compatible (SPIR-V 1.0). Parameters arrive as push constants.
layout(push_constant) uniform Push {
  uint m;
  uint k;
  uint n;
  uint mode;
  float alpha;
  uint pad0;
  uint pad1;
  uint pad2;
} PC;
#define P_M PC.m
#define P_K PC.k
#define P_N PC.n
#define P_MODE PC.mode
#define P_ALPHA PC.alpha
