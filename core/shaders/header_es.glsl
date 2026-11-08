#version 310 es
// OpenGL ES 3.1 compute. Parameters arrive as uniforms.
precision highp float;
precision highp int;
uniform uint u_m;
uniform uint u_k;
uniform uint u_n;
uniform uint u_mode;
uniform float u_alpha;
#define P_M u_m
#define P_K u_k
#define P_N u_n
#define P_MODE u_mode
#define P_ALPHA u_alpha
