// SPDX-License-Identifier: Apache-2.0
// OpenGL ES 3.1 compute-shader backend over a headless EGL pbuffer context (Android only).
// Shader text is core/shaders/header_es.glsl + the shared kernel bodies.
#include "backend_util.hpp"

#if defined(__ANDROID__)

#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>  // must follow gl31.h (needs the GL types)

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "hpc/log.hpp"

#include "es_sgemm.inc"
#include "es_conv3x3.inc"
#include "es_reduce.inc"
#include "es_triad.inc"

namespace hpc {
namespace {

std::string eglErrStr(EGLint err) {
  char b[32];
  std::snprintf(b, sizeof b, "0x%04X", static_cast<unsigned>(err));
  return b;
}

std::string glErrStr(GLenum err) {
  char b[32];
  std::snprintf(b, sizeof b, "0x%04X", static_cast<unsigned>(err));
  return b;
}

std::string safeGlString(GLenum name) {
  const GLubyte* p = glGetString(name);
  if (!p) {
    HPC_LOGW("[GLES] glGetString(0x%04X) returned nullptr (glGetError=%s)",
             static_cast<unsigned>(name), glErrStr(glGetError()).c_str());
    return "unknown";
  }
  return sanitizeAscii(reinterpret_cast<const char*>(p));
}

class EglCtx {
 public:
  EGLDisplay dpy = EGL_NO_DISPLAY;
  EGLContext ctx = EGL_NO_CONTEXT;
  EGLSurface surf = EGL_NO_SURFACE;
  std::string error;

  bool init() {
    if (ctx != EGL_NO_CONTEXT && surf != EGL_NO_SURFACE) return true;
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
      EGLint e = eglGetError();
      error = "eglGetDisplay failed (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      return false;
    }
    EGLint maj = 0, min = 0;
    if (!eglInitialize(dpy, &maj, &min)) {
      EGLint e = eglGetError();
      error = "eglInitialize failed (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      return false;
    }
    HPC_LOGI("[GLES] eglInitialize ok: version %d.%d", int(maj), int(min));
    const EGLint cfgAttr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg = nullptr;
    EGLint n = 0;
    if (!eglChooseConfig(dpy, cfgAttr, &cfg, 1, &n) || n < 1) {
      EGLint e = eglGetError();
      error = "no EGL config with OpenGL ES 3 + pbuffer (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      return false;
    }
    const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    surf = eglCreatePbufferSurface(dpy, cfg, pb);
    if (surf == EGL_NO_SURFACE) {
      EGLint e = eglGetError();
      error = "eglCreatePbufferSurface failed (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      return false;
    }
    const EGLint ca[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ca);
    if (ctx == EGL_NO_CONTEXT) {
      EGLint e = eglGetError();
      error = "eglCreateContext failed (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      eglDestroySurface(dpy, surf);
      surf = EGL_NO_SURFACE;
      return false;
    }
    HPC_LOGI("[GLES] EGL context (%p) and 1x1 pbuffer surface (%p) created", ctx, surf);
    return true;
  }
  bool makeCurrent() {
    if (dpy == EGL_NO_DISPLAY || ctx == EGL_NO_CONTEXT || surf == EGL_NO_SURFACE) {
      if (!init()) return false;
    }
    if (eglGetCurrentContext() == ctx) return true;
    if (eglMakeCurrent(dpy, surf, surf, ctx) != EGL_TRUE) {
      EGLint e = eglGetError();
      error = "eglMakeCurrent failed (" + eglErrStr(e) + ")";
      HPC_LOGW("[GLES] %s", error.c_str());
      return false;
    }
    return true;
  }
  void release() {
    if (dpy != EGL_NO_DISPLAY && eglGetCurrentContext() == ctx) {
      eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
  }
  void destroyObjects() {
    if (dpy == EGL_NO_DISPLAY) return;
    release();
    if (ctx != EGL_NO_CONTEXT) {
      eglDestroyContext(dpy, ctx);
      ctx = EGL_NO_CONTEXT;
    }
    if (surf != EGL_NO_SURFACE) {
      eglDestroySurface(dpy, surf);
      surf = EGL_NO_SURFACE;
    }
    // IMPORTANT: Never call eglTerminate(dpy) on EGL_DEFAULT_DISPLAY!
    // EGL_DEFAULT_DISPLAY is shared across the entire process; terminating it from a
    // background thread invalidates the display used by Android's UI RenderThread and
    // crashes the app immediately.
  }
  ~EglCtx() {
    destroyObjects();
  }
};

struct TimerFns {
  void (*gen)(GLsizei, GLuint*) = nullptr;
  void (*del)(GLsizei, const GLuint*) = nullptr;
  void (*begin)(GLenum, GLuint) = nullptr;
  void (*end)(GLenum) = nullptr;
  void (*getObj)(GLuint, GLenum, GLuint64*) = nullptr;
  bool ok() const { return gen && del && begin && end && getObj; }
};

// Workgroup shapes chosen from the driver's GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS. ES 3.1 only
// guarantees 128, and the Arm Mali-G71 reports exactly 128, so the 16x16 / 256 defaults of the
// shared shader bodies cannot be assumed. Both values are powers of two.
struct WgShape { uint32_t wg1d = 256; uint32_t wg2d = 16; };

uint32_t floorPow2(uint32_t v) {
  uint32_t p = 1;
  while (p * 2 <= v) p *= 2;
  return p;
}

WgShape chooseWgShape(int maxInvocations) {
  WgShape s;
  const uint32_t inv = maxInvocations > 0 ? uint32_t(maxInvocations) : 1u;
  s.wg1d = std::min<uint32_t>(256u, floorPow2(inv));
  uint32_t edge = std::min<uint32_t>(16u, floorPow2(inv));
  while (edge > 1 && edge * edge > inv) edge /= 2;
  s.wg2d = edge;
  return s;
}

// "#version" must stay the first line, so the defines are spliced in after it.
std::string withWorkgroup(const char* src, const WgShape& w) {
  std::string s(src);
  const size_t nl = s.find('\n');
  const std::string defs = "#define WG1D " + std::to_string(w.wg1d) + "\n#define WG2D " + std::to_string(w.wg2d) + "\n";
  s.insert(nl == std::string::npos ? s.size() : nl + 1, defs);
  return s;
}

GLuint compileProgram(const char* src, std::string* err) {
  GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(sh, 1, &src, nullptr);
  glCompileShader(sh);
  GLint ok = 0;
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024] = {};
    glGetShaderInfoLog(sh, sizeof log - 1, nullptr, log);
    const std::string cleanLog = sanitizeAscii(log);
    HPC_LOGW("[GLES] shader compile failed: %s", cleanLog.c_str());
    if (err) *err = std::string("GLSL compile failed: ") + cleanLog;
    glDeleteShader(sh);
    return 0;
  }
  GLuint pr = glCreateProgram();
  glAttachShader(pr, sh);
  glLinkProgram(pr);
  glDeleteShader(sh);
  glGetProgramiv(pr, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[1024] = {};
    glGetProgramInfoLog(pr, sizeof log - 1, nullptr, log);
    const std::string cleanLog = sanitizeAscii(log);
    HPC_LOGW("[GLES] program link failed: %s", cleanLog.c_str());
    if (err) *err = std::string("GLSL link failed: ") + cleanLog;
    glDeleteProgram(pr);
    return 0;
  }
  HPC_LOGV("[GLES] compiled & linked compute program %u", unsigned(pr));
  return pr;
}

class GlExecution final : public Execution {
 public:
  GlExecution(std::shared_ptr<EglCtx> c, TimerFns t, WgShape w) : c_(std::move(c)), tf_(t), wg_(w) {}
  ~GlExecution() override {
    if (!c_->makeCurrent()) return;
    GLuint bufs[] = {a_, b_, bias_, partial_, out_, dummy_};
    for (GLuint b : bufs) if (b) glDeleteBuffers(1, &b);
    if (prog_) glDeleteProgram(prog_);
    if (query_ && tf_.ok()) tf_.del(1, &query_);
  }

  GLuint makeBuf(const void* data, size_t bytes) {
    GLuint b = 0;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(std::max<size_t>(16, bytes)), data, GL_DYNAMIC_COPY);
    return b;
  }

  bool build(const Problem& p, std::string* err) {
    if (!c_->makeCurrent()) { if (err) *err = "eglMakeCurrent failed on this thread"; return false; }
    p_ = &p;
    outCount_ = p.outputCount();
    const size_t F = sizeof(float);
    float zero[4] = {0, 0, 0, 0};
    dummy_ = makeBuf(zero, 16);
    const char* src = nullptr;
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense:
        src = kEs_sgemm;
        a_ = makeBuf(p.a.data(), size_t(p.m) * p.k * F);
        b_ = makeBuf(p.b.data(), size_t(p.k) * p.n * F);
        if (p.kernel == Kernel::Dense) bias_ = makeBuf(p.bias.data(), size_t(p.n) * F);
        out_ = makeBuf(nullptr, outCount_ * F);
        gx_ = (p.n + wg_.wg2d - 1) / wg_.wg2d; gy_ = (p.m + wg_.wg2d - 1) / wg_.wg2d;
        break;
      case Kernel::Conv3x3:
        src = kEs_conv3x3;
        a_ = makeBuf(p.a.data(), size_t(p.m) * p.n * F);
        b_ = makeBuf(p.b.data(), 9 * F);
        out_ = makeBuf(nullptr, outCount_ * F);
        gx_ = (p.n + wg_.wg2d - 1) / wg_.wg2d; gy_ = (p.m + wg_.wg2d - 1) / wg_.wg2d;
        break;
      case Kernel::Reduce:
        src = kEs_reduce;
        a_ = makeBuf(p.a.data(), size_t(p.n) * F);
        partial_ = makeBuf(nullptr, kGroups * F);
        out_ = makeBuf(nullptr, F);
        groups_ = std::min<uint32_t>(kGroups, std::max<uint32_t>(1, (p.n + wg_.wg1d - 1) / wg_.wg1d));
        gx_ = groups_; gy_ = 1;
        break;
      case Kernel::Triad:
        src = kEs_triad;
        a_ = makeBuf(p.a.data(), size_t(p.n) * F);
        b_ = makeBuf(p.b.data(), size_t(p.n) * F);
        out_ = makeBuf(nullptr, size_t(p.n) * F);
        gx_ = std::min<uint32_t>(8192, std::max<uint32_t>(1, (p.n + wg_.wg1d - 1) / wg_.wg1d)); gy_ = 1;
        break;
    }
    glFinish();
    const std::string shaderText = withWorkgroup(src, wg_);
    HPC_LOGV("[GLES] workgroup shape: 1D=%u 2D=%ux%u", unsigned(wg_.wg1d), unsigned(wg_.wg2d), unsigned(wg_.wg2d));
    prog_ = compileProgram(shaderText.c_str(), err);
    if (!prog_) return false;
    uM_ = glGetUniformLocation(prog_, "u_m");
    uK_ = glGetUniformLocation(prog_, "u_k");
    uN_ = glGetUniformLocation(prog_, "u_n");
    uMode_ = glGetUniformLocation(prog_, "u_mode");
    uAlpha_ = glGetUniformLocation(prog_, "u_alpha");
    if (tf_.ok()) tf_.gen(1, &query_);
    return glGetError() == GL_NO_ERROR || (err && (*err = "GL error during setup", false));
  }

  void bindAll(GLuint b0, GLuint b1, GLuint b2, GLuint b3) {
    const GLuint bs[4] = {b0, b1, b2, b3};
    for (GLuint i = 0; i < 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, bs[i] ? bs[i] : dummy_);
  }
  void setU(uint32_t m, uint32_t k, uint32_t n, uint32_t mode, float alpha) {
    if (uM_ >= 0) glUniform1ui(uM_, m);
    if (uK_ >= 0) glUniform1ui(uK_, k);
    if (uN_ >= 0) glUniform1ui(uN_, n);
    if (uMode_ >= 0) glUniform1ui(uMode_, mode);
    if (uAlpha_ >= 0) glUniform1f(uAlpha_, alpha);
  }

  bool run(double* deviceMs, std::string* err) override {
    if (deviceMs) *deviceMs = -1;
    if (!c_->makeCurrent()) { if (err) *err = "eglMakeCurrent failed on this thread"; return false; }
    const Problem& p = *p_;
    glUseProgram(prog_);
    const bool timed = tf_.ok() && query_ != 0;
    if (timed) {
      glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint_);  // clear sticky flag
      tf_.begin(GL_TIME_ELAPSED_EXT, query_);
    }
    if (p.kernel == Kernel::Reduce) {
      bindAll(a_, 0, partial_, 0);
      setU(p.m, p.k, p.n, 0, p.alpha);
      glDispatchCompute(gx_, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      bindAll(partial_, 0, out_, 0);
      setU(0, 0, groups_, 0, 0.0f);
      glDispatchCompute(1, 1, 1);
    } else {
      bindAll(a_, b_, out_, bias_);
      setU(p.m, p.k, p.n, p.kernel == Kernel::Dense ? 1u : 0u, p.alpha);
      glDispatchCompute(gx_, gy_, 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    if (timed) tf_.end(GL_TIME_ELAPSED_EXT);
    glFinish();
    if (glGetError() != GL_NO_ERROR) { if (err) *err = "GL error during dispatch"; return false; }
    if (timed) {
      GLint dis = 0;
      glGetIntegerv(GL_GPU_DISJOINT_EXT, &dis);
      GLuint64 ns = 0;
      tf_.getObj(query_, GL_QUERY_RESULT_EXT, &ns);
      // A real dispatch cannot take under 1 microsecond. The Mali-G76 driver returned
      // ~190 ns for a 256^3 sgemm (about 175 TFLOP/s on a phone), so such values are
      // discarded and the report falls back to wall-clock time.
      constexpr GLuint64 kMinPlausibleNs = 1000;
      if (!dis && ns >= kMinPlausibleNs) {
        if (deviceMs) *deviceMs = double(ns) * 1e-6;
      } else if (!dis) {
        HPC_LOGV("[GLES] timer query returned an implausible %llu ns; using wall-clock time", (unsigned long long)ns);
      }
    }
    return true;
  }

  bool read(std::vector<float>& out, std::string* err) override {
    if (!c_->makeCurrent()) return false;
    out.resize(outCount_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_);
    void* m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(outCount_ * sizeof(float)), GL_MAP_READ_BIT);
    if (!m) { if (err) *err = "glMapBufferRange failed"; return false; }
    std::memcpy(out.data(), m, outCount_ * sizeof(float));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    return true;
  }

 private:
  static constexpr uint32_t kGroups = 256;
  std::shared_ptr<EglCtx> c_;
  TimerFns tf_;
  WgShape wg_;
  const Problem* p_ = nullptr;
  size_t outCount_ = 0;
  GLuint a_ = 0, b_ = 0, bias_ = 0, partial_ = 0, out_ = 0, dummy_ = 0, prog_ = 0, query_ = 0;
  GLint uM_ = -1, uK_ = -1, uN_ = -1, uMode_ = -1, uAlpha_ = -1, disjoint_ = 0;
  uint32_t gx_ = 1, gy_ = 1, groups_ = 1;
};

class GlesBackend final : public Backend {
 public:
  GlesBackend(std::shared_ptr<EglCtx> c, TimerFns t, WgShape w, BackendInfo info)
      : c_(std::move(c)), tf_(t), wg_(w), info_(std::move(info)) {}
  ~GlesBackend() override {
    if (c_) c_->destroyObjects();
  }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions&, std::string* err) override {
    if (!c_->makeCurrent()) {
      if (err) *err = c_->error.empty() ? "eglMakeCurrent failed" : c_->error;
      return nullptr;
    }
    std::unique_ptr<GlExecution> e(new GlExecution(c_, tf_, wg_));
    if (!e->build(p, err)) return nullptr;
    return e;
  }

 private:
  std::shared_ptr<EglCtx> c_;
  TimerFns tf_;
  WgShape wg_;
  BackendInfo info_;
};

}  // namespace

void discoverGles(std::vector<std::unique_ptr<Backend>>& out) {
  BackendInfo info;
  info.id = "gles";
  info.api = "OpenGL ES 3.1 compute";
  auto stub = [&](const std::string& why) {
    info.note = sanitizeAscii(why);
    HPC_LOGW("[GLES] unavailable: %s", info.note.c_str());
    out.emplace_back(new StubBackend(info));
  };
  auto c = std::make_shared<EglCtx>();
  if (!c->init()) return stub(c->error);
  if (!c->makeCurrent()) {
    const std::string why = c->error.empty() ? "eglMakeCurrent failed" : c->error;
    c->destroyObjects();
    return stub(why);
  }
  while (glGetError() != GL_NO_ERROR) {}  // clear any prior GL error
  GLint maj = 0, min = 0;
  glGetIntegerv(GL_MAJOR_VERSION, &maj);
  glGetIntegerv(GL_MINOR_VERSION, &min);
  info.device = safeGlString(GL_RENDERER);
  info.vendor = safeGlString(GL_VENDOR);
  info.driver = safeGlString(GL_VERSION);
  {
    // Software GL implementations run on the CPU; everything else is a GPU (vendor-neutral check).
    std::string lower = info.device;
    for (char& ch : lower) ch = char(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
    const bool software = lower.find("swiftshader") != std::string::npos || lower.find("llvmpipe") != std::string::npos ||
                          lower.find("softpipe") != std::string::npos || lower.find("software") != std::string::npos;
    info.deviceClass = software ? "cpu" : "gpu";
  }
  HPC_LOGI("[GLES] context ES %d.%d | renderer='%s' | vendor='%s' | version='%s'",
           int(maj), int(min), info.device.c_str(), info.vendor.c_str(), info.driver.c_str());
  if (maj < 3 || (maj == 3 && min < 1)) {
    c->destroyObjects();
    return stub("context is OpenGL ES " + std::to_string(maj) + "." + std::to_string(min) + ", compute needs 3.1");
  }
  GLint inv = 0, shm = 0;
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &inv);
  glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shm);
  info.detail["max_workgroup_invocations"] = std::to_string(inv);
  info.detail["max_shared_memory_bytes"] = std::to_string(shm);
  const GLubyte* rawExts = glGetString(GL_EXTENSIONS);
  const char* exts = reinterpret_cast<const char*>(rawExts);
  TimerFns tf;
  if (exts && std::strstr(exts, "GL_EXT_disjoint_timer_query")) {
    tf.gen = reinterpret_cast<decltype(tf.gen)>(eglGetProcAddress("glGenQueriesEXT"));
    tf.del = reinterpret_cast<decltype(tf.del)>(eglGetProcAddress("glDeleteQueriesEXT"));
    tf.begin = reinterpret_cast<decltype(tf.begin)>(eglGetProcAddress("glBeginQueryEXT"));
    tf.end = reinterpret_cast<decltype(tf.end)>(eglGetProcAddress("glEndQueryEXT"));
    tf.getObj = reinterpret_cast<decltype(tf.getObj)>(eglGetProcAddress("glGetQueryObjectui64vEXT"));
  }
  info.detail["gpu_timer_query"] = tf.ok() ? "GL_EXT_disjoint_timer_query" : "unavailable (wall-clock only)";
  info.detail["zero_copy"] = "not implemented for GLES";
  info.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
  const WgShape wg = chooseWgShape(inv);
  info.detail["workgroup_1d"] = std::to_string(wg.wg1d);
  info.detail["workgroup_2d"] = std::to_string(wg.wg2d) + "x" + std::to_string(wg.wg2d);
  // The smallest shape the shaders support is 8x8 / 64; ES 3.1 guarantees 128 invocations.
  info.available = wg.wg1d >= 64 && wg.wg2d >= 8 && shm >= 4096;
  if (!info.available) info.note = "needs >= 64 invocations per workgroup and 4 KiB shared memory (limits " + std::to_string(inv) + " / " + std::to_string(shm) + ")";
  // Tear down the probe context/pbuffer now (without eglTerminate); GlesBackend will
  // lazily create a fresh context/surface in prepare() only if a kernel is actually run.
  c->destroyObjects();
  HPC_LOGI("[GLES] discover complete: available=%d maxInv=%d sharedMem=%d timerQuery=%d wg1d=%u wg2d=%u",
           info.available ? 1 : 0, int(inv), int(shm), tf.ok() ? 1 : 0, unsigned(wg.wg1d), unsigned(wg.wg2d));
  out.emplace_back(new GlesBackend(c, tf, wg, info));
}

}  // namespace hpc

#else  // !__ANDROID__

namespace hpc {
void discoverGles(std::vector<std::unique_ptr<Backend>>& out) {
  BackendInfo i;
  i.id = "gles";
  i.api = "OpenGL ES 3.1 compute";
  i.note = "Android-only backend (headless EGL); not built on this platform";
  out.emplace_back(new StubBackend(i));
}
}  // namespace hpc

#endif
