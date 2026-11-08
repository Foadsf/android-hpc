// SPDX-License-Identifier: Apache-2.0
// JNI bridge for org.androidhpc.bench.Native. Everything heavy stays in hpc_core.
#include <jni.h>

#include <android/log.h>

#include <atomic>
#include <mutex>
#include <sstream>
#include <string>

#include "hpc/bench.hpp"

namespace {
std::mutex gMu;
std::string gProgress, gLastText, gLastJson;
std::atomic<bool> gCancel{false};

std::string str(JNIEnv* env, jstring s) {
  if (!s) return "";
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r = c ? c : "";
  if (c) env->ReleaseStringUTFChars(s, c);
  return r;
}
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
  return v;
}
void setProgress(const std::string& s) {
  std::lock_guard<std::mutex> lk(gMu);
  gProgress = s;
  __android_log_print(ANDROID_LOG_INFO, "android-hpc", "%s", s.c_str());
}
}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_discover(JNIEnv* env, jclass, jboolean bigOnly) {
  try {
    hpc::DiscoverOptions o;
    o.cpuBigCoresOnly = bigOnly;
    hpc::Report r = hpc::discoverReport(o);
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = hpc::toText(r);
    gLastJson = hpc::toJson(r);
    return env->NewStringUTF(gLastText.c_str());
  } catch (const std::exception& e) {
    return env->NewStringUTF((std::string("discovery failed: ") + e.what()).c_str());
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_run(JNIEnv* env, jclass, jstring preset, jint warmup, jint iters,
                                                                 jstring backends, jstring kernels, jboolean zeroCopy,
                                                                 jboolean hetero, jboolean bigOnly) {
  try {
    hpc::BenchConfig cfg;
    cfg.preset = str(env, preset);
    cfg.warmup = warmup;
    cfg.iterations = iters;
    cfg.backends = split(str(env, backends));
    for (auto& k : split(str(env, kernels))) {
      hpc::Kernel kk;
      if (hpc::parseKernel(k, kk)) cfg.kernels.push_back(kk);
    }
    cfg.zeroCopy = zeroCopy;
    cfg.hetero = hetero;
    cfg.cpuBigOnly = bigOnly;
    gCancel = false;
    hpc::Report r = hpc::runBenchmarks(cfg, setProgress, &gCancel);
    std::string json = hpc::toJson(r), text = hpc::toText(r);
    std::lock_guard<std::mutex> lk(gMu);
    gLastJson = json;
    gLastText = text;
    gProgress = r.cancelled ? "cancelled" : "done";
    return env->NewStringUTF(json.c_str());
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = std::string("benchmark failed: ") + e.what();
    gLastJson = "{\"error\":\"" + std::string(e.what()) + "\"}";
    return env->NewStringUTF(gLastJson.c_str());
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_progress(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> lk(gMu);
  return env->NewStringUTF(gProgress.c_str());
}
JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_lastText(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> lk(gMu);
  return env->NewStringUTF(gLastText.c_str());
}
JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_lastJson(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> lk(gMu);
  return env->NewStringUTF(gLastJson.c_str());
}
JNIEXPORT void JNICALL Java_org_androidhpc_bench_Native_cancel(JNIEnv*, jclass) { gCancel = true; }
JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_version(JNIEnv* env, jclass) {
  return env->NewStringUTF(hpc::versionString());
}

}  // extern "C"
