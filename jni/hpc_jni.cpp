// SPDX-License-Identifier: Apache-2.0
// JNI bridge for org.androidhpc.bench.Native. Everything heavy stays in hpc_core.
#include <jni.h>

#include <android/log.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "hpc/bench.hpp"
#include "hpc/log.hpp"

namespace {
std::mutex gMu;        // guards the strings below (held only briefly)
std::mutex gWorkMu;    // one discover/run at a time, even if the Activity is recreated mid-run
std::string gProgress, gLastText, gLastJson;
std::atomic<bool> gCancel{false};

std::string str(JNIEnv* env, jstring s) {
  if (!env || !s) return "";
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r = c ? c : "";
  if (c) env->ReleaseStringUTFChars(s, c);
  if (env->ExceptionCheck()) env->ExceptionClear();
  return r;
}

jstring safeJString(JNIEnv* env, const std::string& raw, size_t maxLen = std::numeric_limits<size_t>::max()) {
  const std::string clean = hpc::sanitizeAscii(raw, maxLen);
  return env->NewStringUTF(clean.c_str());
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

// {"error":"..."} with the message sanitised and JSON-escaped (driver text may contain quotes).
std::string errorJson(const std::string& raw) {
  std::string o = "{\"error\":\"";
  for (char c : hpc::sanitizeAscii(raw, 400)) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if (c == '\t') o += "\\t";
    else o += c;
  }
  return o + "\"}";
}

void setProgress(const std::string& s) {
  const std::string clean = hpc::sanitizeAscii(s, 512);
  {
    std::lock_guard<std::mutex> lk(gMu);
    gProgress = clean;
  }
  HPC_LOGI("progress: %s", clean.c_str());
}
}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_org_androidhpc_bench_Native_initLogging(JNIEnv* env, jclass, jstring logPath,
                                                                    jstring breadcrumbPath, jboolean verbose) {
  try {
    const std::string lp = str(env, logPath);
    const std::string bp = str(env, breadcrumbPath);
    hpc::initLog(lp, bp, verbose != JNI_FALSE);
    hpc::installCrashHandlers();
    HPC_LOGI("Native logging initialized: log='%s' breadcrumb='%s' verbose=%d", lp.c_str(), bp.c_str(),
             int(verbose != JNI_FALSE));
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "android-hpc", "initLogging exception: %s", e.what());
  } catch (...) {
    __android_log_print(ANDROID_LOG_ERROR, "android-hpc", "initLogging unknown exception");
  }
}

JNIEXPORT void JNICALL Java_org_androidhpc_bench_Native_setVerbose(JNIEnv*, jclass, jboolean verbose) {
  try {
    hpc::setLogVerbose(verbose != JNI_FALSE);
    HPC_LOGI("Verbose logging set to %d", int(verbose != JNI_FALSE));
  } catch (...) {
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_lastBreadcrumb(JNIEnv* env, jclass) {
  try {
    return safeJString(env, hpc::lastBreadcrumb(), 512);
  } catch (const std::exception& e) {
    return safeJString(env, std::string("error: ") + e.what(), 512);
  } catch (...) {
    return safeJString(env, "error: unknown", 512);
  }
}

JNIEXPORT void JNICALL Java_org_androidhpc_bench_Native_setDisabledBackends(JNIEnv* env, jclass, jstring csv) {
  try {
    hpc::setDisabledBackends(str(env, csv));
  } catch (const std::exception& e) {
    HPC_LOGE("setDisabledBackends exception: %s", e.what());
  } catch (...) {
    HPC_LOGE("setDisabledBackends unknown exception");
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_discover(JNIEnv* env, jclass, jboolean bigOnly) {
  try {
    std::lock_guard<std::mutex> work(gWorkMu);
    HPC_LOGI("JNI discover(bigOnly=%d) begin", int(bigOnly != JNI_FALSE));
    hpc::DiscoverOptions o;
    o.cpuBigCoresOnly = bigOnly != JNI_FALSE;
    hpc::Report r = hpc::discoverReport(o);
    std::string text = hpc::toText(r);
    std::string json = hpc::toJson(r);
    {
      std::lock_guard<std::mutex> lk(gMu);
      gLastText = text;
      gLastJson = json;
    }
    HPC_LOGI("JNI discover complete (%zu backends)", r.backends.size());
    return safeJString(env, json);
  } catch (const std::exception& e) {
    HPC_LOGE("JNI discover failed with std::exception: %s", e.what());
    const std::string msg = "discovery failed: " + hpc::sanitizeAscii(e.what(), 400);
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = msg;
    gLastJson = errorJson(e.what());
    return safeJString(env, gLastJson);
  } catch (...) {
    HPC_LOGE("JNI discover failed with unknown C++ exception");
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = "discovery failed: unknown C++ exception";
    gLastJson = "{\"error\":\"unknown C++ exception\"}";
    return safeJString(env, gLastJson);
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_run(JNIEnv* env, jclass, jstring preset, jint warmup,
                                                               jint iters, jstring backends, jstring kernels,
                                                               jboolean zeroCopy, jboolean hetero, jboolean bigOnly) {
  std::unique_lock<std::mutex> work(gWorkMu, std::try_to_lock);
  if (!work.owns_lock()) {
    HPC_LOGW("JNI run rejected: another discover/run is in progress");
    return safeJString(env, "{\"error\":\"another benchmark run is already in progress\"}");
  }
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
    cfg.zeroCopy = zeroCopy != JNI_FALSE;
    cfg.hetero = hetero != JNI_FALSE;
    cfg.cpuBigOnly = bigOnly != JNI_FALSE;
    gCancel = false;
    HPC_LOGI("JNI run(preset=%s, warmup=%d, iters=%d, zeroCopy=%d, hetero=%d, bigOnly=%d) begin",
             cfg.preset.c_str(), cfg.warmup, cfg.iterations, int(cfg.zeroCopy), int(cfg.hetero), int(cfg.cpuBigOnly));
    hpc::Report r = hpc::runBenchmarks(cfg, setProgress, &gCancel);
    std::string json = hpc::toJson(r), text = hpc::toText(r);
    {
      std::lock_guard<std::mutex> lk(gMu);
      gLastJson = json;
      gLastText = text;
      gProgress = r.cancelled ? "cancelled" : "done";
    }
    HPC_LOGI("JNI run complete: allVerified=%d cancelled=%d", int(r.allVerified), int(r.cancelled));
    return safeJString(env, json);
  } catch (const std::exception& e) {
    HPC_LOGE("JNI run failed with std::exception: %s", e.what());
    std::string errText = std::string("benchmark failed: ") + hpc::sanitizeAscii(e.what(), 512);
    std::string errJson = errorJson(e.what());
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = errText;
    gLastJson = errJson;
    gProgress = "failed";
    return safeJString(env, gLastJson);
  } catch (...) {
    HPC_LOGE("JNI run failed with unknown C++ exception");
    std::lock_guard<std::mutex> lk(gMu);
    gLastText = "benchmark failed: unknown C++ exception";
    gLastJson = "{\"error\":\"unknown C++ exception\"}";
    gProgress = "failed";
    return safeJString(env, gLastJson);
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_progress(JNIEnv* env, jclass) {
  try {
    std::lock_guard<std::mutex> lk(gMu);
    return safeJString(env, gProgress, 512);
  } catch (...) {
    return safeJString(env, "", 512);
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_lastText(JNIEnv* env, jclass) {
  try {
    std::lock_guard<std::mutex> lk(gMu);
    return safeJString(env, gLastText);
  } catch (...) {
    return safeJString(env, "");
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_lastJson(JNIEnv* env, jclass) {
  try {
    std::lock_guard<std::mutex> lk(gMu);
    return safeJString(env, gLastJson);
  } catch (...) {
    return safeJString(env, "{}");
  }
}

JNIEXPORT void JNICALL Java_org_androidhpc_bench_Native_cancel(JNIEnv*, jclass) {
  try {
    gCancel = true;
    HPC_LOGI("Cancellation requested from UI");
  } catch (...) {
  }
}

JNIEXPORT jstring JNICALL Java_org_androidhpc_bench_Native_version(JNIEnv* env, jclass) {
  try {
    return safeJString(env, hpc::versionString(), 128);
  } catch (...) {
    return safeJString(env, "0.1.0", 128);
  }
}

}  // extern "C"
