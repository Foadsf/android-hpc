// SPDX-License-Identifier: Apache-2.0
// NNAPI backend (Android only): runs the `dense` kernel (relu(x*W+b)) as one NNAPI
// FULLY_CONNECTED op on each NNAPI device the platform exposes.
//
// NNAPI is deprecated since Android 15 (see docs/FACT_CHECK.md) but is still present and is the
// only vendor-neutral NDK route to NPUs/DSPs. It is opened with dlopen, never linked. A device
// named "nnapi-reference" is the CPU reference implementation, NOT an NPU, and is labelled so.
#include "backend_util.hpp"

#if defined(__ANDROID__)

#include <algorithm>
#include <cstring>

#include "hpc/dynlib.hpp"
#include "hpc/log.hpp"

namespace hpc {
namespace {

struct NnModel;
struct NnCompilation;
struct NnExecution;
struct NnDevice;
struct NnOperandType {
  int32_t type;
  uint32_t dimensionCount;
  const uint32_t* dimensions;
  float scale;
  int32_t zeroPoint;
};
constexpr int32_t NN_INT32 = 1, NN_TENSOR_FLOAT32 = 3;
constexpr int32_t NN_OP_FULLY_CONNECTED = 9;
constexpr int32_t NN_FUSED_RELU = 1;
constexpr int32_t NN_PREFER_SUSTAINED_SPEED = 2;
constexpr int32_t NN_NO_ERROR = 0;

struct NnApi {
  DynLib lib;
  int (*modelCreate)(NnModel**) = nullptr;
  void (*modelFree)(NnModel*) = nullptr;
  int (*modelAddOperand)(NnModel*, const NnOperandType*) = nullptr;
  int (*modelSetOperandValue)(NnModel*, int32_t, const void*, size_t) = nullptr;
  int (*modelAddOperation)(NnModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*) = nullptr;
  int (*modelIdentify)(NnModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*) = nullptr;
  int (*modelFinish)(NnModel*) = nullptr;
  int (*compCreate)(NnModel*, NnCompilation**) = nullptr;
  int (*compCreateForDevices)(NnModel*, const NnDevice* const*, uint32_t, NnCompilation**) = nullptr;
  int (*compSetPreference)(NnCompilation*, int32_t) = nullptr;
  int (*compFinish)(NnCompilation*) = nullptr;
  void (*compFree)(NnCompilation*) = nullptr;
  int (*execCreate)(NnCompilation*, NnExecution**) = nullptr;
  void (*execFree)(NnExecution*) = nullptr;
  int (*execSetInput)(NnExecution*, int32_t, const NnOperandType*, const void*, size_t) = nullptr;
  int (*execSetOutput)(NnExecution*, int32_t, const NnOperandType*, void*, size_t) = nullptr;
  int (*execCompute)(NnExecution*) = nullptr;                    // API 29
  int (*execSetMeasureTiming)(NnExecution*, bool) = nullptr;     // API 29
  int (*execGetDuration)(NnExecution*, int32_t, uint64_t*) = nullptr;
  int (*getDeviceCount)(uint32_t*) = nullptr;                    // API 29
  int (*getDevice)(uint32_t, NnDevice**) = nullptr;
  int (*deviceGetName)(const NnDevice*, const char**) = nullptr;
  int (*deviceGetType)(const NnDevice*, int32_t*) = nullptr;
  int (*deviceGetFeatureLevel)(const NnDevice*, int64_t*) = nullptr;
  std::string error;

  bool load() {
    std::string tried;
    if (!lib.open({"libneuralnetworks.so"}, &tried)) {
      error = "libneuralnetworks.so not available (tried: " + tried + ")";
      return false;
    }
#define S(field, name) field = lib.sym<decltype(field)>(name);
    S(modelCreate, "ANeuralNetworksModel_create") S(modelFree, "ANeuralNetworksModel_free")
    S(modelAddOperand, "ANeuralNetworksModel_addOperand") S(modelSetOperandValue, "ANeuralNetworksModel_setOperandValue")
    S(modelAddOperation, "ANeuralNetworksModel_addOperation") S(modelIdentify, "ANeuralNetworksModel_identifyInputsAndOutputs")
    S(modelFinish, "ANeuralNetworksModel_finish") S(compCreate, "ANeuralNetworksCompilation_create")
    S(compCreateForDevices, "ANeuralNetworksCompilation_createForDevices") S(compSetPreference, "ANeuralNetworksCompilation_setPreference")
    S(compFinish, "ANeuralNetworksCompilation_finish") S(compFree, "ANeuralNetworksCompilation_free")
    S(execCreate, "ANeuralNetworksExecution_create") S(execFree, "ANeuralNetworksExecution_free")
    S(execSetInput, "ANeuralNetworksExecution_setInput") S(execSetOutput, "ANeuralNetworksExecution_setOutput")
    S(execCompute, "ANeuralNetworksExecution_compute") S(execSetMeasureTiming, "ANeuralNetworksExecution_setMeasureTiming")
    S(execGetDuration, "ANeuralNetworksExecution_getDuration") S(getDeviceCount, "ANeuralNetworks_getDeviceCount")
    S(getDevice, "ANeuralNetworks_getDevice") S(deviceGetName, "ANeuralNetworksDevice_getName")
    S(deviceGetType, "ANeuralNetworksDevice_getType") S(deviceGetFeatureLevel, "ANeuralNetworksDevice_getFeatureLevel")
#undef S
    if (!modelCreate || !modelAddOperand || !modelSetOperandValue || !modelAddOperation ||
        !modelIdentify || !modelFinish || !compCreate || !compFinish || !execCreate ||
        !execSetInput || !execSetOutput) {
      error = "libneuralnetworks.so is missing core NNAPI entry points";
      HPC_LOGW("[NNAPI] %s", error.c_str());
      return false;
    }
    if (!execCompute) {
      error = "NNAPI < 29 (no synchronous compute); not supported by this backend";
      HPC_LOGW("[NNAPI] %s", error.c_str());
      return false;
    }
    HPC_LOGI("[NNAPI] loaded libneuralnetworks.so (getDeviceCount=%p, compCreateForDevices=%p)",
             reinterpret_cast<void*>(getDeviceCount), reinterpret_cast<void*>(compCreateForDevices));
    return true;
  }
};

class NnExec final : public Execution {
 public:
  NnExec(std::shared_ptr<NnApi> api) : api_(std::move(api)) {}
  ~NnExec() override {
    if (comp_) api_->compFree(comp_);
    if (model_) api_->modelFree(model_);
  }
  bool build(const Problem& p, NnDevice* device, std::string* err) {
    p_ = &p;
    const uint32_t M = p.m, K = p.k, N = p.n;
    wt_.resize(size_t(N) * K);  // FULLY_CONNECTED weights are [units, input]; our B is [k, n]
    for (uint32_t kk = 0; kk < K; ++kk)
      for (uint32_t j = 0; j < N; ++j) wt_[size_t(j) * K + kk] = p.b[size_t(kk) * N + j];
    out_.assign(size_t(M) * N, 0.0f);

    const uint32_t dIn[2] = {M, K}, dW[2] = {N, K}, dB[1] = {N}, dOut[2] = {M, N};
    NnOperandType tIn{NN_TENSOR_FLOAT32, 2, dIn, 0, 0}, tW{NN_TENSOR_FLOAT32, 2, dW, 0, 0},
        tB{NN_TENSOR_FLOAT32, 1, dB, 0, 0}, tAct{NN_INT32, 0, nullptr, 0, 0}, tOut{NN_TENSOR_FLOAT32, 2, dOut, 0, 0};
    int r = api_->modelCreate(&model_);
    if (r != NN_NO_ERROR) { if (err) *err = "ANeuralNetworksModel_create: " + std::to_string(r); return false; }
    const NnOperandType* ops[5] = {&tIn, &tW, &tB, &tAct, &tOut};
    for (int i = 0; i < 5; ++i)
      if ((r = api_->modelAddOperand(model_, ops[i])) != NN_NO_ERROR) { if (err) *err = "addOperand: " + std::to_string(r); return false; }
    const int32_t act = NN_FUSED_RELU;
    if ((r = api_->modelSetOperandValue(model_, 1, wt_.data(), wt_.size() * sizeof(float))) != NN_NO_ERROR ||
        (r = api_->modelSetOperandValue(model_, 2, p.bias.data(), p.bias.size() * sizeof(float))) != NN_NO_ERROR ||
        (r = api_->modelSetOperandValue(model_, 3, &act, sizeof act)) != NN_NO_ERROR) {
      if (err) *err = "setOperandValue: " + std::to_string(r);
      return false;
    }
    const uint32_t in[4] = {0, 1, 2, 3}, outIdx[1] = {4}, mIn[1] = {0};
    if ((r = api_->modelAddOperation(model_, NN_OP_FULLY_CONNECTED, 4, in, 1, outIdx)) != NN_NO_ERROR ||
        (r = api_->modelIdentify(model_, 1, mIn, 1, outIdx)) != NN_NO_ERROR || (r = api_->modelFinish(model_)) != NN_NO_ERROR) {
      if (err) *err = "model build: " + std::to_string(r);
      return false;
    }
    if (device && api_->compCreateForDevices) {
      const NnDevice* devs[1] = {device};
      r = api_->compCreateForDevices(model_, devs, 1, &comp_);
    } else {
      r = api_->compCreate(model_, &comp_);
    }
    HPC_LOGV("[NNAPI] compilation create (device=%p) -> rc=%d", static_cast<void*>(device), r);
    if (r != NN_NO_ERROR || !comp_) {
      comp_ = nullptr;
      if (err) *err = "compilation create: " + std::to_string(r);
      return false;
    }
    if (api_->compSetPreference) api_->compSetPreference(comp_, NN_PREFER_SUSTAINED_SPEED);
    r = api_->compFinish(comp_);
    HPC_LOGI("[NNAPI] compilation finish (device=%p) -> rc=%d", static_cast<void*>(device), r);
    if (r != NN_NO_ERROR) {
      if (err) *err = "compilation finish failed (" + std::to_string(r) + "): device likely cannot run fp32 FULLY_CONNECTED";
      return false;
    }
    return true;
  }
  bool run(double* deviceMs, std::string* err) override {
    if (deviceMs) *deviceMs = -1;
    NnExecution* ex = nullptr;
    int r = api_->execCreate(comp_, &ex);
    if (r != NN_NO_ERROR) { if (err) *err = "execution create: " + std::to_string(r); return false; }
    const bool timing = api_->execSetMeasureTiming && api_->execGetDuration && api_->execSetMeasureTiming(ex, true) == NN_NO_ERROR;
    r = api_->execSetInput(ex, 0, nullptr, p_->a.data(), p_->a.size() * sizeof(float));
    if (r == NN_NO_ERROR) r = api_->execSetOutput(ex, 0, nullptr, out_.data(), out_.size() * sizeof(float));
    if (r == NN_NO_ERROR) r = api_->execCompute(ex);
    if (r == NN_NO_ERROR && timing && deviceMs) {
      uint64_t ns = 0;  // ANEURALNETWORKS_DURATION_ON_HARDWARE = 0
      if (api_->execGetDuration(ex, 0, &ns) == NN_NO_ERROR && ns != UINT64_MAX) *deviceMs = double(ns) * 1e-6;
    }
    api_->execFree(ex);
    if (r != NN_NO_ERROR) { if (err) *err = "execution failed: " + std::to_string(r); return false; }
    return true;
  }
  bool read(std::vector<float>& out, std::string*) override { out = out_; return true; }

 private:
  std::shared_ptr<NnApi> api_;
  const Problem* p_ = nullptr;
  NnModel* model_ = nullptr;
  NnCompilation* comp_ = nullptr;
  std::vector<float> wt_, out_;
};

class NnBackend final : public Backend {
 public:
  NnBackend(std::shared_ptr<NnApi> api, NnDevice* dev, BackendInfo info) : api_(std::move(api)), dev_(dev), info_(std::move(info)) {}
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions&, std::string* err) override {
    std::unique_ptr<NnExec> e(new NnExec(api_));
    if (!e->build(p, dev_, err)) return nullptr;
    return e;
  }

 private:
  std::shared_ptr<NnApi> api_;
  NnDevice* dev_;
  BackendInfo info_;
};

}  // namespace

void discoverNnapi(std::vector<std::unique_ptr<Backend>>& out) {
  auto api = std::make_shared<NnApi>();
  if (!api->load()) {
    BackendInfo i;
    i.id = "nnapi";
    i.api = "NNAPI (deprecated since Android 15)";
    i.note = sanitizeAscii(api->error, 256);
    HPC_LOGW("[NNAPI] unavailable: %s", i.note.c_str());
    out.emplace_back(new StubBackend(i));
    return;
  }
  auto base = [&](const std::string& id) {
    BackendInfo i;
    i.id = id;
    i.api = "NNAPI FULLY_CONNECTED (deprecated since Android 15)";
    i.available = true;
    i.kernels = {Kernel::Dense};
    return i;
  };
  uint32_t count = 0;
  if (api->getDeviceCount && api->getDevice && api->deviceGetName) {
    const int rcCount = api->getDeviceCount(&count);
    HPC_LOGI("[NNAPI] ANeuralNetworks_getDeviceCount -> rc=%d count=%u", rcCount, unsigned(count));
    if (rcCount == NN_NO_ERROR && count > 0) {
      if (count > 64) {
        HPC_LOGW("[NNAPI] clamping unreasonable device count %u -> 64", unsigned(count));
        count = 64;
      }
      for (uint32_t d = 0; d < count; ++d) {
        NnDevice* dev = nullptr;
        const int rcDev = api->getDevice(d, &dev);
        if (rcDev != NN_NO_ERROR || !dev) {
          HPC_LOGW("[NNAPI] ANeuralNetworks_getDevice(%u) failed: rc=%d dev=%p",
                   unsigned(d), rcDev, static_cast<void*>(dev));
          continue;
        }
        const char* rawName = nullptr;
        const int rcName = api->deviceGetName(dev, &rawName);
        const std::string safeName = (rcName == NN_NO_ERROR && rawName) ? sanitizeAscii(rawName, 256) : "unknown";
        int32_t type = 0;
        int rcType = -1;
        if (api->deviceGetType) {
          rcType = api->deviceGetType(dev, &type);
          if (rcType != NN_NO_ERROR) type = 0;
        }
        static const char* tn[] = {"unknown", "other", "cpu", "gpu", "accelerator"};
        BackendInfo i = base("nnapi:" + std::to_string(d));
        i.device = safeName.empty() ? "unknown" : safeName;
        i.detail["device_type"] = (type >= 0 && type < 5) ? tn[type] : "unknown";
        int64_t fl = 0;
        int rcFl = -1;
        if (api->deviceGetFeatureLevel) {
          rcFl = api->deviceGetFeatureLevel(dev, &fl);
          if (rcFl == NN_NO_ERROR) i.detail["feature_level"] = std::to_string(fl);
        }
        if (i.device == "nnapi-reference") i.note = "CPU reference implementation, NOT an NPU";
        else if (type == 4) i.note = "accelerator (NPU/DSP) as reported by the vendor driver";
        HPC_LOGI("[NNAPI] device[%u]: name='%s' (rc=%d) type=%d (%s, rc=%d) featureLevel=%lld (rc=%d)",
                 unsigned(d), i.device.c_str(), rcName, int(type), i.detail["device_type"].c_str(),
                 rcType, static_cast<long long>(fl), rcFl);
        out.emplace_back(new NnBackend(api, dev, i));
      }
    } else if (rcCount != NN_NO_ERROR) {
      count = 0;
    }
  }
  BackendInfo a = base("nnapi:auto");
  a.device = "runtime-selected";
  a.note = count == 0 ? "device enumeration unavailable (NNAPI < 29); runtime chooses the device"
                      : "runtime may split or fall back across the devices above";
  HPC_LOGI("[NNAPI] added fallback backend nnapi:auto (enumerated count=%u)", unsigned(count));
  out.emplace_back(new NnBackend(api, nullptr, a));
}

}  // namespace hpc

#else

namespace hpc {
void discoverNnapi(std::vector<std::unique_ptr<Backend>>& out) {
  BackendInfo i;
  i.id = "nnapi";
  i.api = "NNAPI";
  i.note = "Android-only backend; not available on this platform";
  out.emplace_back(new StubBackend(i));
}
}  // namespace hpc

#endif
