// SPDX-License-Identifier: Apache-2.0
// Vulkan compute backend. SPIR-V comes from the GLSL in core/shaders, compiled at build
// time by glslc (SPIR-V 1.0 / Vulkan 1.0 target for maximum compatibility). The Vulkan
// loader is opened at run time, so the app still starts where Vulkan is absent.
#define VK_NO_PROTOTYPES
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <set>

#include "backend_util.hpp"
#include "hpc/dynlib.hpp"
#include "hpc/log.hpp"

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

#include "spv_sgemm.inc"
#include "spv_conv3x3.inc"
#include "spv_reduce.inc"
#include "spv_triad.inc"

namespace hpc {
namespace {

#define HPC_VK_INSTANCE(X)                                                                        \
  X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)             \
  X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceMemoryProperties)                           \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)            \
  X(vkEnumerateDeviceExtensionProperties)

#define HPC_VK_DEVICE(X)                                                                          \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer)                     \
  X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory)      \
  X(vkMapMemory) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges)                                    \
  X(vkInvalidateMappedMemoryRanges) X(vkCreateShaderModule) X(vkDestroyShaderModule)              \
  X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout)        \
  X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline)                     \
  X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets)                \
  X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool)                        \
  X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkBeginCommandBuffer)                     \
  X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants)     \
  X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdWriteTimestamp)              \
  X(vkCmdResetQueryPool) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkGetQueryPoolResults)      \
  X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) X(vkWaitForFences) X(vkQueueSubmit)         \
  X(vkDeviceWaitIdle)

struct InstFns {
#define X(n) PFN_##n n = nullptr;
  HPC_VK_INSTANCE(X)
#undef X
  PFN_vkGetPhysicalDeviceProperties2 getProps2 = nullptr;
};
struct DevFns {
#define X(n) PFN_##n n = nullptr;
  HPC_VK_DEVICE(X)
#undef X
#if defined(__ANDROID__)
  PFN_vkGetAndroidHardwareBufferPropertiesANDROID getAhbProps = nullptr;
#endif
};

std::string vkr(VkResult r) { return "VkResult " + std::to_string(int(r)); }

struct Runtime {
  DynLib lib;
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  VkInstance inst = VK_NULL_HANDLE;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  InstFns f;
  std::vector<VkPhysicalDevice> phys;
  std::string error;

  bool init() {
    std::string tried;
    if (!lib.open({
#if defined(_WIN32)
            "vulkan-1.dll"
#else
            "libvulkan.so", "libvulkan.so.1"
#endif
        },
                  &tried)) {
      error = "Vulkan loader not found (tried: " + tried + ")";
      return false;
    }
    gipa = lib.sym<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (!gipa) {
      error = "vkGetInstanceProcAddr missing";
      return false;
    }
    auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    auto enumVer = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(gipa(nullptr, "vkEnumerateInstanceVersion"));
    if (!createInstance) {
      error = "vkCreateInstance missing";
      return false;
    }
    uint32_t loaderVer = VK_API_VERSION_1_0;
    if (enumVer && enumVer(&loaderVer) != VK_SUCCESS) loaderVer = VK_API_VERSION_1_0;
    apiVersion = loaderVer >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;
    HPC_LOGI("[VK] loader version %u.%u.%u (requesting instance API %u.%u)",
             VK_VERSION_MAJOR(loaderVer), VK_VERSION_MINOR(loaderVer), VK_VERSION_PATCH(loaderVer),
             VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "android-hpc";
    app.apiVersion = apiVersion;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkResult r = createInstance(&ci, nullptr, &inst);
    if (r != VK_SUCCESS || !inst) {
      error = "vkCreateInstance failed: " + vkr(r);
      HPC_LOGW("[VK] %s", error.c_str());
      return false;
    }
    HPC_LOGI("[VK] vkCreateInstance succeeded (%p)", static_cast<void*>(inst));
#define X(n) f.n = reinterpret_cast<PFN_##n>(gipa(inst, #n));
    HPC_VK_INSTANCE(X)
#undef X
    if (apiVersion >= VK_API_VERSION_1_1) {
      f.getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(inst, "vkGetPhysicalDeviceProperties2"));
    }
    if (!f.vkEnumeratePhysicalDevices || !f.vkGetPhysicalDeviceProperties ||
        !f.vkGetPhysicalDeviceFeatures || !f.vkGetPhysicalDeviceMemoryProperties ||
        !f.vkGetPhysicalDeviceQueueFamilyProperties || !f.vkCreateDevice ||
        !f.vkGetDeviceProcAddr || !f.vkEnumerateDeviceExtensionProperties) {
      error = "incomplete Vulkan instance function table";
      HPC_LOGW("[VK] %s", error.c_str());
      return false;
    }
    uint32_t n = 0;
    r = f.vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      error = "vkEnumeratePhysicalDevices(count) failed: " + vkr(r);
      HPC_LOGW("[VK] %s", error.c_str());
      return false;
    }
    if (n > 64) {
      HPC_LOGW("[VK] clamping unreasonable physical device count %u -> 64", n);
      n = 64;
    }
    phys.resize(n, VK_NULL_HANDLE);
    if (n) {
      r = f.vkEnumeratePhysicalDevices(inst, &n, phys.data());
      if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        error = "vkEnumeratePhysicalDevices(list) failed: " + vkr(r);
        HPC_LOGW("[VK] %s", error.c_str());
        return false;
      }
      phys.resize(n);
    }
    phys.erase(std::remove(phys.begin(), phys.end(), VK_NULL_HANDLE), phys.end());
    HPC_LOGI("[VK] enumerated %zu physical device(s)", phys.size());
    if (phys.empty()) {
      error = "Vulkan present but reports no physical devices";
      return false;
    }
    return true;
  }
  ~Runtime() {
    // Deliberately do NOT call vkDestroyInstance(inst, nullptr) during the process
    // lifetime. On Android, vkDestroyInstance causes libvulkan.so to dlclose() the
    // vendor ICD shared library while ICD worker threads or TLS destructors may still
    // be active, causing SIGSEGV after discovery returns.
  }
};

struct Buf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  uint8_t* ptr = nullptr;  // non-null when the memory is host-mapped (direct path)
  bool coherent = true;
  std::unique_ptr<Buf> stage;  // host staging buffer for device-local-only memory
#if defined(__ANDROID__)
  AHardwareBuffer* ahb = nullptr;
#endif
};

class Gpu {
 public:
  Gpu(std::shared_ptr<Runtime> rt, VkPhysicalDevice pd) : rt(std::move(rt)), pd(pd) {
    if (pd && this->rt && this->rt->f.vkGetPhysicalDeviceProperties) {
      this->rt->f.vkGetPhysicalDeviceProperties(pd, &props);
      props.deviceName[sizeof(props.deviceName) - 1] = '\0';
    }
    if (pd && this->rt && this->rt->f.vkGetPhysicalDeviceMemoryProperties) {
      this->rt->f.vkGetPhysicalDeviceMemoryProperties(pd, &mem);
      if (mem.memoryTypeCount > VK_MAX_MEMORY_TYPES) mem.memoryTypeCount = VK_MAX_MEMORY_TYPES;
      if (mem.memoryHeapCount > VK_MAX_MEMORY_HEAPS) mem.memoryHeapCount = VK_MAX_MEMORY_HEAPS;
    }
  }
  ~Gpu() {
    if (dev) {
      if (d.vkDeviceWaitIdle) d.vkDeviceWaitIdle(dev);
      if (pool) d.vkDestroyCommandPool(dev, pool, nullptr);
      d.vkDestroyDevice(dev, nullptr);
    }
  }

  std::shared_ptr<Runtime> rt;
  VkPhysicalDevice pd;
  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceMemoryProperties mem{};
  VkDevice dev = VK_NULL_HANDLE;
  DevFns d;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t qfi = 0;
  uint32_t tsBits = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  bool ahbEnabled = false;

  bool uma() const {
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      return true;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return false;
    // OTHER / VIRTUAL: unified if every device-local memory type is host visible.
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
      const auto fl = mem.memoryTypes[i].propertyFlags;
      if ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return false;
    }
    return true;
  }

  static std::set<std::string> deviceExtensions(const Runtime& rt, VkPhysicalDevice pd) {
    std::set<std::string> s;
    if (!rt.f.vkEnumerateDeviceExtensionProperties || !pd) return s;
    uint32_t n = 0;
    VkResult r = rt.f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0) return s;
    if (n > 4096) n = 4096;
    std::vector<VkExtensionProperties> v(n);
    r = rt.f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, v.data());
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return s;
    v.resize(std::min<size_t>(n, v.size()));
    for (auto& e : v) {
      e.extensionName[sizeof(e.extensionName) - 1] = '\0';
      s.insert(sanitizeAscii(e.extensionName));
    }
    return s;
  }

  bool init(std::string* err) {
    if (dev) return true;
    auto& f = rt->f;
    uint32_t nq = 0;
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qs.data());
    qfi = UINT32_MAX;
    // Prefer a compute-only family (async compute), else any compute-capable one.
    for (uint32_t i = 0; i < nq; ++i)
      if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        qfi = i;
        break;
      }
    if (qfi == UINT32_MAX)
      for (uint32_t i = 0; i < nq; ++i)
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          qfi = i;
          break;
        }
    if (qfi == UINT32_MAX) {
      if (err) *err = "no compute queue family";
      return false;
    }
    tsBits = qs[qfi].timestampValidBits;

    std::vector<const char*> exts;
#if defined(__ANDROID__)
    const auto have = deviceExtensions(*rt, pd);
    if (rt->apiVersion >= VK_API_VERSION_1_1 && have.count(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME) &&
        have.count(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
      exts.push_back(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
      exts.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
      // Core in 1.1, but enable the KHR names when the driver lists them.
      for (const char* e : {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
                            VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME})
        if (have.count(e)) exts.push_back(e);
      ahbEnabled = true;
    }
#endif
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    VkResult r = f.vkCreateDevice(pd, &dci, nullptr, &dev);
    if (r != VK_SUCCESS || !dev) {
      dev = VK_NULL_HANDLE;
      if (err) *err = "vkCreateDevice failed: " + vkr(r);
      HPC_LOGW("[VK] vkCreateDevice failed: %s", vkr(r).c_str());
      return false;
    }
    HPC_LOGI("[VK] vkCreateDevice ok (%p), queueFamily=%u, exts=%zu", static_cast<void*>(dev), qfi, exts.size());
#define X(n) d.n = reinterpret_cast<PFN_##n>(f.vkGetDeviceProcAddr(dev, #n));
    HPC_VK_DEVICE(X)
#undef X
    if (!d.vkGetDeviceQueue || !d.vkCreateCommandPool || !d.vkCreateBuffer ||
        !d.vkAllocateMemory || !d.vkCreateComputePipelines || !d.vkQueueSubmit) {
      if (err) *err = "incomplete Vulkan device function table";
      HPC_LOGW("[VK] incomplete Vulkan device function table");
      return false;
    }
#if defined(__ANDROID__)
    if (ahbEnabled)
      d.getAhbProps = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
          f.vkGetDeviceProcAddr(dev, "vkGetAndroidHardwareBufferPropertiesANDROID"));
    if (!d.getAhbProps) ahbEnabled = false;
#endif
    d.vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfi;
    r = d.vkCreateCommandPool(dev, &pci, nullptr, &pool);
    if (r != VK_SUCCESS) {
      if (err) *err = "vkCreateCommandPool failed: " + vkr(r);
      HPC_LOGW("[VK] vkCreateCommandPool failed: %s", vkr(r).c_str());
      return false;
    }
    return true;
  }

  int findType(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) return int(i);
    return -1;
  }

  bool createRaw(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, Buf& b, std::string* err) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = d.vkCreateBuffer(dev, &bi, nullptr, &b.buf);
    if (r != VK_SUCCESS) {
      b.buf = VK_NULL_HANDLE;
      if (err) *err = "vkCreateBuffer: " + vkr(r);
      return false;
    }
    VkMemoryRequirements mr;
    d.vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    const int t = findType(mr.memoryTypeBits, want);
    if (t < 0) {
      if (err) *err = "no memory type with requested properties";
      return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = uint32_t(t);
    r = d.vkAllocateMemory(dev, &ai, nullptr, &b.mem);
    if (r != VK_SUCCESS) {
      b.mem = VK_NULL_HANDLE;
      if (err) *err = "vkAllocateMemory: " + vkr(r);
      return false;
    }
    r = d.vkBindBufferMemory(dev, b.buf, b.mem, 0);
    if (r != VK_SUCCESS) {
      if (err) *err = "vkBindBufferMemory: " + vkr(r);
      return false;
    }
    b.size = size;
    b.coherent = (mem.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      void* p = nullptr;
      r = d.vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p);
      if (r != VK_SUCCESS) {
        if (err) *err = "vkMapMemory: " + vkr(r);
        return false;
      }
      b.ptr = static_cast<uint8_t*>(p);
    }
    return true;
  }

  void destroy(Buf& b) {
    if (b.stage) destroy(*b.stage);
    if (b.ptr) d.vkUnmapMemory(dev, b.mem);
    if (b.buf) d.vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) d.vkFreeMemory(dev, b.mem, nullptr);
#if defined(__ANDROID__)
    if (b.ahb) AHardwareBuffer_release(b.ahb);
#endif
    b = Buf{};
  }

  static constexpr VkBufferUsageFlags kUsage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  // Storage buffer. Unified-memory GPUs (phones, iGPUs) use host-visible device memory directly;
  // discrete GPUs use device-local memory plus a staging buffer.
  bool makeStorage(VkDeviceSize size, Buf& b, std::string* err) {
    size = std::max<VkDeviceSize>(16, (size + 3) & ~VkDeviceSize(3));
    if (uma()) {
      const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      const VkMemoryPropertyFlags co = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      const VkMemoryPropertyFlags dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      for (VkMemoryPropertyFlags want : {dl | hv | co, hv | co, hv}) {
        if (findType(0xFFFFFFFFu, want) < 0) continue;
        std::string e;
        if (createRaw(size, kUsage, want, b, &e)) return true;
        destroy(b);
      }
      if (err) *err = "no usable host-visible memory type";
      return false;
    }
    if (!createRaw(size, kUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b, err)) return false;
    b.stage.reset(new Buf);
    if (!createRaw(size, kUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, *b.stage, err))
      return false;
    return true;
  }

  bool oneShot(const std::function<void(VkCommandBuffer)>& fn, std::string* err) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VkResult r = d.vkAllocateCommandBuffers(dev, &ai, &cb);
    if (r != VK_SUCCESS) {
      if (err) *err = "vkAllocateCommandBuffers: " + vkr(r);
      return false;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d.vkBeginCommandBuffer(cb, &bi);
    fn(cb);
    d.vkEndCommandBuffer(cb);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    d.vkCreateFence(dev, &fi, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    r = d.vkQueueSubmit(queue, 1, &si, fence);
    if (r == VK_SUCCESS) r = d.vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    d.vkDestroyFence(dev, fence, nullptr);
    d.vkFreeCommandBuffers(dev, pool, 1, &cb);
    if (r != VK_SUCCESS) {
      if (err) *err = "queue submit/wait: " + vkr(r);
      return false;
    }
    return true;
  }

  void flush(Buf& b, bool invalidate) {
    if (b.coherent || !b.ptr) return;
    VkMappedMemoryRange rg{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    rg.memory = b.mem;
    rg.size = VK_WHOLE_SIZE;
    if (invalidate) d.vkInvalidateMappedMemoryRanges(dev, 1, &rg);
    else d.vkFlushMappedMemoryRanges(dev, 1, &rg);
  }

  bool upload(Buf& b, const void* data, size_t bytes, std::string* err) {
    if (b.ptr && !b.stage) {
      std::memcpy(b.ptr, data, bytes);
      flush(b, false);
      return true;
    }
    if (!b.stage) {  // imported AHardwareBuffer etc.: written elsewhere
      if (err) *err = "buffer not writable from host";
      return false;
    }
    std::memcpy(b.stage->ptr, data, bytes);
    flush(*b.stage, false);
    return oneShot(
        [&](VkCommandBuffer cb) {
          VkBufferCopy c{0, 0, VkDeviceSize(bytes)};
          d.vkCmdCopyBuffer(cb, b.stage->buf, b.buf, 1, &c);
          VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          d.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
        },
        err);
  }

  bool download(Buf& b, void* data, size_t bytes, std::string* err) {
    if (b.ptr && !b.stage) {
      flush(b, true);
      std::memcpy(data, b.ptr, bytes);
      return true;
    }
    if (!b.stage) {
      if (err) *err = "buffer not readable";
      return false;
    }
    if (!oneShot(
            [&](VkCommandBuffer cb) {
              VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
              mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
              d.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb,
                                     0, nullptr, 0, nullptr);
              VkBufferCopy c{0, 0, VkDeviceSize(bytes)};
              d.vkCmdCopyBuffer(cb, b.buf, b.stage->buf, 1, &c);
            },
            err))
      return false;
    flush(*b.stage, true);
    std::memcpy(data, b.stage->ptr, bytes);
    return true;
  }

#if defined(__ANDROID__)
  // Zero-copy input: allocate an AHardwareBuffer (BLOB), fill it from the CPU, import it as a
  // VkBuffer. Returns false (with reason) on any failure; the caller then falls back to a normal copy.
  bool importAhbInput(const void* data, size_t bytes, Buf& b, std::string* err) {
    if (!ahbEnabled) {
      if (err) *err = "VK_ANDROID_external_memory_android_hardware_buffer not enabled";
      return false;
    }
    bytes = std::max<size_t>(16, (bytes + 3) & ~size_t(3));
    AHardwareBuffer_Desc desc{};
    desc.width = uint32_t(bytes);
    desc.height = 1;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    AHardwareBuffer* ahb = nullptr;
    if (AHardwareBuffer_allocate(&desc, &ahb) != 0 || !ahb) {
      if (err) *err = "AHardwareBuffer_allocate failed";
      return false;
    }
    void* p = nullptr;
    if (AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &p) != 0 || !p) {
      AHardwareBuffer_release(ahb);
      if (err) *err = "AHardwareBuffer_lock failed";
      return false;
    }
    std::memset(p, 0, bytes);
    std::memcpy(p, data, bytes);
    AHardwareBuffer_unlock(ahb, nullptr);
    b.ahb = ahb;

    VkAndroidHardwareBufferPropertiesANDROID ap{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkResult r = d.getAhbProps(dev, ahb, &ap);
    if (r != VK_SUCCESS) {
      if (err) *err = "vkGetAndroidHardwareBufferPropertiesANDROID: " + vkr(r);
      return false;
    }
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ext;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    r = d.vkCreateBuffer(dev, &bi, nullptr, &b.buf);
    if (r != VK_SUCCESS) {
      b.buf = VK_NULL_HANDLE;
      if (err) *err = "vkCreateBuffer(external): " + vkr(r);
      return false;
    }
    VkMemoryRequirements mr;
    d.vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    const uint32_t bits = mr.memoryTypeBits & ap.memoryTypeBits;
    int t = -1;
    for (uint32_t i = 0; i < mem.memoryTypeCount && t < 0; ++i)
      if (bits & (1u << i)) t = int(i);
    if (t < 0) {
      if (err) *err = "no memory type common to buffer and AHardwareBuffer";
      return false;
    }
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = b.buf;
    VkImportAndroidHardwareBufferInfoANDROID imp{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
    imp.buffer = ahb;
    imp.pNext = &ded;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &imp;
    ai.allocationSize = ap.allocationSize;
    ai.memoryTypeIndex = uint32_t(t);
    r = d.vkAllocateMemory(dev, &ai, nullptr, &b.mem);
    if (r != VK_SUCCESS) {
      b.mem = VK_NULL_HANDLE;
      if (err) *err = "vkAllocateMemory(import AHB): " + vkr(r);
      return false;
    }
    r = d.vkBindBufferMemory(dev, b.buf, b.mem, 0);
    if (r != VK_SUCCESS) {
      if (err) *err = "vkBindBufferMemory(import AHB): " + vkr(r);
      return false;
    }
    b.size = bytes;
    b.coherent = true;
    return true;
  }
#endif
};

struct Push {
  uint32_t m, k, n, mode;
  float alpha;
  uint32_t pad[3];
};

class VkExecution final : public Execution {
 public:
  VkExecution(std::shared_ptr<Gpu> g) : g_(std::move(g)) {}
  ~VkExecution() override { release(); }

  bool build(const Problem& p, const PrepareOptions& opt, std::string* err) {
    Gpu& g = *g_;
    auto& d = g.d;
    p_ = &p;
    outCount_ = p.outputCount();
    const size_t F = sizeof(float);

    if (g.props.limits.maxComputeWorkGroupInvocations < 256 || g.props.limits.maxComputeSharedMemorySize < 4096) {
      if (err) *err = "device cannot run 256-invocation workgroups";
      return false;
    }

    // ---- buffers ----
    auto make = [&](Buf& b, size_t bytes) { return g.makeStorage(bytes, b, err); };
    std::vector<float> zero16(4, 0.0f);
    bool needB = false, needBias = false;
    size_t aBytes = 0, bBytes = 0, biasBytes = 0, cBytes = outCount_ * F;
    const float *aData = p.a.data(), *bData = nullptr, *biasData = nullptr;
    switch (p.kernel) {
      case Kernel::Sgemm: aBytes = size_t(p.m) * p.k * F; bBytes = size_t(p.k) * p.n * F; needB = true; bData = p.b.data(); break;
      case Kernel::Dense: aBytes = size_t(p.m) * p.k * F; bBytes = size_t(p.k) * p.n * F; needB = needBias = true;
        bData = p.b.data(); biasBytes = size_t(p.n) * F; biasData = p.bias.data(); break;
      case Kernel::Conv3x3: aBytes = size_t(p.m) * p.n * F; bBytes = 9 * F; needB = true; bData = p.b.data(); break;
      case Kernel::Reduce: aBytes = size_t(p.n) * F; break;
      case Kernel::Triad: aBytes = size_t(p.n) * F; bBytes = size_t(p.n) * F; needB = true; bData = p.b.data(); break;
    }

    zeroCopy_ = false;
#if defined(__ANDROID__)
    bool zcA = false, zcB = false;
    if (opt.zeroCopy) {
      std::string e;
      zcA = g.importAhbInput(aData, aBytes, a_, &e);
      if (!zcA) { g.destroy(a_); zcNote_ = e; }
      if (needB) {
        zcB = g.importAhbInput(bData, bBytes, b_, &e);
        if (!zcB) { g.destroy(b_); if (zcNote_.empty()) zcNote_ = e; }
      }
      zeroCopy_ = zcA && (!needB || zcB);
      if (!zeroCopy_) {
        // Previously only stored in the report detail, so a device log never said why.
        HPC_LOGI("[VK] zero-copy AHardwareBuffer import not used, falling back to copy: %s",
                 zcNote_.empty() ? "(no reason reported)" : zcNote_.c_str());
      }
    }
    if (!zcA) { if (!make(a_, aBytes) || !g.upload(a_, aData, aBytes, err)) return false; }
    if (needB && !zcB) { if (!make(b_, bBytes) || !g.upload(b_, bData, bBytes, err)) return false; }
#else
    (void)opt;
    if (!make(a_, aBytes) || !g.upload(a_, aData, aBytes, err)) return false;
    if (needB) { if (!make(b_, bBytes) || !g.upload(b_, bData, bBytes, err)) return false; }
#endif
    if (needBias) { if (!make(bias_, biasBytes) || !g.upload(bias_, biasData, biasBytes, err)) return false; }
    if (!make(dummy_, 16) || !g.upload(dummy_, zero16.data(), 16, err)) return false;

    constexpr uint32_t kReduceGroups = 256;
    const bool reduce = p.kernel == Kernel::Reduce;
    if (reduce) {
      if (!make(partial_, kReduceGroups * F)) return false;
      if (!make(out_, F)) return false;
    } else {
      if (!make(out_, cBytes)) return false;
    }

    // ---- pipeline ----
    const unsigned char* spv = nullptr;
    size_t spvSize = 0;
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense: spv = kSpv_sgemm; spvSize = kSpv_sgemm_size; break;
      case Kernel::Conv3x3: spv = kSpv_conv3x3; spvSize = kSpv_conv3x3_size; break;
      case Kernel::Reduce: spv = kSpv_reduce; spvSize = kSpv_reduce_size; break;
      case Kernel::Triad: spv = kSpv_triad; spvSize = kSpv_triad_size; break;
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spvSize;
    smi.pCode = reinterpret_cast<const uint32_t*>(spv);
    VkResult r = d.vkCreateShaderModule(g.dev, &smi, nullptr, &sm_);
    if (r != VK_SUCCESS) { if (err) *err = "vkCreateShaderModule: " + vkr(r); return false; }

    VkDescriptorSetLayoutBinding lb[4];
    for (uint32_t i = 0; i < 4; ++i)
      lb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 4;
    dli.pBindings = lb;
    r = d.vkCreateDescriptorSetLayout(g.dev, &dli, nullptr, &dsl_);
    if (r != VK_SUCCESS) { if (err) *err = "vkCreateDescriptorSetLayout: " + vkr(r); return false; }
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    r = d.vkCreatePipelineLayout(g.dev, &pli, nullptr, &pl_);
    if (r != VK_SUCCESS) { if (err) *err = "vkCreatePipelineLayout: " + vkr(r); return false; }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm_, "main", nullptr};
    cpi.layout = pl_;
    r = d.vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe_);
    if (r != VK_SUCCESS) { if (err) *err = "vkCreateComputePipelines: " + vkr(r); return false; }

    // ---- descriptors (set 0: main pass, set 1: reduce pass 2) ----
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    r = d.vkCreateDescriptorPool(g.dev, &dpi, nullptr, &dp_);
    if (r != VK_SUCCESS) { if (err) *err = "vkCreateDescriptorPool: " + vkr(r); return false; }
    VkDescriptorSetLayout layouts[2] = {dsl_, dsl_};
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = dp_;
    dsa.descriptorSetCount = 2;
    dsa.pSetLayouts = layouts;
    r = d.vkAllocateDescriptorSets(g.dev, &dsa, ds_);
    if (r != VK_SUCCESS) { if (err) *err = "vkAllocateDescriptorSets: " + vkr(r); return false; }

    auto bind = [&](VkDescriptorSet set, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3) {
      VkBuffer bs[4] = {b0, b1, b2, b3};
      VkDescriptorBufferInfo info[4];
      VkWriteDescriptorSet w[4];
      for (uint32_t i = 0; i < 4; ++i) {
        info[i] = {bs[i] ? bs[i] : dummy_.buf, 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info[i], nullptr};
      }
      d.vkUpdateDescriptorSets(g.dev, 4, w, 0, nullptr);
    };
    if (reduce) {
      bind(ds_[0], a_.buf, VK_NULL_HANDLE, partial_.buf, VK_NULL_HANDLE);
      bind(ds_[1], partial_.buf, VK_NULL_HANDLE, out_.buf, VK_NULL_HANDLE);
    } else {
      bind(ds_[0], a_.buf, needB ? b_.buf : VK_NULL_HANDLE, out_.buf, needBias ? bias_.buf : VK_NULL_HANDLE);
    }

    // ---- timestamps + fence ----
    hasTs_ = g.tsBits > 0 && g.props.limits.timestampPeriod > 0;
    if (hasTs_) {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = 2;
      if (d.vkCreateQueryPool(g.dev, &qi, nullptr, &qp_) != VK_SUCCESS) hasTs_ = false;
    }
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    d.vkCreateFence(g.dev, &fi, nullptr, &fence_);

    // ---- record the (re-submittable) command buffer ----
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    r = d.vkAllocateCommandBuffers(g.dev, &cai, &cb_);
    if (r != VK_SUCCESS) { if (err) *err = "vkAllocateCommandBuffers: " + vkr(r); return false; }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    d.vkBeginCommandBuffer(cb_, &bi);
    if (hasTs_) {
      d.vkCmdResetQueryPool(cb_, qp_, 0, 2);
      d.vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp_, 0);
    }
    d.vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    Push pc{p.m, p.k, p.n, p.kernel == Kernel::Dense ? 1u : 0u, p.alpha, {0, 0, 0}};
    auto dispatch = [&](VkDescriptorSet set, const Push& push, uint32_t gx, uint32_t gy) {
      d.vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pl_, 0, 1, &set, 0, nullptr);
      d.vkCmdPushConstants(cb_, pl_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
      d.vkCmdDispatch(cb_, gx, gy, 1);
    };
    auto barrier = [&](VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags dstStage) {
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, src, dst};
      d.vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    switch (p.kernel) {
      case Kernel::Sgemm: case Kernel::Dense: case Kernel::Conv3x3:
        dispatch(ds_[0], pc, (p.n + 15) / 16, (p.m + 15) / 16);
        break;
      case Kernel::Reduce: {
        const uint32_t groups = std::min<uint32_t>(kReduceGroups, std::max<uint32_t>(1, (p.n + 255) / 256));
        dispatch(ds_[0], pc, groups, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Push p2 = pc;
        p2.n = groups;  // pass 2 sums the per-workgroup partials with one workgroup
        dispatch(ds_[1], p2, 1, 1);
        break;
      }
      case Kernel::Triad:
        dispatch(ds_[0], pc, std::min<uint32_t>(8192, std::max<uint32_t>(1, (p.n + 255) / 256)), 1);
        break;
    }
    barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT);
    if (hasTs_) d.vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp_, 1);
    d.vkEndCommandBuffer(cb_);
    return true;
  }

  bool run(double* deviceMs, std::string* err) override {
    Gpu& g = *g_;
    if (deviceMs) *deviceMs = -1;
    g.d.vkResetFences(g.dev, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb_;
    VkResult r = g.d.vkQueueSubmit(g.queue, 1, &si, fence_);
    if (r == VK_SUCCESS) r = g.d.vkWaitForFences(g.dev, 1, &fence_, VK_TRUE, 30ull * 1000 * 1000 * 1000);
    if (r != VK_SUCCESS) {
      if (err) *err = "submit/wait: " + vkr(r);
      return false;
    }
    if (hasTs_ && deviceMs) {
      uint64_t t[2] = {0, 0};
      if (g.d.vkGetQueryPoolResults(g.dev, qp_, 0, 2, sizeof t, t, sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        const uint64_t mask = g.tsBits >= 64 ? ~0ull : ((1ull << g.tsBits) - 1);
        const uint64_t dt = (t[1] - t[0]) & mask;
        *deviceMs = double(dt) * double(g.props.limits.timestampPeriod) * 1e-6;
      }
    }
    return true;
  }

  bool read(std::vector<float>& out, std::string* err) override {
    out.resize(outCount_);
    return g_->download(out_, out.data(), outCount_ * sizeof(float), err);
  }
  bool zeroCopy() const override { return zeroCopy_; }
  const std::string& zeroCopyNote() const { return zcNote_; }

 private:
  void release() {
    Gpu& g = *g_;
    if (!g.dev) return;
    auto& d = g.d;
    d.vkDeviceWaitIdle(g.dev);
    if (cb_) d.vkFreeCommandBuffers(g.dev, g.pool, 1, &cb_);
    if (fence_) d.vkDestroyFence(g.dev, fence_, nullptr);
    if (qp_) d.vkDestroyQueryPool(g.dev, qp_, nullptr);
    if (dp_) d.vkDestroyDescriptorPool(g.dev, dp_, nullptr);
    if (pipe_) d.vkDestroyPipeline(g.dev, pipe_, nullptr);
    if (pl_) d.vkDestroyPipelineLayout(g.dev, pl_, nullptr);
    if (dsl_) d.vkDestroyDescriptorSetLayout(g.dev, dsl_, nullptr);
    if (sm_) d.vkDestroyShaderModule(g.dev, sm_, nullptr);
    for (Buf* b : {&a_, &b_, &bias_, &dummy_, &partial_, &out_}) g.destroy(*b);
  }

  std::shared_ptr<Gpu> g_;
  const Problem* p_ = nullptr;
  size_t outCount_ = 0;
  Buf a_, b_, bias_, dummy_, partial_, out_;
  VkShaderModule sm_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkPipelineLayout pl_ = VK_NULL_HANDLE;
  VkPipeline pipe_ = VK_NULL_HANDLE;
  VkDescriptorPool dp_ = VK_NULL_HANDLE;
  VkDescriptorSet ds_[2] = {};
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkQueryPool qp_ = VK_NULL_HANDLE;
  bool hasTs_ = false;
  bool zeroCopy_ = false;
  std::string zcNote_;
};

class VulkanBackend final : public Backend {
 public:
  VulkanBackend(std::shared_ptr<Runtime> rt, VkPhysicalDevice pd, int index) : gpu_(new Gpu(rt, pd)) {
    const auto& pr = gpu_->props;
    static const char* types[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    info_.id = "vulkan:" + std::to_string(index);
    info_.api = sanitizeAscii("Vulkan " + std::to_string(VK_VERSION_MAJOR(pr.apiVersion)) + "." +
                              std::to_string(VK_VERSION_MINOR(pr.apiVersion)) + " compute (SPIR-V)", 256);
    info_.device = sanitizeAscii(pr.deviceName, 256);
    info_.vendor = sanitizeAscii(vendorName(pr.vendorID), 256);
    info_.driver = sanitizeAscii(hexStr(pr.driverVersion), 256);
    info_.kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
    info_.available = pr.limits.maxComputeWorkGroupInvocations >= 256;
    if (!info_.available) info_.note = "maxComputeWorkGroupInvocations < 256";
    auto& dt = info_.detail;
    dt["device_type"] = pr.deviceType < 5 ? types[pr.deviceType] : "other";
    dt["memory_model"] = gpu_->uma() ? "unified (host-visible device memory, no staging)" : "discrete (device-local + staging)";
    dt["max_workgroup_invocations"] = std::to_string(pr.limits.maxComputeWorkGroupInvocations);
    dt["max_shared_memory_bytes"] = std::to_string(pr.limits.maxComputeSharedMemorySize);
    dt["timestamp_period_ns"] = std::to_string(pr.limits.timestampPeriod);
    VkPhysicalDeviceFeatures ft{};
    rt->f.vkGetPhysicalDeviceFeatures(pd, &ft);
    dt["shaderFloat64"] = ft.shaderFloat64 ? "yes" : "no";
    if (rt->apiVersion >= VK_API_VERSION_1_1 && pr.apiVersion >= VK_API_VERSION_1_1 && rt->f.getProps2) {
      VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
      VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
      rt->f.getProps2(pd, &p2);
      dt["subgroup_size"] = std::to_string(sg.subgroupSize);
    }
    size_t extCount = 0;
#if defined(__ANDROID__)
    const auto have = Gpu::deviceExtensions(*rt, pd);
    extCount = have.size();
    dt["ahb_import_extension"] = (have.count(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME) &&
                                  have.count(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME))
                                     ? "listed (import path unverified on this device until run with zero-copy)"
                                     : "not listed";
#endif
    HPC_LOGI("[VK] device %s: name='%s' vendor='%s' api=%u.%u.%u driver=%s type=%s memTypes=%u maxInv=%u sharedMem=%u exts=%zu available=%d",
             info_.id.c_str(), info_.device.c_str(), info_.vendor.c_str(),
             VK_VERSION_MAJOR(pr.apiVersion), VK_VERSION_MINOR(pr.apiVersion), VK_VERSION_PATCH(pr.apiVersion),
             info_.driver.c_str(), dt["device_type"].c_str(), unsigned(gpu_->mem.memoryTypeCount),
             unsigned(pr.limits.maxComputeWorkGroupInvocations), unsigned(pr.limits.maxComputeSharedMemorySize),
             extCount, info_.available ? 1 : 0);
  }
  const BackendInfo& info() const override { return info_; }
  std::unique_ptr<Execution> prepare(const Problem& p, const PrepareOptions& opt, std::string* err) override {
    if (!gpu_->init(err)) return nullptr;
    std::unique_ptr<VkExecution> e(new VkExecution(gpu_));
    if (!e->build(p, opt, err)) return nullptr;
    if (opt.zeroCopy) {
      info_.detail["zero_copy_last"] =
          e->zeroCopy() ? "AHardwareBuffer import succeeded" : sanitizeAscii("fell back to copy: " + e->zeroCopyNote(), 256);
    }
    return e;
  }

 private:
  std::shared_ptr<Gpu> gpu_;
  BackendInfo info_;
};

}  // namespace

void discoverVulkan(std::vector<std::unique_ptr<Backend>>& out) {
  auto rt = std::make_shared<Runtime>();
  if (!rt->init()) {
    BackendInfo i;
    i.id = "vulkan";
    i.api = "Vulkan compute (SPIR-V)";
    i.note = sanitizeAscii(rt->error, 256);
    HPC_LOGW("[VK] unavailable: %s", i.note.c_str());
    out.emplace_back(new StubBackend(i));
    return;
  }
  for (size_t k = 0; k < rt->phys.size(); ++k) {
    if (!rt->phys[k]) continue;
    out.emplace_back(new VulkanBackend(rt, rt->phys[k], int(k)));
  }
}

}  // namespace hpc
