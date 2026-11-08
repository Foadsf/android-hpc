// SPDX-License-Identifier: Apache-2.0
//
// SysML v2 (textual) model of android-hpc: requirements, architecture, allocation, verification.
//
// This is the corrected successor of the model in "HPC MVP on Android with SysML2 Requirements.md".
// See docs/FACT_CHECK.md for every defect that was fixed and why.
//
// STATUS: written against the OMG SysML v2 textual grammar and checked by hand. It has NOT been
// machine-validated in the environment that produced it (no SysML v2 parser could be installed
// there). Validate before relying on it, e.g. with Sensmetry SysIDE:
//     pip install syside          (then load model/AndroidHPC.sysml with syside.Environment)
// or the OMG SysML v2 Pilot Implementation. Report parser findings as issues.
//
// Traceability: each requirement names the test or command that verifies it. Requirements are
// evaluated against *measured* attributes (e.g. allResultsVerified) rather than constants, so a
// requirement cannot be "satisfied" by a hard-coded true.

package AndroidHPC {
    private import ScalarValues::*;
    private import VerificationCases::*;

    // ------------------------------------------------------------------------------------
    // 1. Stakeholders and concerns (stakeholder is a role *usage* typed by a part def)
    // ------------------------------------------------------------------------------------
    package Stakeholders {
        part def Developer;
        part def Researcher;
        part def PlatformVendor;

        concern def PortabilityConcern {
            doc /* Does it run on devices from different SoC vendors without vendor SDKs? */
            stakeholder dev : Developer;
            stakeholder vendor : PlatformVendor;
        }
        concern def MeasurementTrustConcern {
            doc /* Can a reported number be believed: verified, labelled honestly, device-timed? */
            stakeholder researcher : Researcher;
            stakeholder dev : Developer;
        }
    }

    // ------------------------------------------------------------------------------------
    // 2. Domain vocabulary
    // ------------------------------------------------------------------------------------
    package Domain {
        enum def KernelKind {
            enum sgemm;
            enum conv3x3;
            enum reduce;
            enum triad;
            enum dense;
        }
        enum def BackendKind {
            enum cpu;
            enum vulkan;
            enum opencl;
            enum gles;
            enum nnapi;
            enum syclShim;
        }

        item def Problem {
            attribute kernel : KernelKind;
            attribute m : Natural;
            attribute k : Natural;
            attribute n : Natural;
            attribute alpha : Real;
        }

        attribute def TimingStats {
            attribute min_ms : Real;
            attribute median_ms : Real;
            attribute mean_ms : Real;
            attribute p95_ms : Real;
            attribute stddev_ms : Real;
        }

        item def BenchResult {
            attribute backendId : String;
            attribute kernel : KernelKind;
            attribute wall : TimingStats;
            attribute device : TimingStats [0..1];   // absent when the API cannot time on-device
            attribute gflops : Real;
            attribute gbps : Real;
            attribute verified : Boolean;            // matches the fp64 oracle within tolerance
            attribute maxAbsErr : Real;
            attribute zeroCopy : Boolean;            // true only if an AHardwareBuffer import succeeded
        }

        calc def ArithmeticIntensity {
            in flops : Real;
            in bytes : Real;
            return : Real = flops / bytes;
        }
        calc def AchievedGflops {
            in flops : Real;
            in seconds : Real;
            return : Real = flops / seconds / 1000000000.0;
        }
        calc def SgemmFlops {
            in m : Natural;
            in k : Natural;
            in n : Natural;
            return : Real = 2.0 * m * k * n;
        }
    }

    // ------------------------------------------------------------------------------------
    // 3. Functional architecture (action usages are what get sequenced)
    // ------------------------------------------------------------------------------------
    package Functions {
        private import Domain::*;

        action def DiscoverBackends {
            out item backendCount : Natural;
        }
        action def PrepareExecution {
            in item problem : Problem;
        }
        action def RunTimed {
            in item problem : Problem;
            in attribute warmup : Natural = 3;
            in attribute iterations : Natural = 10;
            out item result : BenchResult;
        }
        action def VerifyAgainstOracle {
            in item result : BenchResult;
            out attribute verified : Boolean;
        }
        action def ReportJsonAndText {
            in item result : BenchResult;
        }

        action def RunBenchmarkSuite {
            in item problem : Problem;

            action discover : DiscoverBackends;
            action prepare : PrepareExecution;      // pipeline build + upload: never inside a timed region
            action run : RunTimed;
            action verify : VerifyAgainstOracle;
            action report : ReportJsonAndText;

            bind prepare.problem = problem;
            bind run.problem = problem;
            flow from run.result to verify.result;
            flow from run.result to report.result;

            first start;
            then discover;
            then prepare;
            then run;
            then verify;
            then report;
            then done;
        }
    }

    // ------------------------------------------------------------------------------------
    // 4. Logical architecture
    // ------------------------------------------------------------------------------------
    package LogicalArchitecture {
        private import Domain::*;
        private import Functions::*;

        port def ComputePort {
            in item problem : Problem;
            out item result : BenchResult;
        }
        // Host side uses the conjugate, so the two ends have opposite directions.
        interface def HostBackendInterface {
            end host : ~ComputePort;
            end backend : ComputePort;
        }

        part def MemoryManager {
            doc /* Unified-memory GPUs use host-visible device memory directly; discrete GPUs use
                   device-local memory plus staging. AHardwareBuffer sharing is an optional,
                   experimental Vulkan path that reports success only if the import worked. */
        }

        abstract part def ComputeBackend {
            port compute : ComputePort;
            attribute genuine : Boolean = true;   // false for stand-ins (sycl-shim)
            attribute available : Boolean;        // probed at run time
        }
        part def CpuBackend : ComputeBackend {
            doc /* Persistent thread pool, dynamic scheduling, optional core pinning. */
        }
        part def VulkanBackend : ComputeBackend {
            doc /* SPIR-V from shared GLSL (glslc at build time), GPU timestamps, two-pass reduction. */
        }
        part def OpenClBackend : ComputeBackend {
            doc /* Runtime dlopen of the vendor ICD; event profiling. Not an NDK API. */
        }
        part def GlesBackend : ComputeBackend {
            doc /* OpenGL ES 3.1 compute over a headless EGL pbuffer context. */
        }
        part def NnapiBackend : ComputeBackend {
            doc /* FULLY_CONNECTED via NNAPI per device; deprecated since Android 15. */
        }
        part def SyclShimBackend : ComputeBackend {
            doc /* SYCL-shaped API on the CPU thread pool. NOT SYCL: genuine = false. */
        }

        part def HostRuntime {
            port cpuPort : ~ComputePort;
            port vulkanPort : ~ComputePort;
            port openClPort : ~ComputePort;
            port glesPort : ~ComputePort;
            port nnapiPort : ~ComputePort;
            port syclPort : ~ComputePort;
            part memory : MemoryManager;
            perform action suite : RunBenchmarkSuite;
        }

        part def AndroidHPCSystem {
            attribute minSdk : Natural = 26;
            attribute allResultsVerified : Boolean;       // measured, set by a run
            attribute attemptedBackendCount : Natural;    // measured, set by a run

            part host : HostRuntime;
            part cpu : CpuBackend;
            part vk : VulkanBackend [0..*];
            part cl : OpenClBackend [0..*];
            part gl : GlesBackend [0..1];
            part nn : NnapiBackend [0..*];
            part sycl : SyclShimBackend;

            interface cpuLink : HostBackendInterface connect host.cpuPort to cpu.compute;
            interface vkLink : HostBackendInterface connect host.vulkanPort to vk.compute;
            interface clLink : HostBackendInterface connect host.openClPort to cl.compute;
            interface glLink : HostBackendInterface connect host.glesPort to gl.compute;
            interface nnLink : HostBackendInterface connect host.nnapiPort to nn.compute;
            interface syclLink : HostBackendInterface connect host.syclPort to sycl.compute;
        }
    }

    // ------------------------------------------------------------------------------------
    // 5. Physical / deployment view and allocation (allocation relates usages, not definitions)
    // ------------------------------------------------------------------------------------
    package Deployment {
        private import LogicalArchitecture::*;

        part def MulticoreCpu {
            attribute coreCount : Natural;
            attribute architecture : String;
        }
        part def Gpu {
            attribute vendor : String;
        }
        part def Npu {
            attribute vendor : String;
            attribute exposedVia : String [*];
        }
        part def Memory {
            attribute unified : Boolean;
        }
        part def SystemOnChip {
            part cpuCluster : MulticoreCpu;
            part gpu : Gpu;
            part npu : Npu [0..1];
            part memory : Memory;
        }
        part def AndroidRuntime {
            attribute apiLevel : Natural;
        }
        part def ApplicationProcess {
            part hpc : AndroidHPCSystem;
        }

        part def AndroidDevice {
            part soc : SystemOnChip;
            part os : AndroidRuntime;
            part app : ApplicationProcess;

            allocate app.hpc.cpu to soc.cpuCluster;
            allocate app.hpc.vk to soc.gpu;
            allocate app.hpc.cl to soc.gpu;
            allocate app.hpc.gl to soc.gpu;
            allocate app.hpc.nn to soc.npu;
        }
    }

    // ------------------------------------------------------------------------------------
    // 6. Requirements (each evaluated against attributes of the system or a run)
    // ------------------------------------------------------------------------------------
    package Requirements {
        private import LogicalArchitecture::*;

        requirement def <'R-PORT'> Portability {
            doc /* Runs on Android 8.0 (API 26) and later with no vendor SDK linked at build time:
                   Vulkan, OpenCL and NNAPI are opened with dlopen and degrade to "unavailable".
                   Verify: apkanalyzer / llvm-readelf -d shows no libvulkan, libOpenCL or
                   libneuralnetworks in DT_NEEDED; install on devices from >= 3 SoC vendors. */
            subject system : AndroidHPCSystem;
            require constraint { system.minSdk <= 26 }
        }
        requirement def <'R-MULTIAPI'> MultiApiBackends {
            doc /* One Backend interface fronts CPU, Vulkan, OpenCL, OpenGL ES 3.1 compute, NNAPI. */
            subject system : AndroidHPCSystem;
            require constraint { system.attemptedBackendCount >= 1 }
        }
        requirement def <'R-SPIRV'> SpirvSharedKernels {
            doc /* The Vulkan path consumes SPIR-V produced at build time (glslc) from the same GLSL
                   bodies the GLES path uses. Verify: ctest spirv_validates (spirv-val). */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-HETERO'> HeterogeneousExecution {
            doc /* CPU and an accelerator run disjoint row ranges of one SGEMM concurrently and the
                   combined output is verified. Verify: hpc_cli reports the "Heterogeneous sgemm"
                   line with verified yes. */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-BENCH'> BenchmarkSuite {
            doc /* Five kernels (sgemm, conv3x3, reduce, triad, dense), warm-up, repeated runs,
                   min/median/mean/p95/stddev, device-side timing when the API offers it, JSON and
                   text reports. Energy is NOT measured. */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-CORRECT'> VerifiedResults {
            doc /* Every reported result matches an fp64 CPU oracle within a per-kernel tolerance;
                   the verifier is itself tested with a corrupt-output negative control. */
            subject system : AndroidHPCSystem;
            require constraint { system.allResultsVerified }
        }
        requirement def <'R-HONEST'> HonestReporting {
            doc /* Stand-ins are labelled (genuine = false). Unavailable backends keep a reason.
                   Zero-copy is reported only when the import succeeded. NPU claims require a device
                   that NNAPI reports as an accelerator, not "nnapi-reference". */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-MEM'> MemoryModels {
            doc /* Unified (direct host-visible) and discrete (staged) GPU memory are both handled;
                   AHardwareBuffer import is an optional experimental path with copy fallback. */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-OBS'> Observability {
            doc /* Device discovery with capability probes (memory model, subgroup size, fp64,
                   workgroup limits, timer support, CPU topology and ISA). */
            subject system : AndroidHPCSystem;
        }
        requirement def <'R-EXT'> Extensibility {
            doc /* A new backend implements Backend/Execution; a new kernel adds a Problem case,
                   an oracle and per-backend kernels. No change to the harness. */
            subject system : AndroidHPCSystem;
        }
    }

    // ------------------------------------------------------------------------------------
    // 7. Concrete usages, satisfaction and verification
    // ------------------------------------------------------------------------------------
    package Verification {
        private import LogicalArchitecture::*;
        private import Requirements::*;

        part mvp : AndroidHPCSystem;

        requirement portability : Portability;
        requirement multiApi : MultiApiBackends;
        requirement spirv : SpirvSharedKernels;
        requirement hetero : HeterogeneousExecution;
        requirement bench : BenchmarkSuite;
        requirement correct : VerifiedResults;
        requirement honest : HonestReporting;
        requirement memory : MemoryModels;
        requirement observability : Observability;
        requirement extensibility : Extensibility;

        satisfy portability by mvp;
        satisfy multiApi by mvp;
        satisfy spirv by mvp;
        satisfy hetero by mvp;
        satisfy bench by mvp;
        satisfy correct by mvp;
        satisfy honest by mvp;
        satisfy memory by mvp;
        satisfy observability by mvp;
        satisfy extensibility by mvp;

        verification def HostBackendsMatchOracle {
            doc /* ctest backends_match_fp64_reference on a machine with real GPUs. */
            subject sut : AndroidHPCSystem;
            objective {
                verify correct;
            }
            return verdict : VerdictKind;
        }
        verification def SpirvIsValid {
            doc /* ctest spirv_validates. */
            subject sut : AndroidHPCSystem;
            objective {
                verify spirv;
            }
            return verdict : VerdictKind;
        }
        verification def HeterogeneousSplitVerified {
            doc /* hpc_cli --kernel sgemm; require the heterogeneous line with verified yes. */
            subject sut : AndroidHPCSystem;
            objective {
                verify hetero;
            }
            return verdict : VerdictKind;
        }
        verification def DeviceMatrix {
            doc /* Run the APK on >= 3 devices from different SoC vendors and archive the JSON. */
            subject sut : AndroidHPCSystem;
            objective {
                verify portability;
                verify honest;
            }
            return verdict : VerdictKind;
        }
    }

    // ------------------------------------------------------------------------------------
    // 8. Views (expose belongs to view usages, never to view definitions)
    // ------------------------------------------------------------------------------------
    package Views {
        view def ArchitectureView;
        view architectureView : ArchitectureView {
            expose AndroidHPC::LogicalArchitecture::**;
        }
        view def RequirementsView;
        view requirementsView : RequirementsView {
            expose AndroidHPC::Requirements::**;
        }
    }
}
