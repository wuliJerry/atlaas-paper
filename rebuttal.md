We thank the reviewers for their detailed feedback. Below we address the recurring concerns once (§ Common), then provide reviewer-specific responses. All requested changes will be incorporated in the camera-ready.

## Common Concerns

**C1: *Generality beyond Gemmini and VTA — reconfigurable, sparse, and irregular designs.***

We applied the unmodified pipeline to three more accelerators of different classes. **FEATHER (ISCA'24)** addresses Reviewer A's reconfigurable-interconnect concern: a reconfigurable spatial array with a BIRRD Benes-network reducer, architecturally opposite to a systolic mesh. The lift recovers FEATHER's hand-designed ISA 2.0 (the authors' `minisa`, ISPASS'26) almost instruction-for-instruction, recovering *fused* micro-ops (`SetIVN_OVNLayout`, `Load_ExecuteStreaming_addr`, `ExecuteMapping` subsuming the WVN load) that the hand-written ISA splits. The extracted spec drives an ACT-generated backend that compiles FEATHER's 50-workload evaluation suite end-to-end; holding tiling fixed to `minisa`'s own `choose_tile_sizes`, the lifted ISA needs geomean 1.27× (up to 1.43×) fewer instructions, with the largest savings on the flagship FHE-bootstrapping and ZKP kernels and parity on large compute-bound matmuls, never a regression. We report *instruction count* (`minisa`'s headline metric) rather than cycles because no independent FEATHER cycle simulator exists; cycle numbers would come from `minisa`'s own analytical model. Gemmini's functional-Spike `rdcycle` tracks instruction count closely, so the metric is consistent across both.

**NVDLA**'s sparse weight-decompression unit and **SPAGHETTI (HPCA'21)** (sparse–sparse FP32 SpGEMM) address Reviewer A's sparse-accelerator concern. The pipeline captures both end-to-end: NVDLA's popcount-indexed weight scatter lifts with base-lane Z3-proven semantics; SPAGHETTI's IEEE-754 recoded-FP multiply–add units (`OuterDot`, `CooSCALFU`, `CooSCALNode`, `Adder`) all reach the dense `clamp(dot(A,B)+C)` compute template. FP arithmetic and CSR/COO value streams capture as **Tensor**; the sparse coordinate machinery (merge-sort, shape/shift transforms, virtual-channel allocation, descriptor control) captures as **Control** (C2). FEATHER, NVDLA, and SPAGHETTI will appear as camera-ready case studies. Generality comes from structural cues — sign-extension chains, MAC fan-in, clamp idioms, accumulator def–use chains — that are RTL-synthesis patterns rather than design-class-specific. We do not claim coverage of *synthesis-optimized* hardware (retimed, register-fused, or technology-mapped datapaths where these idioms can be obscured); the camera-ready will state this limitation explicitly.

**C2: *Per-instruction coverage of captured semantics.***

Our paper introduced *opaque* as the Stage-3 fallback for instructions that do not match a tensor template, and Reviewer C asks about *partially lifted instructions* in the same vein. This template-fit framing conflates two distinct cases: an instruction with no tensor semantics versus one whose tensor semantics are present but match no single dense template. We separate them: every instruction is captured (Failed = 0 for all five accelerators), and we report each captured instruction's role — **Tensor** (produces or moves numeric tensor values, e.g. MAC, dot, reduce, DMA of value arrays) or **Control** (manages config / addressing / coordinate / sequencing state, no numeric output). Across 38 instructions on five accelerators (Gemmini, VTA, FEATHER (C1), NVDLA, SPAGHETTI):

| Accelerator | Tensor | Control | Failed | Instructions |
|---|:---:|:---:|:---:|:---:|
| Gemmini   | 7 | 4 | 0 | 11 |
| VTA       | 3 | 1 | 0 | 4  |
| FEATHER   | 4 | 4 | 0 | 8  |
| NVDLA (`WL_dec`) | 1 | 1 | 0 | 2  |
| SPAGHETTI | 6 | 7 | 0 | 13 |
| **Total** | **21** | **17** | **0** | **38** |

Cases a template-fit grading would mark partial — VTA's adder-tree GEMM, FEATHER's BIRRD reduce, NVDLA's popcount-indexed weight scatter — all classify as **Tensor**: they carry value semantics, just not as a single dense template. We thank Reviewers A and C for raising these questions, which prompted us to rethink the framing; the camera-ready will adopt the role taxonomy throughout. We also acknowledge Reviewer A's Weakness 2: the headline 92.9% reduction is per-PE, and the overall 24.8% (Gemmini) / 41.2% (VTA) figures already in the paper reflect the control-heavy reality. The camera-ready will lead with the overall numbers and present 92.9% as a per-module compute-core result rather than the headline.

**C3: *Performance framing — the role of discovered features.***

Reviewers A and B note that the paper's 1.014× geomean on standard MLP/ResNet/MobileNet workloads is essentially parity; the extracted spec's value is automation and correctness, not generic speedup. To make the discovered-features story concrete (Reviewer A's Strength 4), we ran a **feature-ablation** experiment: two ACT-generated backends differing only by the StoreController **pooling instruction** — a feature discovered by TensorLift but absent from the hand-written reference. Fed an identical conv → 2×2-maxpool graph, the pooling-aware spec compiles to a single fused `LOOP_CONV_WS` with pooling; the pooling-unaware spec is forced into hardware convolution plus a host-CPU max-pool pass:

| conv → pool shape | w/o pool (cyc) | w/ pool (cyc) | speedup |
|---|---:|---:|---:|
| 16×16×32 |  62,224 | 22,224 |  2.8× |
| 24×24×48 | 238,942 | 32,472 |  7.4× |
| 32×32×32 | 260,743 | 22,310 | 11.7× |

Output is bit-identical in every case. ACT performs no implicit software fallback — the CPU pass appears only because the un-fused pool must be covered by *some* instruction in the ablated spec. The main end-to-end result remains parity on standard workloads, while discovered features translate to substantial wins when workloads specifically exercise them.

## Response to Reviewer A

**R-A.1 *End-to-end lifting time on the largest module and scaling with design size / unroll bound.***

At Gemmini DIM=16, wall-clock per module is PE 1.3 s, LoadController 6.5 s, StoreController 7.2 s, and **ExecuteController ≈5250 s** (flatten 48 s; extract 4540 s; lift 619 s; assemble 43 s; Z3 25 ms). Sweeping DIM ∈ {4, 8, 16, 32} on the ExecuteController (all times CPU-s, sequential-equivalent; DIM=32 ran sharded across 32 cores):

| DIM | PEs | Extract (CPU-s) | Lift (CPU-s) | Z3 (ms) | TAIDL lines |
|---:|---:|---:|---:|---:|---:|
|  4 |   16 |      29 |   4.5 | 25.4 | 40 |
|  8 |   64 |     231 |    42 | 26.2 | 40 |
| 16 |  256 |   4,540 |   619 | 25.4 | 41 |
| 32 | 1024 | 201,062 | 7,820 | 25.4 | 41 |

Per-PE extraction grows super-linearly (1.8 → 3.6 → 17.7 → 196 s/PE) because the dependency cone deepens with unroll depth 2·DIM+4. Two invariants stay flat: the assembled TAIDL spec stays ≈41 lines at every DIM (`TILE_DIM` auto-inferred), and the Z3 MAC proof stays at ≈25 ms over all 2⁶⁰ inputs — one per-PE proof certifies all DIM² PEs by systolic homogeneity, so verification is O(1) in DIM. Extraction and lift are embarrassingly parallel: sharding DIM=32's 1024 PEs across 32 cores cuts wall-clock to 1.9 h.

**R-A.2 *Fraction of instructions that lift to full tensor semantics vs. opaque, with failure causes.***

See (C2). For Gemmini specifically, seven instructions are **Tensor** (`compute_preloaded`, `compute_accumulated`, `preload` operand-staging, `mvin` / `mvin2` / `mvin3`, `mvout`); four are **Control** (`config_ex` / `_ld` / `_st` / `_norm`); none failed.

**R-A.3 *Correctness guarantees for Stage-3 TAIDL assembly.***

Stage-3 assembly is not independently SMT-proven end-to-end; the formal proofs establish equivalence between the Stage-2 lifted MLIR and the scalar RTL-extracted model (autoGenILA's per-register LLVM IR), via Z3, for the core compute and DMA primitives (PE MAC, weight-stationary dataflow mux, DMA copy, and the VTA datapath); the remaining ops (pooling, im2col-conv, DMA `mvin` / `mvout`) are validated against Spike golden data. Stage-3's correctness risk is bounded by three mechanisms.

(i) **HLO templating** is a fixed, accelerator-agnostic mapping from a closed set of Stage-2 annotations to HLO: `tensor_op = dot_product | mac` → `convert + dot + add` (the `add` is present iff the lifter recovered an accumulator read, else a `copy`); `scalar_op = mul` → `multiply` (with optional `clamp` / quantize when the lifter detected requantization); `scalar_op = add` → reduction `add`. The templates contain no per-instruction or per-accelerator logic — dispatch is purely on the recovered annotation/role.

(ii) **CISC macro composition** (`loop_ws`, `loop_conv_ws`) reuses the per-primitive MAC semantics already proven Z3-equivalent to RTL in Stage-2; the macro body is the same `dot + add` primitive wrapped only in shape plumbing (`reshape` / `bitcast` / `convert`), iterating over a loop nest whose bound *registers* are identified from the controller's RTL (the bound values are runtime configuration fields).

(iii) **FSM ordering** (e.g., `compute_preloaded` may only fire after `preload`) is recovered from the RTL control register: the active-state guards come from explicit state-comparison logic in the extracted RTL, and every ordering edge corresponds to an instruction that demonstrably writes that control register; the complementary idle-state guard is inferred under a binary-FSM assumption.

We also acknowledge that the paper's abstract states correctness is validated through Z3 SMT throughout, which overclaims relative to the scope above; the camera-ready abstract will align with the Z3 + Spike golden-data split. The camera-ready will also add a worked example tying each of (i)–(iii) to the corresponding RTL evidence.

## Response to Reviewer B

**R-B.1 *Handling RTL with highly irregular or dynamic control flow.***

See (C1) for FEATHER's BIRRD reconfigurable reduction network, and SPAGHETTI's NoC routing and coordinate merge-sort for sparse irregular control. The pipeline never attempts to mis-lift: instructions that manage control, addressing, or coordinate state classify as **Control** in (C2), where their bit-level semantics are preserved verbatim.

**R-B.2 *Compilation-time overhead compared to traditional verification flows.***

TensorLift is a **one-time spec-extraction cost**, not a per-compile overhead. The dominant cost is autoGenILA extraction (4540 s on Gemmini's ExecuteController at DIM=16), paid once per accelerator; downstream lift (619 s), assembly (43 s), and Z3 (≈25 ms) reuse the extracted MLIR for every subsequent compile. This replaces manual TAIDL authoring — an expert-weeks effort that motivated this work and that traditional formal-verification flows do not produce; verification cost is flat in DIM (one per-PE MAC proof certifies the whole mesh).

## Response to Reviewer C

**R-C.1 *Why two accelerators suffice; results on more diverse, control-heavy, or synthesis-optimized designs.***

See (C1). The five accelerators evaluated — Gemmini, VTA, FEATHER, NVDLA, SPAGHETTI — span systolic, adder-tree, reconfigurable spatial, and sparse-decode / SpGEMM classes. Synthesis-optimized hardware remains a limitation, which the camera-ready will state explicitly.

**R-C.2 *More details on failure cases and partially lifted instructions.***

See (C2). Cases a template-fit grading would mark partial — VTA's adder-tree GEMM, FEATHER's BIRRD reduce, NVDLA's sparse weight scatter — classify as Tensor: value semantics captured, just not in a single dense template. (C3) covers the discovered-features practical value.
