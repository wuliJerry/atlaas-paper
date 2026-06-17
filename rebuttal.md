We thank the reviewers for their detailed feedback. Below we address the recurring concerns once (§ Common), then provide reviewer-specific responses. All requested changes will be incorporated in the camera-ready.

## Common Concerns

**C1: *Generality beyond Gemmini and VTA, including irregular control flow.***

In response to the reviewers' generality concern, we additionally applied the unmodified pipeline to **FEATHER (ISCA'24)** — a reconfigurable spatial array with a BIRRD Benes-network reducer, architecturally opposite to a systolic mesh. The lift recovers FEATHER's hand-designed ISA 2.0 (the authors' `minisa`) almost instruction-for-instruction and in fact recovers *fused* micro-ops (`SetIVN_OVNLayout`, `Load_ExecuteStreaming_addr`, `ExecuteMapping` subsuming the WVN load) that the hand-written ISA splits. The extracted spec drives a real ACT-generated backend that compiles FEATHER's entire 50-workload evaluation suite end-to-end. Holding tiling fixed to `minisa`'s own `choose_tile_sizes`, the lifted ISA needs geomean 1.27× (up to 1.43×) fewer instructions — largest savings on the flagship FHE-bootstrapping and ZKP kernels, parity on large compute-bound matmuls; the gap grows with array size (1.10× at 4×4 → 1.27× at 16×16), never a regression. We will add this as a third camera-ready case study, including the per-instruction coverage and workload-level instruction-count results. Generality comes from structural cues — sign-extension chains, MAC fan-in, clamp idioms, accumulator def–use chains — that are RTL-synthesis patterns rather than systolic-specific.

**C2: *Per-instruction coverage and classification of partially lifted cases.***

We instrumented every `tensorlift-opt` pass and the TAIDL assembler to record, for each of the 23 hardware instructions across Gemmini, VTA, and FEATHER (C1), which *Stage-3 assembly path* the lifted MLIR follows: a *full* tensor template (`compute` or `DMA`), a *partial* body in which correct data-path semantics are present in the lifted MLIR but no existing whole-tensor template matches, or an *opaque* control-only fallback. Thus this table measures template coverage, not whether Stage 2 successfully extracted the instruction semantics. Across the three accelerators:

| Accelerator         | Full | Partial | Opaque | Instructions |
|---------------------|:---:|:---:|:---:|:---:|
| Gemmini (HW instrs) | 6 | 1 | 4 | 11 |
| VTA                 | 0 | 3 | 1 | 4 |
| FEATHER (MINISA)    | 3 | 1 | 4 | 8 |

Every residual maps to one of six causes: FSM/config-register state machine; operand data-staging into PE registers; adder-tree MAC fan-in; opcode-mux decoder dispatch; DMA-command / address generation; and reconfigurable reduction-network routing. **Opaque is a correctness-preserving fallback by construction**: when no template matches, the assembler emits a control-only body rather than incorrect TAIDL, so the assembler avoids fabricating tensor semantics when a template does not apply. In particular, VTA's `TensorGemm` semantics align with the documented ISA, but its adder-tree fan-in does not match the current Gemmini-style `dot_product` template, so it is classified as *partial* in this table. The camera-ready will include the full table with the per-instruction cause column.

**C3: *Performance framing — the role of discovered features.***

We agree with Reviewers A and B that the paper's 1.014× geomean on standard MLP/ResNet/MobileNet workloads is essentially parity; the value of the extracted spec is automation and correctness, not a generic speedup. To make the discovered-features story concrete (Reviewer A's Strength 4), we ran a **feature-ablation** experiment: two ACT-generated backends differing only by the StoreController **pooling instruction** — a feature discovered by TensorLift but absent from the hand-written reference. This is not a claim of generic speedup; it isolates the benefit of exposing a real hardware feature to the compiler. Fed an identical conv → 2×2-maxpool graph, the pooling-aware spec compiles to a single fused `LOOP_CONV_WS` with pooling; the pooling-unaware spec is forced into hardware convolution plus a host-CPU max-pool pass:

| conv → pool shape | w/o pool (cyc) | w/ pool (cyc) | speedup |
|---|---:|---:|---:|
| 16×16×32 |  62,224 | 22,224 |  2.8× |
| 24×24×48 | 238,942 | 32,472 |  7.4× |
| 32×32×32 | 260,743 | 22,310 | 11.7× |

Output is bit-identical in every case. ACT performs no implicit software fallback — the CPU pass appears only because the un-fused pool must be covered by *some* instruction in the ablated spec, so modeling the discovered hardware feature is what drives the speedup. Thus the main end-to-end result remains parity on standard workloads, while discovered features translate to substantial wins when workloads specifically exercise them.

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

See (C2). For Gemmini specifically, the two systolic compute instructions (`compute_preloaded`, `compute_accumulated`) and the four DMA instructions (`mvin` / `mvin2` / `mvin3`, `mvout`) reach full templates; `preload` is operand staging (partial); the four `config_*` opcodes are FSM-only by construction (opaque).

**R-A.3 *Correctness guarantees for Stage-3 TAIDL assembly.***

Stage-3 assembly is not independently SMT-proven end-to-end; the formal proofs in the paper establish equivalence between Stage-2 lifted MLIR and the scalar RTL-extracted model. Stage-3's correctness risk is bounded by three mechanisms: (i) **HLO templating** is a fixed, audited mapping from a closed set of Stage-2 annotations to HLO operations (`dot_product` → `convert + dot + add`, with optional `clamp`; `pool` → `reduce(max)`; `im2col_matmul` → `reshape + dot + add`); the templates contain no instruction-specific logic. (ii) **CISC macro composition** (e.g., `loop_ws`, `loop_conv_ws`) reuses per-primitive semantics already proven Z3-equivalent to RTL in Stage-2; the macro body is the same primitive op iterated over loop bounds recovered from the controller's RTL. (iii) **FSM ordering** (e.g., `compute_preloaded` may only fire after `preload`) is *recovered* from RTL transition relations, not synthesized: each ordering edge corresponds to an actual control-state update in the RTL. The camera-ready will add a worked example tying each of (i)–(iii) to the corresponding RTL evidence.

## Response to Reviewer B

**R-B.1 *Handling RTL with highly irregular or dynamic control flow.***

See (C1) for FEATHER, whose BIRRD reconfigurable reduction network is the concrete instance. Conceptually, the contract is that the pipeline *never attempts to mis-lift*: when control flow does not match a tensor template the instruction falls into the partial or opaque category of (C2), where its bit-level semantics are preserved verbatim. The cause taxonomy is small (six causes) and predictable, and covers every residual we observed across the three accelerators.

**R-B.2 *Compilation-time overhead compared to traditional verification flows.***

TensorLift is a **one-time spec-extraction cost**, not a per-compile overhead. The dominant cost is autoGenILA extraction (4540 s on Gemmini's ExecuteController at DIM=16), paid once per accelerator; downstream lift (619 s), assembly (43 s), and Z3 (≈25 ms) reuse the extracted MLIR for every subsequent compile. This replaces manual TAIDL authoring — an expert-weeks effort that motivated this work and that traditional formal-verification flows do not produce. Verification cost is flat in DIM (a single per-PE MAC proof certifies the whole systolic mesh), so it does not scale with design size.

## Response to Reviewer C

**R-C.1 *Why two accelerators suffice; results on more diverse, control-heavy, or synthesis-optimized designs.***

See (C1). FEATHER is the concrete instance and is architecturally diverse from Gemmini: spatial array + reconfigurable BIRRD reducer, not a systolic mesh. The lift recovers its per-PE MAC, activation/requantize, and partial-sum store fully; the BIRRD switch's reduction is recovered (only its routing remains opaque); the four layout/addressing opcodes are FSM-only by construction. The pipeline runs on FEATHER without any accelerator-specific changes, providing direct evidence that the structural cues used for lifting generalize beyond systolic GEMM engines.

**R-C.2 *More details on failure cases and partially lifted instructions.***

See (C2) for the per-instruction taxonomy across all 23 instructions on Gemmini, VTA, and FEATHER, with every residual mapped to one of six causes. This taxonomy is a Stage-3 template-coverage classification: for example, VTA's GEMM semantics are extracted and spec-aligned, but remain *partial* because the adder-tree fan-in does not yet collapse to the existing `dot_product` template. Opaque is a conservative fallback, not a silent failure. The complement — the practical value of the *fully* lifted and *discovered* features — is in (C3).
