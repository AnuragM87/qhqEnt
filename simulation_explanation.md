# QHQ Waveplate Polarization Correction Simulation — Detailed Explanation

## 1. Introduction and Motivation

In quantum communication and quantum information experiments, **entangled photon pairs** are a cornerstone resource. A pair of photons in a maximally entangled Bell state — such as |Φ⁺⟩ = (|HH⟩ + |VV⟩)/√2 — exhibits perfect quantum correlations that enable protocols like quantum key distribution (QKD), quantum teleportation, and Bell inequality tests.

However, real-world optical channels (fibers, free-space links) introduce **unitary polarization rotations** and **depolarizing noise** that degrade the entangled state. The measured two-photon density matrix ρ_exp, reconstructed via quantum state tomography, typically shows reduced fidelity with respect to the intended Bell state.

This simulation addresses the problem of **recovering Bell-state fidelity** by computationally determining the optimal orientation angles of passive wave-plate compensators — specifically, two sets of **Quarter-wave plate – Half-wave plate – Quarter-wave plate (QHQ)** assemblies, one in each arm of the entangled photon pair — and then applying the corresponding unitary correction to the experimentally measured density matrix.

---

## 2. Physical Setup: The QHQ Compensator

### 2.1 Wave-plate Jones Matrices

Each waveplate is characterized by a **Jones matrix** that describes its action on the polarization state of a single photon in the {|H⟩, |V⟩} basis.

#### Quarter-Wave Plate (QWP)

A QWP introduces a π/2 phase retardation between its fast and slow axes. When its fast axis is oriented at angle θ from the horizontal, the Jones matrix is:

```
         ⎡ cos²θ + i·sin²θ      (1 − i)·sinθ·cosθ ⎤
U_Q(θ) = ⎢                                          ⎥
         ⎣ (1 − i)·sinθ·cosθ    sin²θ + i·cos²θ    ⎦
```

where θ is specified in degrees and internally converted to radians.

#### Half-Wave Plate (HWP)

A HWP introduces a π phase retardation. Its Jones matrix at orientation θ is:

```
         ⎡  cos2θ    sin2θ ⎤
U_H(θ) = ⎢                 ⎥
         ⎣  sin2θ   −cos2θ ⎦
```

### 2.2 QHQ Assembly per Arm

Each arm of the entangled pair passes through a cascade of three waveplates in the order **QWP → HWP → QWP**. The composite single-arm unitary is:

```
U_arm(α, β, γ) = U_Q(α) · U_H(β) · U_Q(γ)
```

where α, β, γ are the three waveplate angles (in degrees) for that arm.

The QHQ configuration is deliberately chosen because it can implement **any arbitrary SU(2) rotation** of the polarization. This universality follows from the Euler-angle decomposition of rotations: two QWPs provide phase shifts about different axes, and the HWP provides a π-rotation, together spanning the full SU(2) group.

### 2.3 Two-Photon Correction Operator

Since the entangled state involves two photons — one in each arm — the total correction operator is the **Kronecker (tensor) product** of the two single-arm unitaries:

```
U_total = U_arm1(α₁, β₁, γ₁) ⊗ U_arm2(α₂, β₂, γ₂)
```

This is a 4×4 unitary matrix acting on the two-qubit Hilbert space spanned by {|HH⟩, |HV⟩, |VH⟩, |VV⟩}.

The corrected density matrix is then:

```
ρ_corrected = U_total · ρ_in · U_total†
```

where U_total† denotes the conjugate transpose.

---

## 3. Target Bell States

The simulation supports three maximally entangled Bell states as correction targets:

| State | Ket Representation | Density Matrix ρ_target |
|-------|-------------------|------------------------|
| Φ⁺ | (|HH⟩ + |VV⟩) / √2 | ψ·ψ†, with ψ = [1, 0, 0, 1]ᵀ/√2 |
| Φ⁻ | (|HH⟩ − |VV⟩) / √2 | ψ·ψ†, with ψ = [1, 0, 0, −1]ᵀ/√2 |
| Ψ⁻ | (|HV⟩ − |VH⟩) / √2 | ψ·ψ†, with ψ = [0, 1, −1, 0]ᵀ/√2 |

Each target state is constructed as a rank-1 projector ρ_target = |ψ⟩⟨ψ|, which is a 4×4 Hermitian, positive semi-definite matrix with unit trace.

---

## 4. Input: Quantum State Tomography Data

### 4.1 Loading the Density Matrix

The experimental density matrix is obtained via **quantum state tomography** — a set of projective measurements in multiple polarization bases (HH, HV, VH, VV, DA, etc.) from which the full 4×4 complex matrix ρ_exp is reconstructed.

The tomography data is stored as a plain-text file containing a 4×4 complex matrix with space-separated entries. For example:

```
0.42        0          0          (0.36+0.12j)
0           0.08       (0.02-0.06j)  0
0           (0.02+0.06j)  0.08    0
(0.36-0.12j) 0          0          0.42
```

Upon loading, the matrix is:
1. **Hermitian-symmetrized**: ρ ← (ρ + ρ†) / 2, to correct for any numerical asymmetry.
2. **Trace-normalized**: ρ ← ρ / Tr(ρ), to ensure the density matrix represents a valid quantum state.

---

## 5. Preprocessing: Noise Compensation

Before the waveplate optimization, the raw tomography data can optionally be cleaned using one of two noise-compensation methods.

### 5.1 Eigenvalue Filtering

This method projects the density matrix onto the subspace of its dominant eigenstates, discarding small eigenvalues associated with noise.

**Algorithm:**
1. Eigendecompose: ρ = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|, where λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄.
2. Keep only the top *k* eigenvalues (default k = 1 for near-pure states):
   ```
   ρ_clean = Σᵢ₌₁ᵏ λᵢ |vᵢ⟩⟨vᵢ|
   ```
3. Re-normalize: ρ_clean ← ρ_clean / Tr(ρ_clean).

**Physical interpretation:** For an ideally pure entangled state, only one eigenvalue should be non-zero (λ₁ ≈ 1). In practice, noise spreads population across multiple eigenstates. By retaining only the dominant eigenstate(s), we effectively project out the noise contributions and increase the purity Tr(ρ²) of the reconstructed state.

### 5.2 Depolarization Compensation

This method models the experimental noise as **isotropic depolarization** and analytically inverts it.

**Noise model:**
```
ρ_noisy = (1 − p) · ρ_ideal + p · I/4
```

where p ∈ [0, 1] is the depolarization fraction and I/4 is the maximally mixed two-qubit state.

**Inversion:**
```
ρ_clean = (ρ_noisy − p · I/4) / (1 − p)
```

If p is not provided, it is estimated from the smallest eigenvalue of ρ_noisy:
```
p_est = λ_min × d     (d = 4 for two qubits)
```

This follows from the fact that the smallest eigenvalue of a depolarized state is approximately p/d.

After inversion, the result is:
1. Hermitian-symmetrized.
2. Negative eigenvalues clipped to zero (to maintain physicality).
3. Re-normalized to unit trace.

---

## 6. Fidelity Metric: Uhlmann Fidelity

The quality of the correction is quantified by the **Uhlmann fidelity** between the corrected state ρ and the target state σ:

```
F(ρ, σ) = ( Tr √(√ρ · σ · √ρ) )²
```

For computational stability, the implementation uses an eigenvalue-based formulation:

```
F(ρ, σ) = ( Σᵢ √λᵢ )²
```

where {λᵢ} are the eigenvalues of the matrix product ρ · σ, with negative eigenvalues (arising from numerical noise) clipped to zero.

**Properties of Uhlmann fidelity:**
- F = 1 if and only if ρ = σ (identical states).
- F = 0 for orthogonal states.
- 0 ≤ F ≤ 1 for all physical states.
- It reduces to |⟨ψ|φ⟩|² when both states are pure.
- It is unitarily invariant: F(UρU†, UσU†) = F(ρ, σ).

Before computing fidelity, both matrices are Hermitian-symmetrized, trace-normalized, and eigenvalue-cleaned (negative eigenvalues set to zero) for numerical robustness.

---

## 7. Cost Function

The optimization problem is cast as a **minimization** problem. The cost function to be minimized is:

```
C(α₁, β₁, γ₁, α₂, β₂, γ₂) = 1 − F(ρ_target, U_total · ρ_in · U_total†)
```

where the six angles (three per arm) are the free parameters. Each angle is bounded to the range [0°, 180°] — covering the full physical range of waveplate orientations, since wave-plate Jones matrices are periodic in 180°.

The cost landscape is **non-convex** with multiple local minima, necessitating global or multi-start optimization strategies.

---

## 8. Optimization Algorithms

The simulation provides seven optimization strategies, ranging from simple local methods to robust global search algorithms.

### 8.1 L-BFGS-B Multistart

- **Method:** Gradient-based quasi-Newton optimizer (L-BFGS-B) with box constraints [0, 180]° per angle.
- **Strategy:** Launch *n* independent runs (default n = 8) from random initial points; retain the best result.
- **Strengths:** Fast convergence near a minimum.
- **Weaknesses:** Can converge to local minima; performance depends on starting points.

### 8.2 Powell Multistart

- **Method:** Powell's conjugate-direction method — **derivative-free**, which is advantageous for noisy or non-smooth fidelity landscapes.
- **Strategy:** *n* random restarts (default n = 10), each running up to 2000 iterations with tight tolerances (xtol = ftol = 10⁻⁶).
- **Strengths:** Robust for functions where gradient computation is unreliable.

### 8.3 Differential Evolution (DE)

- **Method:** Population-based stochastic global optimizer inspired by biological evolution.
- **Key parameters:**
  - Strategy: `best1bin` — mutant vectors formed from the best individual.
  - Population size: 15.
  - Mutation factor: (0.5, 1.0) — dithered for diversity.
  - Crossover: 0.7.
  - Max iterations: 300.
  - Polish: enabled (local L-BFGS-B refinement after convergence).
- **Strengths:** Effective at escaping local minima; requires no gradient.

### 8.4 DE Robust (Recommended)

- **Method:** Production-grade variant with multiple internal runs, each using a different random seed for reproducibility.
- **Enhancements over basic DE:**
  - Strategy: `best1exp` (exponential crossover for deeper parameter correlation).
  - Larger population: 25.
  - Higher mutation range: (0.5, 1.5).
  - Higher crossover: 0.9.
  - 500 max iterations per run.
  - **Custom Polish:** Each DE run is followed by a dedicated Powell refinement (3000 iterations, xtol = ftol = 10⁻⁸) for sub-degree angular precision.
  - Early termination when target fidelity is reached.
- **Total runs:** 5 (default), each with seed = 42 + i for deterministic variation.
- **This is the recommended optimizer for laboratory use.**

### 8.5 DE Multi-run

- **Method:** Simple wrapper that runs the basic DE optimizer multiple times (default 3) and returns the best result.
- **Use case:** Quick global search when DE Robust is too slow.

### 8.6 Hybrid (DE → Powell)

- **Method:** Two-phase approach:
  1. **Global phase:** Differential Evolution to locate the basin of the global minimum.
  2. **Local phase:** Powell refinement seeded at the DE solution for fine-tuning.
- **Strengths:** Combines global exploration with local precision.

### 8.7 COBYLA

- **Method:** Constrained Optimization BY Linear Approximations — a derivative-free trust-region method.
- **Parameters:**
  - Box constraints implemented as inequality constraints: 0 ≤ xᵢ ≤ 180.
  - Initial trust-region radius: 30°.
  - Max iterations: 3000.
  - Constraint tolerance: 10⁻⁶.
- **Strengths:** Handles constraints natively; good for noisy landscapes.

### 8.8 Outer Iteration Loop

All optimizers are invoked within an outer loop (configurable number of iterations, default 5–10). In each iteration:
1. The optimizer is called and returns candidate angles and fidelity.
2. If the new fidelity exceeds the previous best, the solution is updated.
3. If the best fidelity reaches the target threshold (default 0.98), the loop terminates early.

This multi-iteration approach further mitigates the risk of settling on suboptimal solutions and adds robustness beyond what the optimizer alone provides.

---

## 9. Correction and Output

Once the optimal angles {α₁*, β₁*, γ₁*, α₂*, β₂*, γ₂*} are determined:

1. **Construct the correction operator:**
   ```
   U_total = [ U_Q(α₁*)·U_H(β₁*)·U_Q(γ₁*) ] ⊗ [ U_Q(α₂*)·U_H(β₂*)·U_Q(γ₂*) ]
   ```

2. **Apply the correction:**
   ```
   ρ_corrected = U_total · ρ_in · U_total†
   ```

3. **Evaluate results:**
   - **Fidelity (raw):** F(ρ_target, ρ_raw) — baseline before any processing.
   - **Fidelity (preprocessed):** F(ρ_target, ρ_in) — after noise compensation.
   - **Fidelity (corrected):** F(ρ_target, ρ_corrected) — after waveplate correction.
   - **Purity:** Tr(ρ²) — quantifies how mixed/pure the state is (1.0 for a pure state, 0.25 for maximally mixed two-qubit state).

---

## 10. Visualization

The simulation produces several diagnostic plots:

### 10.1 State Population Bar Chart
- Displays the diagonal elements of the density matrix, representing the probabilities of measuring each computational basis state: |HH⟩, |HV⟩, |VH⟩, |VV⟩.
- Side-by-side comparison of input (raw) vs. corrected populations.
- For a perfect |Φ⁺⟩ state, the populations should be: P(HH) = P(VV) = 0.5, P(HV) = P(VH) = 0.

### 10.2 Density Matrix Heatmap
- Absolute-value heatmap of the corrected density matrix |ρ_corrected|.
- Uses the "inferno" colormap with cell annotations showing numerical magnitudes.
- Provides a visual fingerprint of the quantum state structure — for a Bell state, the corners (|HH⟩⟨HH| and |VV⟩⟨VV| for Φ±) should be bright, with significant off-diagonal coherences.

### 10.3 Target vs. Corrected Comparison
- Side-by-side display of the numerically formatted target and corrected density matrices, allowing direct element-wise comparison.

---

## 11. Software Architecture

The simulation is organized as a modular Python package with clear separation of concerns:

```
qhqEnt/
├── gui.py                  # Streamlit web interface
├── main.py                 # CLI entry point
├── tomography.txt          # Sample tomography data
├── src/
│   ├── io.py               # Density matrix file loading
│   ├── waveplates.py       # QWP and HWP Jones matrices
│   ├── qhqstates.py        # QHQ arm & total unitary construction
│   ├── states.py           # Bell state definitions
│   ├── fidelity.py         # Uhlmann fidelity computation
│   ├── cost.py             # Optimization cost function
│   ├── optimizer.py         # Seven optimization algorithms
│   └── noise_compensation.py  # Depolarization & eigenvalue filtering
```

### Module Dependency Flow

```
tomography.txt
     │
     ▼
  io.py (load & normalize)
     │
     ▼
noise_compensation.py (optional preprocessing)
     │
     ▼
optimizer.py ──► cost.py ──► qhqstates.py ──► waveplates.py
     │               │
     │               ▼
     │          fidelity.py
     │               ▲
     ▼               │
  states.py (target Bell state)
     │
     ▼
 ρ_corrected (output)
```

### Dependencies

- **NumPy** — linear algebra, matrix operations, eigenvalue decomposition.
- **SciPy** — optimization routines (`minimize`, `differential_evolution`).
- **Matplotlib** — plotting (population charts, density matrix heatmaps).
- **Streamlit** — interactive web-based GUI.

---

## 12. Complete Simulation Workflow

The end-to-end workflow of the simulation proceeds as follows:

1. **Data Acquisition:** An experimentally reconstructed 4×4 complex density matrix (from quantum state tomography) is loaded from a text file.

2. **Hermitianization & Normalization:** The loaded matrix is symmetrized to enforce Hermiticity and normalized to unit trace.

3. **Preprocessing (optional):**
   - *Eigenvalue filtering* — projects onto the dominant eigenstate(s), removing noise.
   - *Depolarization compensation* — analytically inverts the isotropic depolarizing channel.

4. **Target Selection:** The user selects a target Bell state (Φ⁺, Φ⁻, or Ψ⁻).

5. **Initial Fidelity Computation:** The Uhlmann fidelity between the raw state and the target state is computed as a baseline.

6. **Optimization:** One of seven algorithms searches the 6-dimensional parameter space of QHQ waveplate angles [α₁, β₁, γ₁, α₂, β₂, γ₂] ∈ [0°, 180°]⁶ to maximize the fidelity (equivalently, minimize the cost 1 − F).

7. **Correction:** The optimal unitary U_total is constructed from the best angles and applied as ρ_corrected = U·ρ·U†.

8. **Evaluation:** Fidelity and purity metrics are computed for the raw, preprocessed, and corrected states. The improvement (Δ fidelity) is reported.

9. **Visualization:** Population bar charts, density matrix heatmaps, and numerical matrix comparisons are generated.

---

## 13. Mathematical Summary

### Key Equations

| Quantity | Expression |
|----------|-----------|
| QWP Jones matrix | U_Q(θ) — see Section 2.1 |
| HWP Jones matrix | U_H(θ) — see Section 2.1 |
| Single-arm unitary | U_arm = U_Q(α) · U_H(β) · U_Q(γ) |
| Total two-photon unitary | U_total = U_arm1 ⊗ U_arm2 |
| Corrected state | ρ_corrected = U_total · ρ_in · U_total† |
| Uhlmann fidelity | F(ρ, σ) = (Σᵢ √λᵢ)², where λᵢ = eig(ρ·σ) |
| Cost function | C = 1 − F(ρ_target, ρ_corrected) |
| Depolarization model | ρ_noisy = (1−p)·ρ_ideal + p·I/4 |
| Eigenvalue filter | ρ_clean = Σᵢ₌₁ᵏ λᵢ|vᵢ⟩⟨vᵢ| / Tr(·) |
| Purity | P = Tr(ρ²), with 1/d ≤ P ≤ 1 |

### Parameter Space

- **Dimension:** 6 (three angles per arm).
- **Domain:** [0°, 180°]⁶ (periodic boundary — waveplate rotations repeat every 180°).
- **Landscape:** Non-convex, with multiple local minima requiring global optimization.

---

## 14. Significance

This simulation demonstrates a practical, software-in-the-loop approach to **compensating polarization errors in entangled photon sources**. The key contributions are:

1. **Generality:** The QHQ configuration can correct any unitary polarization distortion, making it applicable to arbitrary fiber and free-space channels.

2. **Computational calibration:** Instead of tedious manual waveplate alignment, the optimal angles are determined computationally from a single tomographic measurement.

3. **Robustness:** Multiple optimization strategies (including global search) ensure reliable convergence even in noisy, multi-modal cost landscapes.

4. **Noise-aware preprocessing:** Built-in depolarization compensation and eigenvalue filtering improve fidelity recovery for noisy experimental data.

5. **Practical output:** The optimized angles can be directly applied to physical waveplate mounts in a laboratory setting.
