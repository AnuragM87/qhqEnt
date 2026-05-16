# QHQ Waveplate Polarization Correction Simulation — Detailed Report

---

## 1. Introduction and Motivation

In quantum communication experiments, **entangled photon pairs** in maximally entangled Bell states (e.g., |Φ⁺⟩ = (|HH⟩ + |VV⟩)/√2) are a fundamental resource for protocols such as quantum key distribution, quantum teleportation, and Bell inequality tests.

Real-world optical channels — fibers and free-space links — introduce **unitary polarization rotations** that degrade the entangled state. The experimentally reconstructed two-photon density matrix ρ_exp, obtained via quantum state tomography, typically shows reduced fidelity with respect to the intended Bell state.

This simulation addresses the problem of **recovering Bell-state fidelity** by computationally determining the optimal orientation angles of passive waveplate compensators — specifically, two sets of **Quarter-wave plate – Half-wave plate – Quarter-wave plate (QHQ)** assemblies, one per arm — and applying the corresponding unitary correction to the measured density matrix.

---

## 2. Physical Setup: The QHQ Compensator

### 2.1 Wave-plate Jones Matrices

Each waveplate acts on the polarization state of a single photon in the {|H⟩, |V⟩} basis via a **Jones matrix**.

#### Quarter-Wave Plate (QWP)

A QWP introduces a π/2 phase retardation between its fast and slow axes. At orientation angle θ:

```
         ⎡ cos²θ + i·sin²θ      (1 − i)·sinθ·cosθ ⎤
U_Q(θ) = ⎢                                          ⎥
         ⎣ (1 − i)·sinθ·cosθ    sin²θ + i·cos²θ    ⎦
```

In the implementation ([waveplates.py](file:///Users/anuragchaudhary/Documents/qhqEnt/src/waveplates.py)):

```python
def UQ(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    off = s * c * (1 - 1j)
    return np.array([
        [c*c + 1j*s*s, off],
        [off,          s*s + 1j*c*c]
    ], dtype=complex)
```

#### Half-Wave Plate (HWP)

A HWP introduces a π phase retardation. At orientation θ:

```
         ⎡  cos2θ    sin2θ ⎤
U_H(θ) = ⎢                 ⎥
         ⎣  sin2θ   −cos2θ ⎦
```

```python
def UH(theta_deg):
    th = np.deg2rad(theta_deg)
    c2, s2 = np.cos(2*th), np.sin(2*th)
    return np.array([[c2, s2], [s2, -c2]], dtype=complex)
```

### 2.2 QHQ Assembly per Arm

Each arm passes through three waveplates in the order **QWP → HWP → QWP**. The composite single-arm unitary:

```
U_arm(α, β, γ) = U_Q(α) · U_H(β) · U_Q(γ)
```

The QHQ configuration is chosen because it can implement **any arbitrary SU(2) rotation** of the polarization — this follows from the Euler-angle decomposition of rotations on the Bloch sphere.

### 2.3 Two-Photon Correction Operator

The total correction for the two-photon entangled state is the **Kronecker (tensor) product** of the two single-arm unitaries:

```
U_total = U_arm1(α₁, β₁, γ₁) ⊗ U_arm2(α₂, β₂, γ₂)
```

This is a 4×4 unitary matrix acting on the two-qubit Hilbert space {|HH⟩, |HV⟩, |VH⟩, |VV⟩}. The corrected density matrix:

```
ρ_corrected = U_total · ρ_in · U_total†
```

---

## 3. Target Bell States

The simulation supports all four maximally entangled Bell states as correction targets:

| State | Ket Representation | State Vector ψ |
|-------|-------------------|----------------|
| Φ⁺ | (|HH⟩ + |VV⟩) / √2 | [1, 0, 0, 1]ᵀ/√2 |
| Φ⁻ | (|HH⟩ − |VV⟩) / √2 | [1, 0, 0, −1]ᵀ/√2 |
| Ψ⁺ | (|HV⟩ + |VH⟩) / √2 | [0, 1, 1, 0]ᵀ/√2 |
| Ψ⁻ | (|HV⟩ − |VH⟩) / √2 | [0, 1, −1, 0]ᵀ/√2 |

Each target is constructed as a rank-1 projector ρ_target = |ψ⟩⟨ψ| — a 4×4 Hermitian, positive semi-definite matrix with unit trace.

---

## 4. Input: Quantum State Tomography Data

### 4.1 Loading the Density Matrix

The experimental density matrix is obtained via **quantum state tomography** and stored as a plain-text 4×4 complex matrix. Upon loading ([io.py](file:///Users/anuragchaudhary/Documents/qhqEnt/src/io.py)):

1. **Hermitian-symmetrized**: ρ ← (ρ + ρ†) / 2
2. **Trace-normalized**: ρ ← ρ / Tr(ρ)

Example tomography data used in this simulation:

```
(0.458+0j)      (-0.0198-0.0211j)  (0.0388+0.0022j)   (0.474+0.0231j)
(-0.0198+0.0211j) (0.00757+0j)     (-0.00388+0.00451j) (-0.0296+0.0327j)
(0.0388-0.0022j)  (-0.00388-0.00451j) (0.00546+0j)     (0.0491-0.000724j)
(0.474-0.0231j)   (-0.0296-0.0327j)   (0.0491+0.000724j) (0.528+0j)
```

This matrix is already approximately Hermitian with dominant diagonal elements P(HH) ≈ 0.458 and P(VV) ≈ 0.528, and a strong off-diagonal coherence ρ₁₄ ≈ 0.474, consistent with a noisy Φ⁺ state.

---

## 5. Fidelity Metric: Uhlmann Fidelity

The quality of correction is quantified by the **Uhlmann fidelity** between corrected state ρ and target σ:

```
F(ρ, σ) = ( Tr √(√ρ · σ · √ρ) )²
```

The implementation uses an **eigenvalue-based formulation** for numerical stability ([fidelity.py](file:///Users/anuragchaudhary/Documents/qhqEnt/src/fidelity.py)):

```
F(ρ, σ) = ( Σᵢ √λᵢ )²
```

where {λᵢ} are the eigenvalues of the product ρ·σ, with negative eigenvalues clipped to zero.

**Properties:**
- F = 1 ⟺ ρ = σ (identical states)
- F = 0 for orthogonal states
- Unitarily invariant: F(UρU†, UσU†) = F(ρ, σ)
- Reduces to |⟨ψ|φ⟩|² for pure states

Before computing fidelity, both matrices are Hermitian-symmetrized, trace-normalized, and eigenvalue-cleaned for numerical robustness.

---

## 6. Cost Function

The optimization is cast as **minimization**. The cost function ([cost.py](file:///Users/anuragchaudhary/Documents/qhqEnt/src/cost.py)):

```
C(α₁, β₁, γ₁, α₂, β₂, γ₂) = 1 − F(ρ_target, U_total · ρ_in · U_total†)
```

Six angles (three per arm) are free parameters, each bounded to [0°, 180°] — covering the full physical range since waveplate Jones matrices are 180°-periodic. The cost landscape is **non-convex** with multiple local minima.

---

## 7. Optimization Algorithms

Seven optimization strategies are provided in [optimizer.py](file:///Users/anuragchaudhary/Documents/qhqEnt/src/optimizer.py):

### 7.1 L-BFGS-B Multistart
- Gradient-based quasi-Newton optimizer with box constraints
- *n* = 8 random restarts, retain best result
- Fast convergence near a minimum; susceptible to local minima

### 7.2 Powell Multistart
- Derivative-free conjugate-direction method
- *n* = 10 restarts, 2000 iterations each, xtol = ftol = 10⁻⁶
- Robust for noisy or non-smooth landscapes

### 7.3 Differential Evolution (DE)
- Population-based stochastic global optimizer
- Strategy: `best1bin`, popsize = 15, mutation = (0.5, 1.0), crossover = 0.7
- 300 max iterations, L-BFGS-B polish enabled

### 7.4 DE Robust (Recommended)
- Production-grade variant with multiple seeded internal runs
- Strategy: `best1exp`, popsize = 25, mutation = (0.5, 1.5), crossover = 0.9
- 500 max iterations per run + **Powell polish** (3000 iter, tol = 10⁻⁸)
- 5 runs with deterministic seeds (42 + i), early termination at target fidelity
- **Recommended for laboratory calibration use**

### 7.5 DE Multi-run
- Simple wrapper: runs basic DE 3 times, returns best result
- Quick global search when DE Robust is too slow

### 7.6 Hybrid (DE → Powell)
- Two-phase: DE for global basin location → Powell for fine-tuning
- Combines global exploration with local precision

### 7.7 COBYLA
- Derivative-free trust-region method with native constraint handling
- Box constraints as inequalities, initial radius = 30°, 3000 max iterations

### 7.8 Outer Iteration Loop
All optimizers are invoked within a configurable outer loop (default 5–10 iterations). Each iteration: run optimizer → update best if improved → terminate early if target fidelity reached.

---

## 8. Empirical Optimizer Comparison

All seven optimizers were benchmarked under three preprocessing conditions — **raw tomography data** (no filtering), **eigenvalue-filtered data** (rank = 1), and **depolarization compensation** — targeting the Φ⁺ Bell state.

### 8.1 Initial Fidelity (Before Optimization)

| Preprocessing | Fidelity |
|---------------|----------|
| Raw Data (no filter) | **0.9679** |
| Eigenvalue Filter (rank = 1) | **0.9910** |
| Depolarization Compensation (p ≈ 0.000004) | **0.9679** |

### 8.2 Corrected Fidelity After Optimization

| Optimizer Algorithm | Raw Data (No Filter) | Eigenvalue Filter | Depolarization Comp. |
|---------------------|:--------------------:|:-----------------:|:--------------------:|
| Multistart L-BFGS-B (150 restarts) | 0.9605 | 0.9700 | 0.9337 |
| Powell Multistart | 0.9741 | 0.9973 | 0.9741 |
| Differential Evolution | 0.9741 | 0.9973 | 0.9737 |
| **DE Robust (recommended)** | **0.9741** | **0.9973** | **0.9741** |
| DE Multi-run | 0.9741 | 0.9973 | 0.9741 |
| Hybrid (DE→Powell) | 0.9741 | 0.9973 | 0.9741 |
| COBYLA | 0.9741 | 0.9973 | 0.9741 |

**Note on Depolarization Compensation:** The estimated noise fraction is extremely small (p ≈ 0.000004), meaning the experimental state has negligible isotropic depolarization. As a result, depolarization compensation has virtually **no effect** — the corrected fidelities match the raw data results (F ≈ 0.9741). The eigenvalue filter remains the most effective preprocessing method, boosting the achievable fidelity to 0.9973.

### 8.3 L-BFGS-B Restart Analysis

To investigate whether increasing the number of random restarts improves L-BFGS-B performance, the multistart L-BFGS-B optimizer was tested with varying restart counts:

| Restarts | Raw Data | Eigenvalue Filtered | Time |
|:--------:|:--------:|:-------------------:|:----:|
| 8 | 0.9370 | 0.9687 | 0.3s |
| 50 | 0.9605 | 0.9896 | 1.3s |
| 100 | 0.9415 | 0.9910 | 1.3s |
| 150 | ~0.96–0.97 | ~0.97 | ~2s |
| **Global optimum** (other methods) | **0.9741** | **0.9973** | — |

L-BFGS-B plateaus at F ≈ 0.96–0.97 regardless of restart count (8 to 150+). The results are also **non-monotonic** — 100 restarts can perform worse than 50 due to the stochastic nature of random initialization. This confirms that the gradient-based approach is fundamentally limited by the non-convex structure of the waveplate cost landscape.

### 8.4 Key Findings

1. **Eigenvalue filtering improves all results**: Corrected fidelity increases from ~0.9741 → ~0.9973 across all optimizers (except L-BFGS-B).
2. **L-BFGS-B is unsuitable**: Gradient-based optimization gets trapped in local minima regardless of restart count.
3. **All derivative-free optimizers converge to the same global optimum**: Powell, DE, DE Robust, DE Multi-run, Hybrid, and COBYLA all reach F ≈ 0.9741 (raw) / 0.9973 (filtered).
4. **COBYLA is the fastest** (~0.02s) while matching the expensive DE methods.
5. **Fidelity ceiling at 0.9973**: The residual gap (Δ ≈ 0.0027) represents non-unitary noise that waveplates cannot correct.

**Recommendation:** Use **Powell Multistart** or **COBYLA** for fast, reliable results. Use **DE Robust** when maximum reliability is critical (e.g., laboratory calibration). Avoid L-BFGS-B for this non-convex problem.

---

## 9. Correction and Output

Once optimal angles {α₁*, β₁*, γ₁*, α₂*, β₂*, γ₂*} are found:

1. **Construct** U_total from the best angles
2. **Apply**: ρ_corrected = U_total · ρ_in · U_total†
3. **Evaluate**: Fidelity (raw), Fidelity (corrected), Purity Tr(ρ²), and the improvement Δ fidelity

---

## 10. Visualization

The simulation produces diagnostic plots via the [Streamlit GUI](file:///Users/anuragchaudhary/Documents/qhqEnt/gui.py):

- **Population Bar Chart**: Diagonal elements of ρ (probabilities of |HH⟩, |HV⟩, |VH⟩, |VV⟩), comparing input vs. corrected
- **Density Matrix Heatmap**: |ρ_corrected| with cell annotations, using the YlOrRd colormap
- **Target vs. Corrected Comparison**: Side-by-side numerical matrix display

---

## 11. Software Architecture

```
qhqEnt/
├── gui.py                    # Streamlit web interface
├── main.py                   # CLI entry point
├── tomography.txt            # Sample tomography data
├── src/
│   ├── io.py                 # Density matrix loading & normalization
│   ├── waveplates.py         # QWP and HWP Jones matrices
│   ├── qhqstates.py          # QHQ arm & total unitary construction
│   ├── states.py             # Bell state definitions
│   ├── fidelity.py           # Uhlmann fidelity computation
│   ├── cost.py               # Optimization cost function (1 − F)
│   └── optimizer.py          # Seven optimization algorithms
```

### Module Dependency Flow

```
tomography.txt → io.py (load & normalize) → optimizer.py → cost.py → qhqstates.py → waveplates.py
                                                  ↓                         ↓
                                            states.py (target)       fidelity.py
                                                  ↓
                                            ρ_corrected (output)
```

### Dependencies
- **NumPy** — linear algebra, eigenvalue decomposition
- **SciPy** — optimization routines (`minimize`, `differential_evolution`)
- **Matplotlib** — plotting
- **Streamlit** — interactive web GUI

---

## 12. Simulation Limitations

> [!IMPORTANT]
> The following limitations define the boundary conditions within which results from this simulation should be interpreted.

### 11.1 Unitary-Only Correction Model
The QHQ compensator applies a **unitary transformation** to the density matrix. This means it can only correct for **coherent (unitary) polarization rotations** — birefringence, fiber stress, and geometric phase accumulation. It **cannot correct for**:
- **Depolarizing noise** (isotropic mixing with the maximally mixed state)
- **Dephasing** (loss of off-diagonal coherence without population redistribution)
- **Photon loss** or asymmetric channel attenuation
- **Non-unitary errors** in general (any process that reduces the purity of the state)

If the dominant source of fidelity degradation is depolarization or decoherence rather than a unitary rotation, the QHQ optimization will plateau well below F = 1.

### 11.2 Ideal Waveplate Assumption
The simulation models waveplates as **perfect optical elements** — exact π/2 (QWP) and π (HWP) retardation, zero absorption, and wavelength-independent behaviour. In practice:
- Real waveplates have **retardation errors** (±1–2° deviation from nominal π/2 or π)
- **Wavelength sensitivity**: retardation varies with photon wavelength; broadband sources exacerbate this
- **Surface reflections and absorption** reduce photon counts and introduce additional state perturbation
- **Mounting alignment uncertainty** adds systematic angular offsets beyond the optimized values

### 11.3 Tomography Accuracy Dependence
The entire optimization hinges on the quality of the input density matrix from quantum state tomography. Limitations include:
- **Finite counting statistics**: tomographic reconstruction from a limited number of photon coincidence counts introduces statistical noise in the matrix elements
- **Systematic measurement errors**: imperfect polarization analyzers, detector imbalance, and dark counts propagate into the reconstructed ρ_exp
- **Non-physical matrices**: raw maximum-likelihood or linear-inversion tomography may yield density matrices that are not strictly positive semi-definite, requiring post-hoc correction (Hermitianization, eigenvalue clipping)

### 11.4 Non-Convex Optimization Landscape
The 6-dimensional cost surface C = 1 − F has **multiple local minima**. Despite deploying global (DE) and multi-start strategies:
- There is **no guarantee of finding the global optimum** in finite runtime
- Different optimizer runs may converge to different local solutions, leading to **non-unique angle prescriptions**
- The outer iteration loop mitigates but does not eliminate this risk

### 11.5 Static Single-Shot Correction
The simulation performs a **one-time, static optimization** based on a single tomographic snapshot. It does not account for:
- **Time-varying polarization drift** in fiber channels (e.g., due to temperature fluctuations or mechanical vibration)
- **Adaptive or feedback-based correction** that tracks and compensates for drift in real time
- In a deployed system, periodic re-tomography and re-optimization would be required

### 11.6 Two-Qubit Polarization Space Only
The simulation is restricted to **two-qubit polarization-encoded states** in the {|H⟩, |V⟩} basis:
- It does not handle higher-dimensional entanglement (qutrits, multi-photon states)
- It does not model spatial, temporal, or frequency degrees of freedom that may be entangled with polarization
- The 4×4 density matrix framework limits applicability to biphoton polarization experiments

### 11.7 No Experimental Feedback Loop
The optimized waveplate angles are computed purely numerically. The simulation does not:
- Interface with physical motorized waveplate mounts
- Perform closed-loop verification (apply angles → re-measure → re-optimize)
- Account for mechanical hysteresis or step-resolution limits of rotation stages

### 11.8 Limited Target State Set
Only the four maximally entangled Bell states (Φ⁺, Φ⁻, Ψ⁺, Ψ⁻) are implemented as targets. Arbitrary target states (non-maximally entangled or mixed states) are not supported.

### 11.9 Numerical Precision
- Eigenvalue-based fidelity computation clips small negative eigenvalues to zero, which introduces a minor systematic bias
- Waveplate angle resolution is limited by floating-point precision (~10⁻¹⁵ rad), which is far below the mechanical resolution of physical mounts (~0.01°)
- For near-pure states with very high fidelity (F > 0.999), numerical noise in the eigenvalue computation can dominate the residual cost

---

## 13. Mathematical Summary

| Quantity | Expression |
|----------|-----------|
| QWP Jones matrix | U_Q(θ) — see Section 2.1 |
| HWP Jones matrix | U_H(θ) — see Section 2.1 |
| Single-arm unitary | U_arm = U_Q(α) · U_H(β) · U_Q(γ) |
| Total two-photon unitary | U_total = U_arm1 ⊗ U_arm2 |
| Corrected state | ρ_corrected = U_total · ρ_in · U_total† |
| Uhlmann fidelity | F(ρ, σ) = (Σᵢ √λᵢ)², λᵢ = eig(ρ·σ) |
| Cost function | C = 1 − F(ρ_target, ρ_corrected) |
| Purity | P = Tr(ρ²), with 1/d ≤ P ≤ 1 |

### Parameter Space
- **Dimension:** 6 (three angles per arm)
- **Domain:** [0°, 180°]⁶
- **Landscape:** Non-convex, multiple local minima

---

## 14. Significance

1. **Generality:** QHQ can correct any unitary polarization distortion — applicable to arbitrary fiber and free-space channels
2. **Computational calibration:** Optimal angles determined computationally from a single tomographic measurement, replacing tedious manual alignment
3. **Robustness:** Multiple optimization strategies ensure reliable convergence in multi-modal cost landscapes
4. **Practical output:** Optimized angles can be directly applied to physical waveplate mounts in a laboratory setting
