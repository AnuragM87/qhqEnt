import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import sys
import contextlib

matplotlib.use("Agg")

# ── Backend imports ──────────────────────────────────────────
from src.io import load_density_matrix
from src.states import rho_phi_plus, rho_phi_minus, rho_psi_minus
from src.optimizer import (
    find_qhq_angles_multistart,
    find_qhq_angles_hybrid,
    find_qhq_angles_de,
    find_qhq_angles_de_robust,
    find_qhq_angles_powell_multistart,
    find_qhq_angles_cobyla,
    find_qhq_angles_de_multirun,
)
from src.qhqstates import U_total
from src.fidelity import uhlmann_fidelity
from src.noise_compensation import depolarization_compensation, eigenvalue_filter

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Polarization Correction QHQ",
    page_icon="⚛️",
    layout="wide",
)

# ── Custom CSS for scientific look ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

    /* Hide Streamlit top header bar */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Global */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }
    h1 { color: #58a6ff !important; font-size: 1.6rem !important; }
    h2 { color: #c9d1d9 !important; font-size: 1.2rem !important; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
    h3 { color: #8b949e !important; font-size: 1.0rem !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #58a6ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #8b949e !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] h2 {
        color: #58a6ff !important;
        font-size: 1.0rem !important;
    }

    /* Matrix display */
    .matrix-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        background: #161b22;
        border: 1px solid #21262d;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
        white-space: pre;
        color: #c9d1d9;
        line-height: 1.6;
    }

    /* Angles table */
    .angles-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .angles-table th {
        background: #21262d;
        color: #8b949e;
        padding: 8px 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #30363d;
    }
    .angles-table td {
        padding: 10px 12px;
        text-align: center;
        color: #e0e0e0;
        border-bottom: 1px solid #21262d;
    }
    .angles-table tr:hover td {
        background: #161b22;
    }

    /* Log console */
    .log-console {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: #0d1117;
        border: 1px solid #21262d;
        padding: 12px;
        border-radius: 4px;
        color: #7ee787;
        max-height: 250px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.5;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .status-ready { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; }
    .status-running { background: #d2992222; color: #d29922; border: 1px solid #d29922; }
    .status-done { background: #23863622; color: #3fb950; border: 1px solid #238636; }
    .status-error { background: #da363322; color: #f85149; border: 1px solid #da3633; }

    /* Section dividers */
    hr { border-color: #21262d !important; }

    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────

def format_matrix(mat, precision=4):
    """Format a complex matrix as aligned text."""
    rows = []
    for i in range(mat.shape[0]):
        cols = []
        for j in range(mat.shape[1]):
            v = mat[i, j]
            re, im = v.real, v.imag
            if abs(im) < 1e-10:
                cols.append(f"{re:>{precision+5}.{precision}f}")
            else:
                sign = "+" if im >= 0 else "-"
                cols.append(f"{re:.{precision}f}{sign}{abs(im):.{precision}f}j")
        rows.append("  ".join(cols))
    return "\n".join(rows)


def make_population_chart(rho_raw, rho_corrected=None):
    """Bar chart of diagonal populations."""
    labels = ["|HH⟩", "|HV⟩", "|VH⟩", "|VV⟩"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    probs_raw = np.real(np.diag(rho_raw))
    if rho_corrected is not None:
        probs_corr = np.real(np.diag(rho_corrected))
        ax.bar(x - width/2, probs_raw, width, label="Input", color="#1f6feb", edgecolor="#58a6ff", linewidth=0.5)
        ax.bar(x + width/2, probs_corr, width, label="Corrected", color="#238636", edgecolor="#3fb950", linewidth=0.5)
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    else:
        ax.bar(x, probs_raw, width, color="#1f6feb", edgecolor="#58a6ff", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color="#c9d1d9")
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Population", fontsize=9, color="#8b949e")
    ax.set_title("State Populations", fontsize=10, color="#c9d1d9", pad=8)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

    fig.tight_layout()
    return fig


def make_density_heatmap(rho, title="Density Matrix |ρ|"):
    """Heatmap of density matrix magnitudes."""
    labels = ["|HH⟩", "|HV⟩", "|VH⟩", "|VV⟩"]
    mag = np.abs(rho)

    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0e1117")
    im = ax.imshow(mag, cmap="inferno", vmin=0, vmax=0.55, aspect="equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=8, color="#c9d1d9")
    ax.set_yticklabels(labels, fontsize=8, color="#c9d1d9")
    ax.set_title(title, fontsize=10, color="#c9d1d9", pad=8)
    ax.tick_params(colors="#8b949e")

    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = mag[i, j]
            color = "#e0e0e0" if val < 0.3 else "#0d1117"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5, color=color, fontfamily="monospace")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    fig.tight_layout()
    return fig


# ── Target state map ─────────────────────────────────────────
TARGET_STATES = {
    "Φ⁺  (|HH⟩+|VV⟩)/√2": rho_phi_plus,
    "Φ⁻  (|HH⟩−|VV⟩)/√2": rho_phi_minus,
    "Ψ⁻  (|HV⟩−|VH⟩)/√2": rho_psi_minus,
}

OPTIMIZERS = {
    "Multistart L-BFGS-B": find_qhq_angles_multistart,
    "Powell Multistart": find_qhq_angles_powell_multistart,
    "Differential Evolution": find_qhq_angles_de,
    "DE Robust (recommended)": find_qhq_angles_de_robust,
    "DE Multi-run": find_qhq_angles_de_multirun,
    "Hybrid (DE → Powell)": find_qhq_angles_hybrid,
    "COBYLA": find_qhq_angles_cobyla,
}

# ── Session state init ───────────────────────────────────────
for key in ["rho_raw", "rho_in", "rho_target", "rho_out",
            "best_angles", "best_fidelity", "initial_fidelity",
            "log_output", "optimized", "preprocess_info"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "optimized" not in st.session_state:
    st.session_state.optimized = False


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Configuration
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚛️ Configuration")

    # ── File Upload ──
    st.markdown("### 📂 Tomography Data")
    upload_mode = st.radio("Input method", ["Upload file", "Use default (tomography.txt)"],
                           horizontal=True, label_visibility="collapsed")

    if upload_mode == "Upload file":
        uploaded = st.file_uploader("Tomography matrix file", type=["txt", "csv", "dat"],
                                     label_visibility="collapsed")
        if uploaded is not None:
            # Save temp file and load
            tmp_path = "/tmp/uploaded_tomography.txt"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            st.session_state.rho_raw = load_density_matrix(tmp_path)
    else:
        try:
            st.session_state.rho_raw = load_density_matrix("tomography.txt")
        except Exception as e:
            st.error(f"Could not load tomography.txt: {e}")

    st.divider()

    # ── Target State ──
    st.markdown("### 🎯 Target State")
    target_name = st.selectbox("Target Bell state", list(TARGET_STATES.keys()),
                                label_visibility="collapsed")
    st.session_state.rho_target = TARGET_STATES[target_name]()

    st.divider()

    # ── Preprocessing ──
    st.markdown("### 🔧 Preprocessing")
    preprocess = st.radio("Method", ["None (raw data)", "Eigenvalue Filter", "Depolarization Compensation"],
                          label_visibility="collapsed")

    if preprocess == "Eigenvalue Filter":
        eigen_rank = st.number_input("Rank (eigenstates to keep)", min_value=1, max_value=4, value=1)
    elif preprocess == "Depolarization Compensation":
        depol_p = st.number_input("Noise fraction p (auto if 0)", min_value=0.0, max_value=0.99, value=0.0, step=0.01)

    st.divider()

    # ── Optimizer ──
    st.markdown("### ⚡ Optimizer")
    optimizer_name = st.selectbox("Algorithm", list(OPTIMIZERS.keys()),
                                   index=3, label_visibility="collapsed")

    col_a, col_b = st.columns(2)
    with col_a:
        max_iters = st.number_input("Iterations", min_value=1, max_value=100, value=5)
    with col_b:
        target_fidelity = st.number_input("Target F", min_value=0.50, max_value=1.00, value=0.98, step=0.01)

    st.divider()

    # ── Run Button ──
    run_clicked = st.button("▶  Run Optimization", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════

st.markdown("# ⚛️ Polarization Correction QHQ")
st.caption("Quantum waveplate angle optimization for Bell state fidelity recovery")

if st.session_state.rho_raw is None:
    st.info("← Load tomography data from the sidebar to begin.")
    st.stop()

rho_raw = st.session_state.rho_raw
rho_target = st.session_state.rho_target

# ── Display raw matrix ───────────────────────────────────────
with st.expander("📋 Loaded Density Matrix (raw)", expanded=False):
    st.markdown(f'<div class="matrix-display">{format_matrix(rho_raw)}</div>', unsafe_allow_html=True)

# ── Run optimization ─────────────────────────────────────────
if run_clicked:
    log_buffer = io.StringIO()

    # Preprocessing
    with contextlib.redirect_stdout(log_buffer):
        if preprocess == "None (raw data)":
            rho_in = rho_raw
        elif preprocess == "Eigenvalue Filter":
            rho_in = eigenvalue_filter(rho_raw, rank=eigen_rank)
        elif preprocess == "Depolarization Compensation":
            p = depol_p if depol_p > 0 else None
            rho_in, _ = depolarization_compensation(rho_raw, p=p)

    st.session_state.rho_in = rho_in
    initial_fidelity = uhlmann_fidelity(rho_target, rho_raw)
    st.session_state.initial_fidelity = initial_fidelity

    # Optimization loop
    optimizer_fn = OPTIMIZERS[optimizer_name]
    best_fidelity = initial_fidelity
    best_angles = None

    progress_bar = st.progress(0, text="Optimizing...")

    with contextlib.redirect_stdout(log_buffer):
        for i in range(max_iters):
            angles, fidelity, _ = optimizer_fn(rho_in, rho_target)
            print(f"Iteration {i+1}/{max_iters}: fidelity = {fidelity:.6f}")

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_angles = angles

            progress_bar.progress((i + 1) / max_iters, text=f"Iteration {i+1}/{max_iters} — best F = {best_fidelity:.6f}")

            if best_fidelity >= target_fidelity:
                print(f"Target fidelity {target_fidelity} reached!")
                break

    progress_bar.empty()

    # Apply correction
    if best_angles is not None:
        U = U_total(*best_angles)
        rho_out = U @ rho_in @ U.conj().T
    else:
        rho_out = rho_in

    # Store results
    st.session_state.best_angles = best_angles
    st.session_state.best_fidelity = best_fidelity
    st.session_state.rho_out = rho_out
    st.session_state.log_output = log_buffer.getvalue()
    st.session_state.optimized = True

# ── Display results ──────────────────────────────────────────
if st.session_state.optimized:
    rho_in = st.session_state.rho_in
    rho_out = st.session_state.rho_out
    rho_target = st.session_state.rho_target
    best_angles = st.session_state.best_angles
    initial_fidelity = st.session_state.initial_fidelity
    fidelity_after = uhlmann_fidelity(rho_target, rho_out)
    fidelity_preprocessed = uhlmann_fidelity(rho_target, rho_in)
    purity_in = np.real(np.trace(rho_in @ rho_in))
    purity_out = np.real(np.trace(rho_out @ rho_out))

    # Status
    if best_angles is not None:
        st.markdown('<span class="status-badge status-done">✓ OPTIMIZATION COMPLETE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">⚠ NO IMPROVEMENT FOUND</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics row ──
    st.markdown("## 📊 Results")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Fidelity (raw)", f"{initial_fidelity:.4f}")
    with m2:
        st.metric("Fidelity (preprocessed)", f"{fidelity_preprocessed:.4f}",
                   delta=f"{fidelity_preprocessed - initial_fidelity:+.4f}")
    with m3:
        st.metric("Fidelity (corrected)", f"{fidelity_after:.4f}",
                   delta=f"{fidelity_after - initial_fidelity:+.4f}")
    with m4:
        st.metric("Purity (output)", f"{purity_out:.4f}",
                   delta=f"{purity_out - purity_in:+.4f}")

    st.markdown("---")

    # ── Angles + Matrices ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("## 🔩 Waveplate Angles")
        if best_angles is not None:
            a1, b1, g1, a2, b2, g2 = best_angles
            angles_html = f"""
            <table class="angles-table">
                <tr>
                    <th></th><th>QWP (α)</th><th>HWP (β)</th><th>QWP (γ)</th>
                </tr>
                <tr>
                    <td style="color:#58a6ff; font-weight:700;">Arm 1</td>
                    <td>{a1:.2f}°</td><td>{b1:.2f}°</td><td>{g1:.2f}°</td>
                </tr>
                <tr>
                    <td style="color:#58a6ff; font-weight:700;">Arm 2</td>
                    <td>{a2:.2f}°</td><td>{b2:.2f}°</td><td>{g2:.2f}°</td>
                </tr>
            </table>
            """
            st.markdown(angles_html, unsafe_allow_html=True)
        else:
            st.warning("Optimizer could not improve on initial fidelity. No angles to apply.")

    with col_right:
        st.markdown("## 📐 Corrected Density Matrix")
        st.markdown(f'<div class="matrix-display">{format_matrix(rho_out)}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Plots ──
    st.markdown("## 📈 Visualization")
    p1, p2 = st.columns(2)

    with p1:
        fig_pop = make_population_chart(rho_raw, rho_out)
        st.pyplot(fig_pop, use_container_width=True)
        plt.close(fig_pop)

    with p2:
        fig_heatmap = make_density_heatmap(rho_out, title="|ρ_corrected|")
        st.pyplot(fig_heatmap, use_container_width=True)
        plt.close(fig_heatmap)

    # ── Target comparison ──
    with st.expander("🎯 Target vs Corrected Comparison", expanded=False):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Target ρ**")
            st.markdown(f'<div class="matrix-display">{format_matrix(rho_target)}</div>', unsafe_allow_html=True)
        with tc2:
            st.markdown("**Corrected ρ**")
            st.markdown(f'<div class="matrix-display">{format_matrix(rho_out)}</div>', unsafe_allow_html=True)

    # ── Log console ──
    with st.expander("🖥️ Optimizer Log", expanded=False):
        log = st.session_state.log_output or "No log output."
        st.markdown(f'<div class="log-console">{log}</div>', unsafe_allow_html=True)

else:
    st.markdown("---")
    st.markdown("""
    ### Getting Started
    1. **Load** your tomography data (sidebar)
    2. **Configure** preprocessing, target state, and optimizer
    3. **Click Run** to optimize waveplate angles
    """)

    # Show raw state plots
    if rho_raw is not None:
        st.markdown("## 📈 Input State")
        p1, p2 = st.columns(2)
        with p1:
            fig = make_population_chart(rho_raw)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with p2:
            fig = make_density_heatmap(rho_raw, title="|ρ_raw|")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
