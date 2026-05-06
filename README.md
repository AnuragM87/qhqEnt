# Polarization Correction QHQ

Quantum waveplate (QHQ) angle optimization for Bell state fidelity recovery.

## Setup

```bash
# Create virtual environment (first time only)
python3 -m venv qhq_env

# Activate environment
source qhq_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit
```

## Run the App

```bash
source qhq_env/bin/activate
streamlit run gui.py
```

Opens at **http://localhost:8501**

## Usage

1. **Load data** — upload a tomography file or use the default `tomography.txt`
2. **Select target** — choose the Bell state (Φ⁺, Φ⁻, or Ψ⁻)
3. **Preprocessing** — optionally apply eigenvalue filtering or depolarization compensation
4. **Choose optimizer** — DE Robust is recommended for best results
5. **Run** — click "Run Optimization" and view the corrected fidelity, waveplate angles, and plots

## CLI (alternative)

```bash
source qhq_env/bin/activate
python main.py
```

## Tomography File Format

Plain text, 4×4 complex matrix (space-separated). Example:

```
0.42 0 0 (0.36+0.12j)
0 0.08 (0.02-0.06j) 0
0 (0.02+0.06j) 0.08 0
(0.36-0.12j) 0 0 0.42
```
