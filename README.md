# Robust Pitch Detection — Experimental Pipeline

A research pipeline implementing the phases described in your proposal, optimized for monophonic audio analysis using Wiener filtering and pYIN.

---

## Architecture & Phases

The pipeline executes the following logic as defined in `pipeline.py`:

1.  **Phase 1: Dataset Preparation** Loads `.wav` files, trims silence at 30dB, and normalizes peak amplitude to 0.89. Ground truth is derived from the filename (e.g., "A4" maps to 440.0Hz).
2.  **Phase 2: Noise Injection** Adds AWGN (Additive White Gaussian Noise) based on specific SNR targets.
3.  **Phase 3: Denoising (Wiener Filter)** Uses an **Optimal Linear Wiener Filter**. It estimates noise power ($P_n$) from the bottom 20% energy frames and applies a gain mask: 
    $$\text{Gain} = \frac{|S|^2 - \alpha P_n}{|S|^2}$$
4.  **Phase 4: Pitch Detection** Uses `librosa.pyin` for fundamental frequency ($f_0$) estimation on both the noisy baseline and the denoised signal.
5.  **Phase 5: Evaluation** Calculates accuracy by comparing estimated $f_0$ against the target note frequency.

---

## Evaluation Metrics

| Metric | Formula / Logic |
| :--- | :--- |
| **RPA** | Raw Pitch Accuracy: % of frames where $|error| < 50$ cents. |
| **Mean Error** | The average absolute deviation in cents for all voiced frames. |
| **Cent Calculation** | $1200 \times \log_2(f_{est} / f_{gt})$ |
