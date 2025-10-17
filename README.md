## 🧠 Neuropixels Cortico–Striatal Analysis Pipeline

### Overview

This repository contains a **MATLAB-based analysis pipeline** developed for *in vivo* Neuropixels electrophysiological recordings.
It was designed to quantify **cortico–striatal synchronization**, **spike–LFP coupling**, and **SWD/IID event dynamics** in *Scn2a*-deficient mice.

The code supports preprocessing, event detection, and visualization of large-scale multi-channel recordings, and was used to generate Neuropixels figures in the manuscript:

> *Cortico–striatal desynchronization underlies seizure susceptibility in Scn2a-deficient mice* (Zhao et al., in preparation)

---

### Features

* **Preprocessing and baseline correction**
  Removes 60 Hz noise, performs detrending and bandpass filtering (0.5–250 Hz).
* **SWD/IID event detection**
  Peak-based algorithm with Hilbert envelope alignment for precise onset identification.
* **Spike–LFP integration**
  Aligns single-unit spike times to SWD windows for rate and phase analysis.
* **Region and neuron classification**
  Automatically labels units by recording depth and waveform type (Ctx, Str, Pal, MSN, FSI, UIN).
* **Visualization**
  Generates event-centered LFP traces, raster plots, and PETHs for cortical vs. striatal comparison.
* **Data export**
  Outputs `.mat` and `.xlsx` results for reproducibility and figure generation.

---

### File Structure

```
/main/
 ├─ detect_swd_v3.m           # Main SWD detection function
 ├─ preprocess_lfp.m          # Filtering and baseline correction
 ├─ plot_swd_events_v2.m      # Event visualization
 ├─ compute_rate_CTX_STR.m    # Spike rate analysis per region
 ├─ export_results.m          # Save results to Excel and MAT files
 └─ example_data/             # Sample dataset (if applicable)
```

---

### Requirements

* **MATLAB R2021a or later**
* **Signal Processing Toolbox**
* (Optional) Statistics and Machine Learning Toolbox

---

### Usage

```matlab
% Example: Detect SWD events and visualize
lfp_data = load('example_data/LFP_mouse01.mat');
swd_events = detect_swd_v3(lfp_data);

% Compute cortical and striatal rates
[rate_CTX, rate_STR] = compute_rate_CTX_STR(swd_events);

% Export to Excel
export_results(rate_CTX, rate_STR, 'output.xlsx');
```

---

### Citation

If you use this code in your research, please cite:

> Zhao, Y.R. *et al.* (in preparation).
> *Cortico–striatal desynchronization underlies seizure susceptibility in Scn2a-deficient mice.*

---

### Third-party Code Acknowledgment
This repository includes adapted analytical components from the following publication:

- **Sorokin et al., Nature Neuroscience (2021).**  
  *Network mechanisms of absence epilepsy*  
  [https://www.nature.com/articles/s41593-021-00997-0](https://www.nature.com/articles/s41593-021-00997-0)

The SWD detection logic (`detect_swd.m`) was adapted from algorithms described in this paper to fit
Neuropixels electrophysiological data.  
All rights for the original design belong to the authors of the cited publication.
