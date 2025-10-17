我不能直接打开外部网页（包括 GitHub 链接），但你给的仓库信息足够了（文件结构、简介、MIT 许可证、语言比例等我都看到了）。
下面是我为你量身写的新版 **README.md**，语气学术、清晰且正式，适合公开科研项目（PTZ 诱导 Neuropixels 同步事件分析）。

---

```markdown
# PTZ-induced Cortico–Striatal Synchronization Analysis (MATLAB)

This repository provides an open-source MATLAB pipeline for analyzing **PTZ-triggered abnormal synchronization events** in Neuropixels recordings.  
It enables standardized preprocessing, SWD/IID detection, and cortico–striatal cross-regional analysis of local field potentials (LFPs) and spike activity, particularly in *Scn2a*-deficient mouse models.

---

## 🧠 Overview

The pipeline is designed for high-density electrophysiology experiments investigating seizure-related network synchronization across cortex and striatum.  
It provides:
- Robust baseline correction and band-limited filtering of Neuropixels LFPs  
- Detection of spike–wave discharges (SWDs) and interictal discharges (IIDs)  
- Burst detection and time-locked population rate analysis  
- Region-specific (STR vs CTX) raster plotting and peak-timing extraction  
- Excel export of per-event cortico–striatal peak latencies  

The analysis reproduces the main processing stages of PTZ-induced synchronization and cortico–striatal delays in *Scn2a* mutant mice.

---

## ⚙️ Key Features

| Category | Description |
|-----------|--------------|
| **Preprocessing** | Bandpass filtering (5–60 Hz), baseline correction, artifact rejection |
| **Event Detection** | SWD/IID detection based on mean-LFP envelope and dynamic thresholds |
| **Cross-Region Analysis** | Automatic striatum–cortex neuron separation from depth map |
| **Burst Analysis** | Logical mask construction for 30 kHz raster alignment |
| **Visualization** | Multi-panel figure generator (spectrogram, filtered LFP, raster, burst overlay) |
| **Export** | Excel/`.mat` output of event windows and regional peak latencies |

---

## 🧩 Repository Structure

```

PTZ_induced_sync/
├─ LICENSE                         # MIT License
├─ README.md                       # Documentation
├─ baseline_correction.m           # Baseline correction algorithm
├─ detect_swd_v2.m                 # SWD/IID detection on mean LFP
├─ neuron_seperation_cor_str.m     # Depth-based STR/CTX labeling
├─ compute_bursts.m                # Burst logical mask generator
├─ aline_LFP_with_raster.m         # LFP–spike raster alignment
├─ on_off_lfp_raster_baseline_*.m  # Burst visualization scripts
├─ ploting_abnormal_LFP.m          # Combined summary figure
├─ datReadToNpy.py                 # NPY conversion utility
└─ external/ (optional)            # Third-party functions if added

````

---

## 🧪 Quick Start

```matlab
% Example workflow
loading_lfp
corrected_baseline = baseline_correction(mean(d), 2500);
A = plot_filtered_lfp(mean(d), 2500);
detect_swd(corrected_baseline);

% Resample SWD windows to 30 kHz for spike alignment
run('compute_bursts.m');

% Region separation and population analysis
run('neuron_seperation_cor_str.m');
````

Output files:

* `neuron_region_labels.mat` — region tags and IDs
* `swd_peak_times_ctx_str.xlsx` — per-event peak timing summary

---

## 📊 Dependencies

* MATLAB R2022b or newer
* Signal Processing Toolbox
* Optional: Python 3.10+ for `.npy` reading (`datReadToNpy.py`)

---

## 🧬 Data Availability

Raw Neuropixels recordings and LFP data are **not included** due to size and privacy constraints.
The scripts are verified on recordings sampled at:

* **AP:** 30 kHz (spike band)
* **LFP:** 2.5 kHz

You can replace file paths in the scripts with your own data folders.

---

## 📚 Citation

If you use this repository or adapt parts of the analysis, please cite the original study whose algorithm inspired this work:

> *Adapted from methodology described in*
> **Nature Neuroscience (2021), DOI: [10.1038/s41593-021-00997-0]**
> "Genetic and circuit-level mechanisms of spike-wave discharges"

and this repository:

> Zhao, Y. (2025). *PTZ-induced cortico–striatal synchronization pipeline (MATLAB)*.
> GitHub: [https://github.com/GrayZZZhao/PTZ_induced_sync](https://github.com/GrayZZZhao/PTZ_induced_sync)

---

## 🪪 License

This repository is released under the [MIT License](LICENSE).
You are free to use, modify, and distribute this code with proper attribution.
Components derived from other open-source works retain their original licenses.

---

## 🧭 Contact

**Author:** Gray (Yuanrui) Zhao
Ph.D. Candidate, Yang Lab, Purdue University
Email: *[yuanruizhao@purdue.edu](mailto:yuanruizhao@purdue.edu)* (or your preferred contact)

---

## 🧩 Acknowledgments

* Dr. Yang Yang and lab members for experimental design and discussions
* Adapted algorithmic components from *Nat Neurosci* (2021) for SWD detection
* MATLAB/Python community contributors for open-source analysis tools

```

---

是否希望我帮你加一个 **README 的短版（300 字以内学术摘要）**，用于 GitHub 的简介栏（repo description + about）？那部分会显示在右上角简介区。
```
