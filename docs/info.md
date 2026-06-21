The analysis of brain connectivity using functional Near-Infrared Spectroscopy (fNIRS) is a multifaceted process that involves measuring coordinated neural oscillations to understand how different brain regions interact. The most important aspects of this analysis range from the fundamental definitions of connectivity types to the rigorous preprocessing required to ensure signal quality.

### 1. Types of Connectivity
Connectivity analysis in fNIRS is generally divided into two main categories:
*   **Functional Connectivity (FC):** This refers to **statistical dependencies** between remote neurophysiological events. It is typically inferred by calculating the correlation (such as Pearson's or Spearman's) between the time series of different brain regions or channels.
*   **Effective Connectivity:** This goes beyond simple correlation to describe the **directed causal influence** that one neural system exerts over another. Common models for estimating this include **Granger Causality** and **Dynamic Causal Modelling (DCM)**.

### 2. Critical Preprocessing and Quality Control
Because fNIRS signals are highly susceptible to noise, rigorous preprocessing is essential for valid connectivity results:
*   **Signal Conversion:** Raw light intensity must be converted to optical density and then to relative concentration changes of **oxygenated (HbO), deoxygenated (HbR), and total hemoglobin (HbT)** using the Modified Beer-Lambert Law. 
*   **Filtering:** A bandpass filter (typically **0.01–0.1 Hz**) is applied to isolate low-frequency oscillations associated with spontaneous neural activity while removing high-frequency physiological noise like heart rate and respiration.
*   **Motion Artifact Correction:** fNIRS is sensitive to head motion, necessitating correction algorithms such as **Spline interpolation, Wavelet decomposition**, or Correlation-based Signal Improvement (CBSI).
*   **Systemic Noise Removal:** Connectivity estimates can be artificially inflated by extracerebral signals (e.g., scalp blood flow). This is addressed using **short-channel regression** or spatial filters like **Principal Component Analysis (PCA)**.

### 3. Connectivity Calculation Methods
Researchers use different spatial approaches to derive connectivity:
*   **Seed-based Analysis:** Connectivity is calculated by predefining a "seed" region and computing its temporal correlation with all other regions.
*   **Whole-brain (or Whole-probe) Analysis:** This examines the temporal correlation between all possible pairs of measurement channels to construct a complete connectivity matrix.
*   **Region of Interest (ROI) Analysis:** Channels are grouped into functionally relevant anatomical regions, and the averaged time series for these ROIs are correlated to reduce data dimensionality.

### 4. Graph Theory Framework
Modern fNIRS analysis frequently employs **graph theory** to characterise the topological organization of the brain as a complex network. In this model, brain regions or channels are **nodes**, and the functional connections between them are **edges**. Key metrics include:
*   **Global Metrics:** These describe the whole-network architecture, such as **global efficiency** (ease of information transfer), **clustering coefficient** (segregation into local groups), and **modularity** (strength of community structure).
*   **Local/Nodal Metrics:** These identify the importance of specific regions, such as **degree centrality** (number of connections) and **betweenness centrality**, to locate "hubs" that are critical for network communication.

### 5. Emerging Applications: Brain Fingerprinting
Recent research has validated that resting-state functional connectivity (rsFC) patterns are sufficiently unique to allow for **subject identification**, known as **"brain fingerprinting"**. Studies show that fNIRS can identify individuals with high accuracy (up to 98%), provided there is sufficient spatial coverage and enough data runs (typically at least four runs) to overcome the low signal-to-noise ratio inherent in individual-level measurements.