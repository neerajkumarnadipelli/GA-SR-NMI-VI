# GA-SR-NMI-VI: Efficient Hyperspectral Band Selection using Genetic Algorithm and Similarity Ranking

This repository implements **GA-SR-NMI-VI**, a hybrid band selection approach for hyperspectral image classification. It combines **Similarity-based Ranking** (SR-NMI-VI) and a **Genetic Algorithm** (GA) to optimize the selection of informative spectral bands, improving accuracy while reducing computational cost.

---
## 📁 Repository Structure

```text
├── GA-SR-NMI-VI.py                   # Core implementation of the GA-SR-NMI-VI algorithm  
├── GA-SR-NMI-VI_cubertdrone.ipynb    # Experiment notebook for Cubert Drone dataset  
├── GA-SR-NMI-VI_hanchaun.ipynb       # Experiment notebook for WHU-Hi-HanChuan dataset  
├── GA-SR-NMI-VI_longkou.ipynb        # Experiment notebook for WHU-Hi-LongKou dataset  
├── GA-SR-NMI-VI_oilspill.ipynb       # Experiment notebook for Oil Spill dataset  
├── Images/                           # Folder for figures, plots, or outputs  

---

## 📖 How It Works

### 🔸 Step 1: Band Ranking - SR-NMI-VI
- Uses **Normalized Mutual Information (NMI)** and **Variation of Information (VI)**.
- Assigns ranks to spectral bands based on redundancy and diversity.

### 🔸 Step 2: Optimal Band Selection - GA
- Genetic Algorithm searches for the optimal subset from top-ranked bands.
- Fitness is evaluated using classification performance (e.g., SVM accuracy).

### 🔸 Step 3: Classification
- The selected bands are evaluated using ML/DL classifiers (SVM, 3D-CNN etc.).

---

## 📊 Datasets

Each notebook runs experiments on a specific public hyperspectral dataset:

| Dataset         | Description                                               | Classes | Bands | Source            |
|----------------|-----------------------------------------------------------|---------|-------|-------------------|
| Oil Spill      | Gulf of Mexico (Oil vs Water)                             | 2       | 224   | IEEE Dataport     |
| Cubert Drone   | Precision agriculture crops over UAS Bangalore            | 4       | 138   | UAV Cubert        |
| WHU-Hi-LongKou | Crop classification (Corn, Soybeans, etc.)                | 9       | 274   | RSIDEA            |
| WHU-Hi-HanChuan| Mixed vegetation, roofs, soil, water                      | 16      | 270   | RSIDEA            |

* Due to size or licensing, datasets are not included. Please download from official sources.*


