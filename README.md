# 🧠 Brain Tumor Segmentation Using FFT Filters

**BME 271D Final Project**  
*Ege Özemek, Max [Last Name], Sasha [Last Name]*  
Duke University, Fall 2024

---

## 📋 Project Overview

This project demonstrates **frequency-domain image segmentation** techniques for brain tumor detection in MRI scans. We implement Fast Fourier Transform (FFT) based filtering methods and compare their performance against traditional spatial-domain approaches.

### Key Features

- **FFT-Based Segmentation**: High-pass and band-pass filtering in frequency domain
- **Baseline Comparisons**: Otsu thresholding and blob detection
- **Quantitative Evaluation**: Dice coefficient, IoU, and boundary accuracy metrics
- **Interactive Demo**: Google Colab notebook with easy image upload
- **Multi-Tier Confidence**: HIGH/MODERATE/LOW confidence detection system

---

## 🎯 Motivation

Automated tumor segmentation in medical imaging is crucial for:
- **Surgical Planning**: Precise tumor localization
- **Treatment Monitoring**: Tracking tumor growth/shrinkage
- **Workflow Efficiency**: Reducing radiologist workload

Frequency-domain methods offer unique advantages:
- Enhanced edge detection through high-frequency filtering
- Texture analysis via band-pass filtering
- Complementary information to spatial methods

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

1. **Open the notebook**: [Tumor_Segmentation_Final_v2.ipynb](Tumor_Segmentation_Final_v2.ipynb)
2. **Upload to Google Colab**: Click "Open in Colab" button
3. **Run all cells**: Runtime → Run all
4. **Upload your MRI image**: When prompted in Cell 2
5. **Optional: Upload ground truth mask**: When prompted in Cell 3 (for quantitative metrics)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/egeozemek/tumor-segmentation.git
cd tumor-segmentation

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook Tumor_Segmentation_Final_v2.ipynb
```

---

## 📊 Methods

### 1. FFT High-Pass Filtering
- **Purpose**: Emphasize tumor boundaries and edges
- **Implementation**: Remove low-frequency components (smooth background)
- **Best for**: Well-defined tumor margins

### 2. FFT Band-Pass Filtering
- **Purpose**: Capture tumor texture and heterogeneity
- **Implementation**: Isolate mid-frequency range
- **Best for**: Irregular or textured tumors

### 3. Otsu Thresholding (Baseline)
- **Purpose**: Automatic intensity-based segmentation
- **Implementation**: Maximize inter-class variance
- **Comparison**: Spatial-domain baseline

### 4. Blob Detection
- **Purpose**: Detect bright, compact regions
- **Implementation**: Shape-based filtering
- **Application**: High-intensity tumor masses

### 5. Hybrid Ensemble
- **Purpose**: Combine multiple methods for robust detection
- **Implementation**: Weighted voting (FFT: 1.5 votes, Blob: 1.0 vote)
- **Confidence**: Multi-tier system (HIGH/MODERATE/LOW)

---

## 📈 Evaluation Metrics

When ground truth masks are provided, the system calculates:

### Dice Similarity Coefficient (DSC)
```
DSC = 2 × |A ∩ B| / (|A| + |B|)
```
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Overlap between prediction and ground truth

### Intersection over Union (IoU)
```
IoU = |A ∩ B| / |A ∪ B|
```
- **Range**: 0 to 1 (higher is better)
- **More strict** than Dice coefficient

### Boundary Accuracy
- **Tolerance**: 2 pixels (~2-4mm in typical MRI)
- **Measures**: Precision of tumor boundary delineation
- **Clinical relevance**: Important for surgical planning

---

## 🎓 Course Concepts Applied

This project demonstrates key concepts from BME 271D:

### Fourier Transform
- Converting spatial domain to frequency domain
- Understanding magnitude and phase components
- FFT for computational efficiency

### Filtering
- **High-pass filters**: Emphasize edges and boundaries
- **Band-pass filters**: Isolate specific frequency ranges
- **Low-pass filters**: Smooth and denoise (preprocessing)

### Signal Processing
- Frequency response analysis
- Convolution and morphological operations
- Multi-scale analysis

---

## 📁 Project Structure

```
tumor-segmentation/
├── Tumor_Segmentation_Final_v2.ipynb   # Main interactive notebook
├── tumor_segmentation.py               # Core FFT functions
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
└── .gitignore                          # Git ignore rules
```

---

## 🖼️ Example Results

### High-Contrast Tumor (tumor_VI)
- **FFT High-Pass**: 77.1% confidence, Dice: 0.217
- **FFT Band-Pass**: 78.4% confidence, Dice: 0.224
- **Hybrid Combined**: 98.7% confidence (HIGH)
- **Outcome**: ✅ Successfully detected and localized

### Low-Contrast Tumor (tumor_III)
- **FFT High-Pass**: 65.2% confidence, Dice: 0.217
- **FFT Band-Pass**: 72.1% confidence, Dice: 0.224
- **Hybrid Combined**: 88.4% confidence (HIGH)
- **Outcome**: ⚠️ Detected but with lower metrics

### Key Finding
**Performance depends on tumor characteristics:**
- High-contrast tumors: Excellent detection (>95% confidence)
- Low-contrast tumors: More challenging but still detected
- Method demonstrates importance of image quality in automated segmentation

---

## 🔬 Clinical Implications

### Strengths
✅ **High precision**: FFT methods selective for bright abnormalities  
✅ **Automated localization**: Green cross marks tumor center  
✅ **Confidence scoring**: Helps prioritize urgent cases  
✅ **Complementary information**: Frequency domain reveals features spatial methods miss

### Limitations
⚠️ **Sensitivity to contrast**: Works best on high-quality, contrast-enhanced images  
⚠️ **Parameter tuning**: Requires adjustment for different imaging protocols  
⚠️ **False positives**: Can detect other bright structures (blood vessels, artifacts)  
⚠️ **No 3D analysis**: Currently processes 2D slices only

### Future Improvements
- **Adaptive thresholding**: Automatically adjust parameters per image
- **Multi-modal fusion**: Combine T1, T2, FLAIR sequences
- **3D volumetric analysis**: Process entire MRI volumes
- **Machine learning integration**: Use FFT features for deep learning
- **Real-time processing**: Optimize for clinical workflow

---

## 📦 Dependencies

```txt
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-image>=0.18.0
pandas>=1.3.0
Pillow>=8.3.0
```

All dependencies are listed in `requirements.txt` and automatically installed in Colab.

---

## 🧪 How to Use Your Own Images

### Step 1: Prepare Your MRI Image
- **Format**: JPEG, PNG, or other common image formats
- **Modality**: T1-weighted contrast-enhanced works best
- **Quality**: Higher resolution = better results

### Step 2: (Optional) Prepare Ground Truth Mask
- **Format**: PNG with binary mask (white = tumor, black = background)
- **Dimensions**: Must match original image exactly
- **Purpose**: Calculate quantitative metrics (Dice, IoU, Boundary Accuracy)

### Step 3: Run the Notebook
1. Open `Tumor_Segmentation_Final_v2.ipynb` in Colab
2. Run setup cell (Cell 1)
3. Upload your MRI in Cell 2
4. Upload mask in Cell 3 (optional)
5. Run all remaining cells

### Step 4: Interpret Results
- **HIGH confidence (>65%)**: Strong tumor detection, urgent follow-up recommended
- **MODERATE confidence (35-65%)**: Suspicious finding, additional imaging advised
- **LOW confidence (20-35%)**: Possible abnormality, monitor and re-evaluate
- **No tumor detected (<20%)**: No significant abnormality detected

---

## 📚 References

### Medical Imaging
1. **The Cancer Imaging Archive (TCIA)**  
   https://www.cancerimagingarchive.net/

2. **BraTS Challenge** - Brain Tumor Segmentation  
   http://braintumorsegmentation.org/

### Segmentation Methods
3. **Otsu, N. (1979)**  
   "A threshold selection method from gray-level histograms"  
   *IEEE Trans. Systems, Man, and Cybernetics*

4. **Canny, J. (1986)**  
   "A computational approach to edge detection"  
   *IEEE Trans. Pattern Analysis and Machine Intelligence*

### Evaluation Metrics
5. **Dice, L.R. (1945)**  
   "Measures of the amount of ecologic association between species"  
   *Ecology*, 26(3), 297-302

6. **Taha, A.A. & Hanbury, A. (2015)**  
   "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool"  
   *BMC Medical Imaging*, 15(1), 29

---

## 🤝 Contributing

This is a course project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Ege Özemek** - Duke University  
**Max [Last Name]** - Duke University  
**Sasha [Last Name]** - Duke University

*BME 271D: Signals and Systems in Biomedical Engineering*  
*Instructor: [Professor Name]*  
*Fall 2024*

---

## 🙏 Acknowledgments

- Duke University BME Department
- BME 271D teaching staff
- The Cancer Imaging Archive for medical images
- Open-source Python scientific computing community

---

## 📞 Contact

For questions about this project:
- **GitHub Issues**: [Create an issue](https://github.com/egeozemek/tumor-segmentation/issues)
- **Email**: [your-email@duke.edu]

---

## ⭐ Citation

If you use this code or methodology in your work, please cite:

```bibtex
@misc{ozemek2024tumor,
  title={Brain Tumor Segmentation Using FFT Filters},
  author={Özemek, Ege and [Max Last Name] and [Sasha Last Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/egeozemek/tumor-segmentation}
}
```

---

<div align="center">

**Made with 🧠 at Duke University**

[⬆ Back to Top](#-brain-tumor-segmentation-using-fft-filters)

</div>
