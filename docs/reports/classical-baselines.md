# Classical (non-neural) baselines for digit recognition

**Question.** All four architectures compared in Week 11 are neural networks, and three of them are convolutional. Do established non-neural methods compete with them on MNIST and on our photographed digits? Which feature representation matters?

## Method

Five classical families were compared with the networks under the same data contract (`hnrs_ml/classical.py`, `research/compare_classical_models.py`):

| Family | Features | Validation grid | Idea |
|---|---|---|---|
| `knn_pixels` | 784 raw pixels | k = 1, 3, 5 | Instance-based: label of the nearest training digits |
| `knn_hog` | 1,296 HOG values | k = 1, 3, 5 | Same, on gradient-orientation features |
| `linear_svm_hog` | HOG | C = 0.1, 1 | Maximum-margin linear separator (one-vs-rest) |
| `rbf_svm_hog` | HOG | C = 1, 10 | Kernel SVM: non-linear boundary via the RBF kernel (one-vs-one) |
| `random_forest_pixels` | raw pixels | 200 trees | Ensemble of decision trees |

**HOG features** (Dalal and Triggs, 2005) are implemented in NumPy: centred gradients, unsigned orientation in nine 20-degree bins with linear interpolation, 4x4-pixel cells, overlapping 2x2-cell blocks and L2-Hys normalisation, giving 1,296 values per crop. HOG describes stroke *direction* rather than pixel *intensity*, which we expected to matter for photographs whose ink thickness and contrast differ from MNIST.

**Controls.** Every model used the same 54,000/6,000 MNIST split as the CNN; the script refuses to run if the split hashes differ from `experiments/baseline/config.json`. Each family's setting was chosen by validation accuracy (ties keep the simpler, earlier setting). The 10,000-image official test set was loaded only after all choices were fixed. The photo comparison gives every model the **identical prepared crops** from the application's number route (connected components, Otsu preparation), so differences come from the classifier alone. The CNN's photo results reproduce the recorded Week 11 and final figures exactly (16/20, 19/20, 37/40, 16/20), which confirms the routes match.

## Official MNIST test results

| Model | Type | Validation | Test correct | Test accuracy | Macro F1 | Fit time | ms / digit | Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **rbf_svm_hog** (C=10) | Kernel SVM | 99.13% | 9,911 | **99.11%** | 0.9911 | 158 s | 7.35 | 88.8 MB |
| resnet18 | CNN | 98.70% | 9,864 | 98.64% | 0.9862 | GPU | 0.60 | 44.8 MB |
| cnn (application default) | CNN | 98.42% | 9,853 | 98.53% | 0.9853 | CPU | 0.08 | 0.4 MB |
| linear_svm_hog (C=1) | Linear SVM | 98.27% | 9,836 | 98.36% | 0.9835 | 22 s | 0.10 | 0.1 MB |
| knn_hog (k=3) | kNN | 97.67% | 9,784 | 97.84% | 0.9782 | <1 s | 1.10 | 280 MB |
| mobilenet_v3_small | CNN | 97.22% | 9,716 | 97.16% | 0.9713 | GPU | 0.49 | 6.2 MB |
| knn_pixels (k=3) | kNN | 97.02% | 9,705 | 97.05% | 0.9703 | <1 s | 0.59 | 170 MB |
| random_forest_pixels | Forest | 96.88% | 9,692 | 96.92% | 0.9690 | 12 s | 0.02 | 265 MB |
| mlp | MLP | 96.22% | 9,660 | 96.60% | 0.9657 | CPU | 0.01 | 0.4 MB |

Network rows are the recorded Week 11 benchmark. Timings are single CPU runs on one laptop (classical prediction includes HOG extraction) and are indicative only. Sizes for classical models are pickled scikit-learn objects; kNN must store the whole training set.

The RBF-SVM's most frequent test confusions are 2→7, 7→9 and 9→4 (six each), which are the same shape pairs that trouble the networks.

## Photographed digits (identical crops)

| Collection | CNN | rbf_svm_hog | linear_svm_hog | knn_hog | knn_pixels | random_forest_pixels |
|---|---:|---:|---:|---:|---:|---:|
| Development single digits (20) | 16 | **18** | 16 | 15 | 7 | 9 |
| Development numbers, exact (20) | **19** | 18 | 16 | 17 | 5 | 9 |
| Final single digits (40)* | **37** | 36 | **37** | 33 | 24 | 26 |
| Final numbers, exact (20)* | 16 | **17** | 12 | 13 | 2 | 3 |

\*The final collection had already been analysed before this study, so these rows are comparison results only. No model was promoted from them.

## Findings

1. **A tuned kernel SVM on HOG features is the most accurate MNIST classifier we tested (99.11%)**, ahead of every network trained under our fixed 5-epoch, single-seed budget. This does not mean SVMs beat CNNs in general. Published CNNs exceed 99.5% with augmentation and longer training. It does show that our networks are budget-limited, and that a well-chosen classical pipeline remains a strong baseline.
2. **Representation matters more than the classifier on photographs.** Raw-pixel kNN and random forest score 96.9–97.1% on MNIST, but collapse on our photos (2–9 of 20 exact numbers). The same kNN on HOG features recovers to 17/20. Pixel-matching methods assume MNIST's stroke width and intensity profile. Gradient orientation is far less sensitive to the thicker, greyer strokes in phone photographs. CNNs learn comparable local, orientation-sensitive filters, which explains their robustness.
3. **Cost trade-off.** The RBF-SVM is about 90 times slower per digit than the CNN (7.3 ms vs 0.08 ms) and over 200 times larger (89 MB vs 0.4 MB), because prediction compares each input with thousands of stored support vectors. kNN and forests are even larger. The CNN gives near-best accuracy at a tiny cost.
4. **Selection decision: the CNN stays the Numbers default.** Under the project's pre-declared no-regression rule, a candidate must not lose exact matches on either development collection. The RBF-SVM gains on development single digits (18 vs 16) but loses one development number (18 vs 19), so it fails the gate. We report it as the strongest benchmark model, not the deployed one. The post-hoc final figures (36/40 and 17/20 against 37/40 and 16/20) are mixed and would not change this decision.

## Limitations

One split, one run per setting and small grids. SVM hyperparameters (gamma, HOG cell size) were not exhaustively searched. Photo collections are small (20–40 images), so a one- or two-image difference is not statistically meaningful. Timings come from one machine. Classical models were not retrained on EMNIST: an RBF-SVM on 628,000 characters is impractical here, as training scales roughly quadratically with sample count.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe -m research.compare_classical_models   # 15-20 minutes on a laptop CPU; needs data/MNIST
```

Outputs: [`results/classical_baselines/`](../../results/classical_baselines/): `validation_grid.csv`, `benchmark.csv`, `photo_summary.csv`, `photo_predictions.csv` (every photo, every model) and `summary.json` (protocol, confusion matrices, environment). The comparison is also shown in the GUI's Model lab (Compare models tab).

## References

- N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," CVPR 2005.
- C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning* 20, 1995.
- Y. LeCun et al., "Gradient-based learning applied to document recognition," *Proc. IEEE* 86(11), 1998: includes kNN and SVM baselines on MNIST.
- [scikit-learn 1.6 user guide: SVM, nearest neighbours, random forests](https://scikit-learn.org/1.6/user_guide.html)
