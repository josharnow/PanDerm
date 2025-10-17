## Josh's Modifications to PanDerm
This directory contains modifications to the PanDerm model broken into different phases of the thesis project for clarity.

### Phase 1: SLICE-3D Dataset Training/Evaluation
1. Preprocess the SLICE-3D dataset using the SkinEHDLF algorithm.
2. Prepare the preprocessed data for training with PanDerm (i.e., convert to CSV format as described by PanDerm authors).
3. Train/evaluate the PanDerm model on the preprocessed SLICE-3D dataset using linear evaluation & compare to SkinEHDLF results, as well as the published PanDerm results using all datasets.

- 

### Phase 2: PanDerm Model Modifications & Training/Evaluation
1. Modify the PanDerm model architecture to use SkinEHDLF components (e.g., the feature extractor) (possibly start with Swan Transformer? Might replace all components instead).
2. Train/evaluate the modified PanDerm model on the SLICE-3D dataset & compare to Phase 1 results.

### Phase 3: Training/Evaluation of Modified PanDerm Model on Augmented Datasets
1. Augment the preprocessed SLICE-3D dataset with additional available datasets used by PanDerm (e.g., HAM10000, ISIC 2019, etc.).
2. Train/evaluate the modified PanDerm model on the augmented dataset & compare to Phase 2 results.
