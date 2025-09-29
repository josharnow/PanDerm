# Defaults (override with: make linear_eval VAR=value) (DEFAULTS ARE FOR PAD-UFES dataset with PanDerm_Large_LP model)
PYTHON ?= python3
CUDA ?= 1
BATCH_SIZE ?= 1000
MODEL ?= PanDerm_Large_LP
NB_CLASSES ?= 6
PERCENT_DATA ?= 1.0
CSV_FILENAME ?= PanDerm_Large_LP_result.csv
PROJECT_DIR ?= /home/PACE/ja50529n/MS\ Thesis/Model/PanDerm
OUTPUT_DIR ?= $(PROJECT_DIR)/output/PanDerm_Large_LP_res
# OUTPUT_DIR ?= $(PROJECT_DIR)/output/PanDerm_Large_LP_res
CSV_PATH ?= /home/PACE/ja50529n/MS\ Thesis/Thesis\ Data/Skin\ Cancer\ Project/PanDerm\ &\ SkinEHDLF/pad-ufes/2000.csv
ROOT_PATH ?= /home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/pad-ufes/images/
# CSV_PATH ?= $(PROJECT_DIR)/Evaluation_datasets/pad-ufes/2000.csv
# ROOT_PATH ?= $(PROJECT_DIR)/Evaluation_datasets/pad-ufes/images
PRETRAINED_CHECKPOINT ?= /home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth	
NUM_WORKERS ?= 4

# CPU threading limits to reduce contention on laptops
OMP_NUM_THREADS ?= 4
MKL_NUM_THREADS ?= 4
PYTORCH_NUM_THREADS ?= 4
# Common BLAS/OpenMP vars (macOS Accelerate uses VECLIB)
OPENBLAS_NUM_THREADS ?= $(OMP_NUM_THREADS)
VECLIB_MAXIMUM_THREADS ?= $(OMP_NUM_THREADS)
NUMEXPR_NUM_THREADS ?= $(OMP_NUM_THREADS)

# macOS-specific duplicated variables (used by linear_eval targets)
MAC_PYTHON ?= python3
MAC_CUDA ?= 0
MAC_BATCH_SIZE ?= 64
MAC_MODEL ?= PanDerm_Large_LP
MAC_NB_CLASSES ?= 6
MAC_PERCENT_DATA ?= 1.0
MAC_CSV_FILENAME ?= PanDerm_Large_LP_result.csv
MAC_OUTPUT_DIR ?= ../output/PanDerm_Large_LP_res/
MAC_CSV_PATH ?= ../Evaluation_datasets/pad-ufes/2000.csv
MAC_ROOT_PATH ?= ../Evaluation_datasets/pad-ufes/images/
MAC_PRETRAINED_CHECKPOINT ?= ../pretrain_weight/panderm_ll_data6_checkpoint-499.pth
MAC_NUM_WORKERS ?= 0

# macOS thread/env tuning
MAC_OMP_NUM_THREADS ?= 4
MAC_MKL_NUM_THREADS ?= 4
MAC_PYTORCH_NUM_THREADS ?= 4
MAC_OPENBLAS_NUM_THREADS ?= $(MAC_OMP_NUM_THREADS)
MAC_VECLIB_MAXIMUM_THREADS ?= $(MAC_OMP_NUM_THREADS)
MAC_NUMEXPR_NUM_THREADS ?= $(MAC_OMP_NUM_THREADS)

.PHONY: linear_eval_phase_1
linear_eval_phase_1:
	@cd classification && \
	mkdir -p "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/PanDerm_Large_LP_res" && \
	ulimit -n 4096; \
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) VECLIB_MAXIMUM_THREADS=$(VECLIB_MAXIMUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) OMP_NUM_THREADS=$(OMP_NUM_THREADS) MKL_NUM_THREADS=$(MKL_NUM_THREADS) PYTORCH_NUM_THREADS=$(PYTORCH_NUM_THREADS) CUDA_VISIBLE_DEVICES=$(CUDA) $(PYTHON) linear_eval.py \
		--batch_size $(BATCH_SIZE) \
		--model "$(MODEL)" \
		--nb_classes $(NB_CLASSES) \
		--percent_data $(PERCENT_DATA) \
		--csv_filename "$(CSV_FILENAME)" \
		--output_dir "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/PanDerm_Large_LP_res" \
		--csv_path "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/pad-ufes/2000.csv" \
		--root_path "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/pad-ufes/images/" \
		--pretrained_checkpoint "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth" \
		--num_workers $(NUM_WORKERS)

linear_eval:
	@cd classification && \
	mkdir -p "$(OUTPUT_DIR)" && \
	ulimit -n 4096; \
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) VECLIB_MAXIMUM_THREADS=$(VECLIB_MAXIMUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) OMP_NUM_THREADS=$(OMP_NUM_THREADS) MKL_NUM_THREADS=$(MKL_NUM_THREADS) PYTORCH_NUM_THREADS=$(PYTORCH_NUM_THREADS) CUDA_VISIBLE_DEVICES=$(CUDA) $(PYTHON) linear_eval.py \
		--batch_size $(BATCH_SIZE) \
		--model "$(MODEL)" \
		--nb_classes $(NB_CLASSES) \
		--percent_data $(PERCENT_DATA) \
		--csv_filename "$(CSV_FILENAME)" \
		--output_dir "$(OUTPUT_DIR)" \
		--csv_path "$(CSV_PATH)" \
		--root_path "$(ROOT_PATH)" \
		--pretrained_checkpoint "$(PRETRAINED_CHECKPOINT)" \
		--num_workers $(NUM_WORKERS)

.PHONY: mac_linear_eval
mac_linear_eval:
	@cd classification && \
	mkdir -p "$(MAC_OUTPUT_DIR)" && \
	ulimit -n 4096; \
	OPENBLAS_NUM_THREADS=$(MAC_OPENBLAS_NUM_THREADS) VECLIB_MAXIMUM_THREADS=$(MAC_VECLIB_MAXIMUM_THREADS) NUMEXPR_NUM_THREADS=$(MAC_NUMEXPR_NUM_THREADS) OMP_NUM_THREADS=$(MAC_OMP_NUM_THREADS) MKL_NUM_THREADS=$(MAC_MKL_NUM_THREADS) PYTORCH_NUM_THREADS=$(MAC_PYTORCH_NUM_THREADS) CUDA_VISIBLE_DEVICES=$(MAC_CUDA) $(MAC_PYTHON) linear_eval.py \
		--batch_size $(MAC_BATCH_SIZE) \
		--model "$(MAC_MODEL)" \
		--nb_classes $(MAC_NB_CLASSES) \
		--percent_data $(MAC_PERCENT_DATA) \
		--csv_filename "$(MAC_CSV_FILENAME)" \
		--output_dir "$(MAC_OUTPUT_DIR)" \
		--csv_path "$(MAC_CSV_PATH)" \
		--root_path "$(MAC_ROOT_PATH)" \
		--pretrained_checkpoint "$(MAC_PRETRAINED_CHECKPOINT)" \
		--num_workers $(MAC_NUM_WORKERS)

# NOTE - After testing, below takes longer on Mac
# Fast variant (opt-in): higher batch, a few DataLoader workers, and more threads.
FAST_BATCH_SIZE ?= 128
FAST_NUM_WORKERS ?= 2
FAST_OMP_THREADS ?= 6
FAST_MKL_THREADS ?= $(FAST_OMP_THREADS)
FAST_TORCH_THREADS ?= $(FAST_OMP_THREADS)
FAST_OPENBLAS_THREADS ?= $(FAST_OMP_THREADS)
FAST_VECLIB_THREADS ?= $(FAST_OMP_THREADS)
FAST_NUMEXPR_THREADS ?= $(FAST_OMP_THREADS)

# macOS fast variants
MAC_FAST_BATCH_SIZE ?= 128
MAC_FAST_NUM_WORKERS ?= 2
MAC_FAST_OMP_THREADS ?= 6
MAC_FAST_MKL_THREADS ?= $(MAC_FAST_OMP_THREADS)
MAC_FAST_TORCH_THREADS ?= $(MAC_FAST_OMP_THREADS)
MAC_FAST_OPENBLAS_THREADS ?= $(MAC_FAST_OMP_THREADS)
MAC_FAST_VECLIB_THREADS ?= $(MAC_FAST_OMP_THREADS)
MAC_FAST_NUMEXPR_THREADS ?= $(MAC_FAST_OMP_THREADS)

.PHONY: linear_eval_fast
linear_eval_fast:
	@cd classification && \
	mkdir -p "$(OUTPUT_DIR)" && \
	ulimit -n 8192; \
	OPENBLAS_NUM_THREADS=$(FAST_OPENBLAS_THREADS) VECLIB_MAXIMUM_THREADS=$(FAST_VECLIB_THREADS) NUMEXPR_NUM_THREADS=$(FAST_NUMEXPR_THREADS) OMP_NUM_THREADS=$(FAST_OMP_THREADS) MKL_NUM_THREADS=$(FAST_MKL_THREADS) PYTORCH_NUM_THREADS=$(FAST_TORCH_THREADS) CUDA_VISIBLE_DEVICES=$(CUDA) $(PYTHON) linear_eval.py \
		--batch_size $(FAST_BATCH_SIZE) \
		--model "$(MODEL)" \
		--nb_classes $(NB_CLASSES) \
		--percent_data $(PERCENT_DATA) \
		--csv_filename "$(CSV_FILENAME)" \
		--output_dir "$(OUTPUT_DIR)" \
		--csv_path "$(CSV_PATH)" \
		--root_path "$(ROOT_PATH)" \
		--pretrained_checkpoint "$(PRETRAINED_CHECKPOINT)" \
		--num_workers $(FAST_NUM_WORKERS)

.PHONY: mac_linear_eval_fast
mac_linear_eval_fast:
	@cd classification && \
	mkdir -p "$(MAC_OUTPUT_DIR)" && \
	ulimit -n 8192; \
	OPENBLAS_NUM_THREADS=$(MAC_FAST_OPENBLAS_THREADS) VECLIB_MAXIMUM_THREADS=$(MAC_FAST_VECLIB_THREADS) NUMEXPR_NUM_THREADS=$(MAC_FAST_NUMEXPR_THREADS) OMP_NUM_THREADS=$(MAC_FAST_OMP_THREADS) MKL_NUM_THREADS=$(MAC_FAST_MKL_THREADS) PYTORCH_NUM_THREADS=$(MAC_FAST_TORCH_THREADS) CUDA_VISIBLE_DEVICES=$(MAC_CUDA) $(MAC_PYTHON) linear_eval.py \
		--batch_size $(MAC_FAST_BATCH_SIZE) \
		--model "$(MAC_MODEL)" \
		--nb_classes $(MAC_NB_CLASSES) \
		--percent_data $(MAC_PERCENT_DATA) \
		--csv_filename "$(MAC_CSV_FILENAME)" \
		--output_dir "$(MAC_OUTPUT_DIR)" \
		--csv_path "$(MAC_CSV_PATH)" \
		--root_path "$(MAC_ROOT_PATH)" \
		--pretrained_checkpoint "$(MAC_PRETRAINED_CHECKPOINT)" \
		--num_workers $(MAC_FAST_NUM_WORKERS)

.PHONY: preprocess_images_phase_1
preprocess_images_phase_1:
	@cd panderm_modifications/phase_1/utils && \
	$(PYTHON) preprocess_images.py