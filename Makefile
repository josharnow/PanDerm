# Defaults (override with: make linear_eval VAR=value) (DEFAULTS ARE FOR PAD-UFES dataset with PanDerm_Large_LP model)
PYTHON ?= python3
CUDA ?= 0
# Smaller batch to avoid CPU OOM/hangs on macOS
BATCH_SIZE ?= 64
MODEL ?= PanDerm_Large_LP
NB_CLASSES ?= 6
PERCENT_DATA ?= 1.0
CSV_FILENAME ?= PanDerm_Large_LP_result.csv
OUTPUT_DIR ?= ../output/PanDerm_Large_LP_res/
CSV_PATH ?= ../Evaluation_datasets/pad-ufes/2000.csv
ROOT_PATH ?= ../Evaluation_datasets/pad-ufes/images/
PRETRAINED_CHECKPOINT ?= ../pretrain_weight/panderm_ll_data6_checkpoint-499.pth
NUM_WORKERS ?= 0

# CPU threading limits to reduce contention on laptops
OMP_NUM_THREADS ?= 4
MKL_NUM_THREADS ?= 4
PYTORCH_NUM_THREADS ?= 4
# Common BLAS/OpenMP vars (macOS Accelerate uses VECLIB)
OPENBLAS_NUM_THREADS ?= $(OMP_NUM_THREADS)
VECLIB_MAXIMUM_THREADS ?= $(OMP_NUM_THREADS)
NUMEXPR_NUM_THREADS ?= $(OMP_NUM_THREADS)

.PHONY: linear_eval
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

.PHONY: preprocess_images_phase_1
preprocess_images_phase_1:
	@cd panderm_modifications/phase_1/utils && \
	$(PYTHON) preprocess_images.py