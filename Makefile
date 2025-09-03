# Defaults (override with: make linear_eval VAR=value)
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

.PHONY: linear_eval
linear_eval:
	@cd classification && \
	mkdir -p "$(OUTPUT_DIR)" && \
	ulimit -n 4096; \
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) MKL_NUM_THREADS=$(MKL_NUM_THREADS) PYTORCH_NUM_THREADS=$(PYTORCH_NUM_THREADS) CUDA_VISIBLE_DEVICES=$(CUDA) $(PYTHON) linear_eval.py \
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