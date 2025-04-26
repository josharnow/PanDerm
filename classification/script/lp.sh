# For PanDerm-Large: test on HAM_clean split
CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Large" \
  --nb_classes 7 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Large_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM_clean.csv" \
  --root_path "/home/share/Uni_Eval/ISIC2018_reader/images/" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

# For PanDerm-Large: test on HAM_derm1m split
CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Large" \
  --nb_classes 7 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Large_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM-official-7-lp.csv" \
  --root_path "/home/share/Uni_Eval/ISIC2018_reader/images/" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Large" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/f17k-113-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Large" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/pad-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Large" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/daffodil-5-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

# ---------------------------------------------------------------------------------------------------------
# For PanDerm-Base: test on HAM_iccv split
CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 7 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM-official-7-lp.csv" \
  --root_path "/home/share/Uni_Eval/HAM10000_clean/ISIC2018/" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/f17k-113-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/pad-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/daffodil-5-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/sd-128-lp.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm-Base" \
  --nb_classes 113 \
  --percent_data 1.0 \
  --csv_filename "PanDerm-Base_result.csv" \
  --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/PanDerm-Large_res/" \
  --csv_path "/home/syyan/XJ/PanDerm-open_source/data/linear_probing/bcn-lp-ws0.csv" \
  --root_path "" \
  --pretrained_checkpoint "/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_bb_data6_checkpoint-499.pth"