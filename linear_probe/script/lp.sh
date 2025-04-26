models=('PanDerm-Large' 'PanDerm-Base' 'SwAVDerm'  'dinov2' 'imgnet_large21k')
checkpoints=('/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm-large.pth' \
             '/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm-base.pth' \
             '/home/share/FM_Code/PanDerm/Model_Weights/swavderm_pretrained.pth' \
             '' \
             '')
#models=('imgnet_large21k')

if [ ${#models[@]} -ne ${#checkpoints[@]} ]; then
  echo "Error: models and checkpoints arrays must have the same length"
  exit 1
fi

for i in "${!models[@]}"; do
  model="${models[$i]}"
  checkpoint="${checkpoints[$i]}"
  csv_file="${model}_Result.csv"
  
  for seed in 0; do
    CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 7 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path "/home/share/Uni_Eval/HAM10000_clean/ISIC2018_splits/HAM_clean.csv" \
      --root_path "/home/share/Uni_Eval/HAM10000_clean/ISIC2018/" \
      --checkpoint "$checkpoint"
      
    CUDA_VISIBLE_DEVICES=0 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 16 \
      --percent_data 1 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path '/home/share/Uni_Eval/WSI_patch_data/patch.csv' \
      --root_path '/home/share/Uni_Eval/WSI_patch_data/images/' \
      --checkpoint "$checkpoint"
      
    CUDA_VISIBLE_DEVICES=0 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path '/home/share/Uni_Eval/MSKCC/mskcc.csv' \
      --root_path '/home/share/Uni_Eval/MSKCC/images/' \
      --checkpoint "$checkpoint"
      
    CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 9 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path '/home/share/Uni_Eval/BCN20000/bcn20000.csv' \
      --root_path '/home/share/Uni_Eval/BCN20000/images/' \
      --checkpoint "$checkpoint"

    CUDA_VISIBLE_DEVICES=2 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path '/home/share/Uni_Eval/HIBA/hiba.csv' \
      --root_path '/home/share/Uni_Eval/HIBA/images/' \
      --checkpoint "$checkpoint"

    CUDA_VISIBLE_DEVICES=2 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 6 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path "/home/share/Uni_Eval/pad-ufes/2000.csv" \
      --root_path "/home/share/Uni_Eval/pad-ufes/images/" \
      --checkpoint "$checkpoint"
      
    CUDA_VISIBLE_DEVICES=3 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path "/home/share/Uni_Eval/Derm7pt/atlas-clinical-all.csv" \
      --root_path "/home/share/Uni_Eval/Derm7pt/images/" \
      --checkpoint "$checkpoint"
      
    CUDA_VISIBLE_DEVICES=3 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/${model}_res/" \
      --csv_path '/home/share/Uni_Eval/DDI/ddi_clean.csv' \
      --root_path '/home/share/Uni_Eval/DDI/images/' \
      --checkpoint "$checkpoint"

    CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
      --batch_size 1000 \
      --model "$model" \
      --nb_classes 23 \
      --percent_data 1.0 \
      --csv_filename "$csv_file" \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/task2/${model}_res/" \
      --csv_path "/home/share/Uni_Eval/Dermnet/dermnet.csv" \
      --root_path "/home/share/Uni_Eval/Dermnet/images/" \
      --checkpoint "$checkpoint"
  done
  wait
done
