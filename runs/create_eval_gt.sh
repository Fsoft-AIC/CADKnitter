python eval/create_eval_gt.py \
    --test_list data/knitcad_data_split_70faces.json \
    --split test \
    --start_idx 0 \
    --end_idx 10 \
    --data_dir data/pkl \
    --output_folder data/eval_gt/