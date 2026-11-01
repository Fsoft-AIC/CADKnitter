python eval/eval.py --generated_dir samples \
                    --ground_truth_dir data/eval_gt \
                    --joint_set_pkl_dir data/pkl \
                    --output_dir eval/eval_results \
                    --test_list data/knitcad_data_split.json \
                    --metrics contact,chamfer,collision \
                    --num_u 32 --num_v 32 \
                    --override