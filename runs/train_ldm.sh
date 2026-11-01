python ldm.py --data data/pkl \
    --list data/knitcad_data_split.json --option surfpos --gpu 0 \
    --env knitcad_ldm_surfpos --train_nepoch 20000 --test_nepoch 200 --save_nepoch 10 --batch_size 256 \
    --max_face 70 --max_edge 40 --max_contact_face 10 --max_text_length 148 \
    --dir_name weights/cadknitter \
    --finetune --weight weights/abc_ldm_surfpos.pt