python vis_paired_step.py render-video \
    data/eval_gt/f018083_cond.step \
    data/eval_gt/f018083_gt.step \
    --fps 24 --duration 5 --resolution 1024 \
    --camera-height-low -1.0 \
    --camera-height 1.0 \
    --camera-distance 2.5 \
    --output-folder vis \
    --color-a lightblue --color-b lightorange