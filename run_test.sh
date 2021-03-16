pip install scipy
pip install Pillow
pip install imageio
pip install matplotlib

# TRAIN=$false
POINT_NUM=$1
CKPT_PATH=$2
CKPT_LOAD=$3
SAVE_IMAGES=$4
SAVE_PTS_FILES=$5
SEED=$6


python train_test.py \
    --train="False" \
    --point_num ${POINT_NUM} \
    --ckpt_path ${CKPT_PATH} \
    --ckpt_load ${CKPT_LOAD} \
    --save_images ${SAVE_IMAGES} \
    --save_pts_files ${SAVE_PTS_FILES} \
    --seed ${SEED} \
    --gpu 1
