pip install scipy
pip install Pillow
pip install imageio
pip install matplotlib

# TRAIN=$true
DATASET_PATH=$1
BATCH_SIZE=$2
POINT_NUM=$3
CKPT_PATH=$4
CKPT_LOAD=$5
RESULT_PATH=$6
EPOCHS=$7
SAVE_AT_EPOCH=$8


python train_test.py \
    --train="True" \
    --dataset_path ${DATASET_PATH} \
    --batch_size ${BATCH_SIZE} \
    --point_num ${POINT_NUM} \
    --ckpt_path ${CKPT_PATH} \
    --ckpt_load ${CKPT_LOAD} \
    --result_path ${RESULT_PATH} \
    --epochs ${EPOCHS} \
    --save_at_epoch ${SAVE_AT_EPOCH} \
    --gpu 1
    
