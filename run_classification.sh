set -e
MODEL=${1:?usage: bash run_classification.sh lstm|felstm|melstm}
SEED=${SEED:-0}

HIDDEN=32
HEAD_CH=64
HEAD_BLOCKS=3
EPOCHS=50
LR=1e-3
SEQ_LEN=15
IMAGE=36
TRAIN_SAMPLES=""
VAL_SAMPLES=""
TEST_SAMPLES=""
CURVE_EVERY=25
CURVE_SIZE=256
POOL=max

DATA_V=2
BG_MODE=opposite
MOTION=piecewise
CORR_LEN=0.0

V_RANGE=2
N_SLOTS=2
VEL_SRC=bootstrap

BATCH=64

SAVE_DIR=${SAVE_DIR:-./experiments_classification}

python moving_mnist/train_classification.py \
  --model "$MODEL" \
  --hidden_size $HIDDEN --head_channels $HEAD_CH --head_blocks $HEAD_BLOCKS \
  --velocity_pool $POOL --v_range $V_RANGE --num_vel_modes $N_SLOTS \
  --velocity_source $VEL_SRC \
  --seq_len $SEQ_LEN --image_size $IMAGE \
  --data_v_range $DATA_V --bg_mode $BG_MODE \
  --motion_mode $MOTION --corr_len $CORR_LEN \
  --batch_size $BATCH --epochs $EPOCHS --lr $LR \
  ${TRAIN_SAMPLES:+--max_train_samples $TRAIN_SAMPLES} \
  ${VAL_SAMPLES:+--val_size $VAL_SAMPLES} \
  ${TEST_SAMPLES:+--test_size $TEST_SAMPLES} \
  --use_lr_scheduler --early_stop_patience 0 \
  --val_curve_interval $CURVE_EVERY --val_curve_size $CURVE_SIZE \
  --save_dir $SAVE_DIR \
  --data_seed 42 --model_seed $SEED \
  --run_name "cf_cls_${MODEL}_s${SEED}" \
  "${@:2}"
