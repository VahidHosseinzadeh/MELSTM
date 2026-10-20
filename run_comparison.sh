set -e
MODEL=${1:?usage: bash run_comparison.sh lstm|felstm|melstm [model_seed] [arm]}
MODEL_SEED=${2:-42}
ARM=${3:-base}

COS0=(--lr_schedule cosine --warmup_epochs 0 --final_lr 1e-5)
NEW=(--no_detach --len_gen_at_end)
XCUR=(--x_curriculum_epochs 25 --x_curriculum_start 15)
XSTO=(--x_curriculum_epochs 25 --x_curriculum_mode stochastic)
case $ARM in
  base)      SCHED=(--use_lr_scheduler); OPT=() ;;
  basend)    SCHED=(--use_lr_scheduler); OPT=(--no_detach) ;;
  cos)       SCHED=("${COS0[@]}"); OPT=("${NEW[@]}") ;;
  cosleaky)  SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" --decoder_act leaky) ;;
  xcur)      SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XCUR[@]}") ;;
  xcurfrz)   SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XCUR[@]}"
                                        --frozen_prob_max 0.5 --frozen_start_epoch 26
                                        --frozen_ramp_epochs 10) ;;
  xcurleaky) SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XCUR[@]}" --decoder_act leaky) ;;
  xsto)      SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XSTO[@]}") ;;
  xstocos)   SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XSTO[@]}" --x_curriculum_shape cosine) ;;
  xstot6)    SCHED=("${COS0[@]}"); OPT=("${NEW[@]}" "${XSTO[@]}" --x_curriculum_width 6) ;;
  *) echo "unknown arm: $ARM (base|basend|cos|cosleaky|xcur|xcurfrz|xcurleaky|xsto|xstocos|xstot6)"; exit 1 ;;
esac
case $ARM in
  xcur*|xsto*) if [ "$MODEL" != melstm ]; then echo "arm $ARM is MELSTM-only"; exit 1; fi ;;
esac

HIDDEN=32

BATCH=32
DEC_LAYERS=1
DEC_HIDDEN=32
SEQ_LEN=25
INPUT_FRAMES=15
GEN_INPUT=15
GEN_SEQ_LEN=100
IMAGE=36
EPOCHS=50
MIN_EPOCHS=40
EARLY_STOP_PATIENCE=0
SEED=42
INIT_SEED=${INIT_SEED:-}
INIT_SCHEME=${INIT_SCHEME:-}
INIT_TAG=${INIT_SEED:+_init${INIT_SEED}}${INIT_SCHEME:+_${INIT_SCHEME}}
SAVE_DIR=./experiments_ms${MODEL_SEED}_${ARM}${INIT_TAG}
if [ -n "$SMOKE" ]; then
  SAVE_DIR=./smoke/ms${MODEL_SEED}_${ARM}
  EPOCHS=2
  MIN_EPOCHS=1
  export WANDB_MODE=offline
fi

MOTION_MODE=piecewise
TRANSITION_MODE=smooth
DATA_V_RANGE=2
MIN_SEGMENT=3
MAX_SEGMENT=6
P_CHANGE=0.25
SMOOTH_PROB=0.8

MOTION=(--motion_mode "$MOTION_MODE" --data_v_range "$DATA_V_RANGE")
case $MOTION_MODE in
  constant)
    MOTION_TAG="const" ;;
  piecewise)
    MOTION+=(--transition_mode "$TRANSITION_MODE"
             --min_segment "$MIN_SEGMENT" --max_segment "$MAX_SEGMENT")
    MOTION_TAG="pw${MIN_SEGMENT}-${MAX_SEGMENT}${TRANSITION_MODE:0:1}" ;;
  stochastic)
    MOTION+=(--transition_mode "$TRANSITION_MODE" --p_change "$P_CHANGE")
    MOTION_TAG="st${P_CHANGE}${TRANSITION_MODE:0:1}" ;;
  accelerate)
    MOTION+=(--min_segment "$MIN_SEGMENT" --max_segment "$MAX_SEGMENT")
    MOTION_TAG="acc${MIN_SEGMENT}-${MAX_SEGMENT}" ;;
  *)
    echo "unknown MOTION_MODE: $MOTION_MODE (constant|piecewise|stochastic|accelerate)"; exit 1 ;;
esac
if [ "$MOTION_MODE" = piecewise ] || [ "$MOTION_MODE" = stochastic ]; then
  if [ "$TRANSITION_MODE" = smooth ]; then
    MOTION+=(--smooth_probability "$SMOOTH_PROB")
  fi
fi
MOTION_TAG="v${DATA_V_RANGE}${MOTION_TAG}"

FE_V_RANGE=$DATA_V_RANGE

COMMON=(
  "${MOTION[@]}"
  --hidden_size "$HIDDEN"
  --decoder_hidden_size "$DEC_HIDDEN"
  --decoder_conv_layers "$DEC_LAYERS"
  --batch_size "$BATCH"
  --grad_clip 1.0
  --data_seed "$SEED"
  --model_seed "$MODEL_SEED"
  --image_size "$IMAGE"
  --seq_len "$SEQ_LEN"
  --input_frames "$INPUT_FRAMES"
  --gen_input_frames "$GEN_INPUT"
  --gen_seq_len "$GEN_SEQ_LEN"
  --lr 1e-3
  "${SCHED[@]}"
  --epochs "$EPOCHS"
  --min_epochs "$MIN_EPOCHS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --num_workers 4
  --check_velocity_predictor
  --len_gen_every 2
  --model_save_dir "$SAVE_DIR"
  --wandb_project FEConvLSTM
  "${OPT[@]}"
)
if [ -n "$INIT_SEED" ]; then
  COMMON+=(--init_seed "$INIT_SEED")
fi
if [ -n "$INIT_SCHEME" ]; then
  COMMON+=(--init_scheme "$INIT_SCHEME")
fi
if [ -n "$EXTRA_ARGS" ]; then
  COMMON+=(${EXTRA_ARGS})
fi
if [ -n "$SMOKE" ]; then
  COMMON+=(--max_train_samples 256 ${SMOKE_EXTRA:-})
fi

RUN_TAG="h${HIDDEN}_${MOTION_TAG}_s${SEED}_ms${MODEL_SEED}_${ARM}${INIT_TAG}"
case $MODEL in
  lstm)
    EXTRA=(--model lstm --v_range 0 --wandb_name "lstm_${RUN_TAG}") ;;
  felstm)
    EXTRA=(--model felstm --v_range "$FE_V_RANGE" --show_h_state --wandb_name "felstm_${RUN_TAG}") ;;
  melstm)
    EXTRA=(--model melstm --num_vel_modes 2 --eval_velocity_mode both --wandb_name "melstm_${RUN_TAG}") ;;
  *)
    echo "unknown model: $MODEL"; exit 1 ;;
esac

MOTION_STAMP="$SAVE_DIR/run_state/motion_${MODEL}.cfg"
MOTION_CFG="${MOTION[*]}"

CKPT=$(ls -t "$SAVE_DIR"/run_state/checkpoint_${MODEL}_*.pth 2>/dev/null | head -1)
RESUME=()
if [ -n "$CKPT" ] && [ -f "$MOTION_STAMP" ] && [ "$(cat "$MOTION_STAMP")" != "$MOTION_CFG" ]; then
  echo ">>> Motion config CHANGED since $CKPT was written:"
  echo "      checkpoint: $(cat "$MOTION_STAMP")"
  echo "      requested : $MOTION_CFG"
  echo ">>> NOT resuming — starting a fresh run so the weights match the data."
  echo "    (old checkpoints are left in place; rm $SAVE_DIR/run_state/checkpoint_${MODEL}_*.pth to clean up)"
  CKPT=""
fi

mkdir -p "$SAVE_DIR/run_state"
printf '%s\n' "$MOTION_CFG" > "$MOTION_STAMP"

if [ -n "$CKPT" ]; then
  echo ">>> Found checkpoint $CKPT — resuming this run."
  RESUME=(--resume "$CKPT")
  rm -f "$SAVE_DIR"/run_state/DONE_${MODEL}_*.flag
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> commit $(git rev-parse --short HEAD 2>/dev/null)  model=$MODEL  data_seed=$SEED  model_seed=$MODEL_SEED  init_seed=${INIT_SEED:-$MODEL_SEED}  init_scheme=${INIT_SCHEME:-default}  arm=$ARM  save_dir=$SAVE_DIR"

python moving_mnist/train.py "${COMMON[@]}" "${EXTRA[@]}" "${RESUME[@]}"
