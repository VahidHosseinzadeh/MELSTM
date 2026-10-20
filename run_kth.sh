set -e
MODE=${1:?usage: bash run_kth.sh lstm|felstm|melstm [seed] [extra train_kth args] | bash run_kth.sh sweep}
SEED=${2:-1}
SAVE_DIR=${SAVE_DIR:-./experiments_kth}

if [ "$MODE" = sweep ]; then
  RUNS=$(ls "$SAVE_DIR"/models | sed -n 's/_best\.pth$//p')
  python -m kth.sweep_kth --runs $RUNS --save_dir "$SAVE_DIR" --out "$SAVE_DIR/sweep"
  python -m kth.motion_sweep_kth --runs $RUNS --save_dir "$SAVE_DIR" --out "$SAVE_DIR/motion_sweep"
  python -m kth.sweep_summary "$SAVE_DIR/sweep" --out "$SAVE_DIR/summary"
  exit 0
fi

case $MODE in
  lstm|felstm) EXTRA=() ;;
  melstm) EXTRA=(--velocity_source tracked --x_curriculum_epochs 200 --pc_search_radius 0) ;;
  *) echo "unknown model: $MODE"; exit 1 ;;
esac

python -m kth.train_kth --model "$MODE" --model_seed "$SEED" \
  --run_name "kth_${MODE}${TAG:+_$TAG}_s${SEED}" --save_dir "$SAVE_DIR" \
  --camera shake --shake_amp 0 2 --shake_period 16 32 --shake_vmax 0 \
  "${EXTRA[@]}" "${@:3}"
