# NSG-VD is only evaluated under the Biased setting in our paper.
# Note: NSG-VD uses its own entry points (train_nsgvd.py / test_nsgvd.py) and configs.

TASK_TYPE="standard"   # Biased setting == standard task type
GENERATOR="Pika"
VARIANT="d"            # d: mmd-d (is_yy_zero=False) / mp: mmd-mp (is_yy_zero=True)

EXP_NAME="${TASK_TYPE}-${GENERATOR}-${VARIANT}"
CKPT_PATH="./results/ckpts/nsg-vd/${EXP_NAME}/best_ckpt.pth"

EVAL_ONLY=False

if [ "$VARIANT" = "mp" ]; then
    IS_YY_ZERO=True
else
    IS_YY_ZERO=False
fi

# Train
if [ "$EVAL_ONLY" != "True" ]; then
    python train_nsgvd.py \
        --config-path "configs/nsg-vd-224x224" \
        --config-name standard.yaml \
        experiment_name="${EXP_NAME}" \
        model.is_yy_zero=${IS_YY_ZERO}
else
    echo "Skip training, evaluate only..."
fi

# Evaluate on all test sets
python -W ignore test_nsgvd.py \
    --config-path "configs/nsg-vd-224x224" \
    --config-name test.yaml \
    experiment_name="${EXP_NAME}" \
    model.is_yy_zero=${IS_YY_ZERO} \
    ckpt_path="${CKPT_PATH}" \
    log_path="./results/test/nsg-vd/${EXP_NAME}"
