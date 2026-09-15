QUALITY_ALIGNMENT="Biased" # Biased / Aligned / Expanded
MODEL=timesformer-k400 # timesformer-{ssv2,k400,scratch} / videomaev2 / demamba / dinov{2,3} / npr / safe / aide
FEATURE_TYPE="video"
DATASET_NAME="RDL"
EXP_NAME="${MODEL}-${QUALITY_ALIGNMENT}"
CKPT_DIR="./results/ckpts/${FEATURE_TYPE}-classifier/${EXP_NAME}/"
CKPT_PATH="${CKPT_DIR}best_auroc_ckpt.pth"

EVAL_ONLY=False

if [ "$QUALITY_ALIGNMENT" = "Biased" ]; then
    TRAIN_REAL_MODEL="Kinetics-400"
    VAL_REAL_MODEL="Kinetics-400"
    DATA_EXPANSION=False
elif [ "$QUALITY_ALIGNMENT" = "Aligned" ]; then
    TRAIN_REAL_MODEL="InternVid-AES"
    VAL_REAL_MODEL="InternVid-AES"
    DATA_EXPANSION=False
elif [ "$QUALITY_ALIGNMENT" = "Expanded" ]; then
    TRAIN_REAL_MODEL="InternVid-AES"
    VAL_REAL_MODEL="InternVid-AES"
    DATA_EXPANSION=True
fi
TRAIN_FAKE_MODEL="Pika"
VAL_FAKE_MODEL="SEINE"

# Train
if [ "$EVAL_ONLY" != "True" ]; then
    python train.py \
        --config-path "configs/${FEATURE_TYPE}-classifier/${MODEL}" \
        --config-name standard.yaml \
        experiment_name="${EXP_NAME}" \
        data.train_real_model="${TRAIN_REAL_MODEL}" \
        data.train_fake_model="${TRAIN_FAKE_MODEL}" \
        data.val_real_model="${VAL_REAL_MODEL}" \
        data.val_fake_model="${VAL_FAKE_MODEL}" \
        data.data_expansion="${DATA_EXPANSION}" \
        log_path="./results/logs/${FEATURE_TYPE}-classifier" \
        save_ckpt_dir=$CKPT_DIR
else
    echo "Skip training, evaluate only..."
fi

# Evaluate
python -W ignore test.py \
    --config-path "configs/${FEATURE_TYPE}-classifier/${MODEL}" \
    --config-name test.yaml \
    experiment_name="${EXP_NAME}" \
    ckpt_path=$CKPT_PATH \
    log_path="./results/test/${FEATURE_TYPE}-classifier" \
    save_csv_file="${DATASET_NAME}.csv"
