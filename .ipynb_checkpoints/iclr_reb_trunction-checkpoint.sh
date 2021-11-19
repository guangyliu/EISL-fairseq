for i in 2 3 4 5 6 7 8 9 10 all
do
base_noise="shuffle"
preix=$base_noise
sym=$base_noise"_iclr_reb_trunc"
cuda=3
loss="cross_entropy_truncation" # cross_entropy
model="bart_base" # "bart_base_iter"
tf_ratio=1.0 #'1.0'
noise="$preix$i"
data="multi30k-bin/$noise"
max_epoch=20
name=${loss:0:2}"_"${noise}"_"${model:0:2}"_"$tf_ratio"_"$sym
TOTAL_NUM_UPDATES=10000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=6000
UPDATE_FREQ=4
# BART_PATH=/home/lptang/fairseq/checkpoints/510_bart/cr_shuffle5_ba_1.0_s5_ce/checkpoint_best.pt

BART_PATH=/home/lptang/fairseq/examples/bart/pretrained/bart.base/model.pt
# BART_PATH="/home/lptang/fairseq/checkpoints/denoising_bart/$base_noise/cr_"$noise"_ba_1.0_cetf_"$i"/checkpoint_best.pt"
BATCH_SIZE=128
if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "crete folder log/out/"$preix
fi
mkdir log/tf/denoising_bart/$preix
CUDA_VISIBLE_DEVICES=$cuda  fairseq-train multi30k-bin/$noise \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang de --target-lang en \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch $model \
    --criterion $loss \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES\
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters\
    --batch-size $BATCH_SIZE\
    --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses     --eval-bleu-remove-bpe     --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric\
    --save-dir checkpoints/denoising_bart/$preix/$name \
    --tf-ratio $tf_ratio --tensorboard-logdir log/tf/denoising_bart/$preix/$name \
     --max-epoch $max_epoch --no-epoch-checkpoints   --patience 3 #--eval-bleu-print-samples  --log-interval 5
     
done

