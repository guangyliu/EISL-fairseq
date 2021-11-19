#!/bin/sh
cuda=0
batch_size="256"
# data: iwslt14.tokenized.de-en    multi30k_de_en
data="iwslt14.tokenized.de-en"
# loss: cross_entropy  bleuloss
loss="bleuloss"
# model: transformer_iwslt_de_en   lstm_wiseman_iwslt_de_en_gumbel
model="lstm_wiseman_iwslt_de_en_gumbel"
# restore-file if use pretrained model
pre=1
# sample 
sample="greedy"
# lr
lr="1e-3"
tf_ratio="1.0"
max_epoch=20
sym="tf10"
preix="tf10_debug"
lr_sch="fixed"
# warmup=1000
restore_file="checkpoints/base/bl_iw_ls_greedy_tf10/checkpoint5.pt"

if [ ! -d "log/out/"$preix ]; then
  mkdir "log/out/"$preix
  echo "crete folder log/out/"$preix
fi

if [ $pre == 1 ]
then 
    name="pre_"${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym"_1"
    CUDA_VISIBLE_DEVICES=$cuda  fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler $lr_sch   --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
    --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --restore-file $restore_file --reset-optimizer --sample-method $sample --tf-ratio $tf_ratio \
    --save-interval 1  --clip-norm 0.1 --max-epoch $max_epoch
else
    name=${loss:0:2}"_"${data:0:2}"_"${model:0:2}"_"$sample"_"$sym
    CUDA_VISIBLE_DEVICES=$cuda  fairseq-train    data-bin/$data     --arch $model --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr $lr --lr-scheduler $lr_sch  --weight-decay 0.0001     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses     --eval-bleu-remove-bpe   --eval-bleu-print-samples  --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric    --batch-size $batch_size  --criterion $loss --save-dir checkpoints/$preix/$name  \
    --log-interval 20 --tensorboard-logdir log/tf/$preix/$name --sample-method $sample --tf-ratio $tf_ratio \
    --save-interval 1 --clip-norm 0.1 --max-epoch $max_epoch
fi


