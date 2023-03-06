#!/bin/bash
export TRANSFORMERS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/
export HF_DATASETS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/datasets
#PLM_MODEL='bert-base-uncased SpanBERT/spanbert-base-cased roberta-base nghuyong/ernie-2.0-en'
#PLM_MODEL='bert-large-uncased SpanBERT/spanbert-large-cased roberta-large nghuyong/ernie-2.0-large-en'

PLM_MODEL='nghuyong/ernie-2.0-large-en'
MLM_MODE='triple'
GPU=3

for MODEL in $PLM_MODEL;
do
for MODE in $MLM_MODE;
do

CUDA_VISIBLE_DEVICES=$GPU /nas-alitranx/yongjiang.jy/jiangchy/anaconda3/envs/jiangchy-pytorch/bin/python pretrain.py \
--mlm_mode $MODE --plm_model $MODEL --bz 16

done
done