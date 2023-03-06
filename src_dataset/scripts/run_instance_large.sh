#!/bin/bash
export TRANSFORMERS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/
export HF_DATASETS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/datasets
DATA_DIR=../dataset_construction/wiki20m
SAVE_DIR=save/$(date '+%Y-%m-%d-%H-%M')
GPU=3
MODE="r"
#MODE="hi,ti"
echo "SAVE DIR: "$SAVE_DIR

list_layer="2,6,10,12,14,16,18,20,22,24"
list_std="1"
#list_sep="sentence none sep split"
list_sep="sentence none"

list_rep="max"
#list_model='bert-large-uncased SpanBERT/spanbert-large-cased'
list_model='roberta-large nghuyong/ernie-2.0-large-en bert-large-uncased SpanBERT/spanbert-large-cased'

for MODEL in $list_model;
do
for STD in $list_std;
do
for SEP in $list_sep;
do
for REP in $list_rep;
do

echo "========== MODEL" $MODEL " LAYER " $LAYER " STD " $STD " SEP " $SEP "=============="
CUDA_VISIBLE_DEVICES=$GPU ~/jiangchy/anaconda3/envs/jiangchy-pytorch/bin/python main.py \
--standardize $STD --embedding contextual --plm_layer $list_layer --sep_strategy $SEP --mode $MODE \
--save_dir $SAVE_DIR --plm_model $MODEL --max_len 64 --rep_strategy $REP --encode_bz 32 --pca 0

done
done
done
done

mkdir  $DATA_DIR/$SAVE_DIR
cp scripts/run_instance_large.sh $DATA_DIR/$SAVE_DIR