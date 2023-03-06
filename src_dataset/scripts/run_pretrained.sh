#!/bin/bash
export TRANSFORMERS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/
export HF_DATASETS_CACHE=/nas-alitranx/yongjiang.jy/jiangchy/.cache/huggingface/datasets
DATA_DIR=../dataset_construction/wiki20m
SAVE_DIR=save/$(date '+%Y-%m-%d-%H-%M')
MODE="h,r,t,hi,ti"
#MODE="hi,ti"

#MODE="r"


echo "SAVE DIR: "$SAVE_DIR

# ROBERTA PRETRIANED
#MODEL_DIR=bert-base-uncased-triple-1227122419_1640579059.245812
#MODEL_DIR=roberta-base-subword-1227135948_1640584788.2375975
#MODEL_DIR=bert-base-uncased-subword-1227115911_1640577551.1496556
#MODEL_DIR=nghuyong/ernie-2.0-en-triple-1227140611_1640585171.3866405
#MODEL_DIR=roberta-base-triple-1227143020_1640586620.8609316
#MODEL_DIR=SpanBERT/spanbert-base-cased-triple-1227131459_1640582099.6196055

#MODEL_DIR=bert-large-uncased-triple-0112162746_1641976066.023168
#MODEL_DIR=roberta-large-triple-0112191501_1641986101.9192762
#MODEL_DIR=nghuyong/ernie-2.0-large-en-triple-0112204200_1641991320.82681

GPU=3

#MODEL_DIR=roberta-base-triple-0622174000_1655890800.480712
#MODEL_DIR=roberta-base-subword-0622173827_1655890707.2087777
#MODEL_DIR=bert-base-uncased-triple-0622163727_1655887047.5361278
#MODEL_DIR=bert-base-uncased-subword-0622163748_1655887068.0773106
#MODEL_DIR=nghuyong/ernie-2.0-en-triple-0622181425_1655892865.458864
#MODEL_DIR=nghuyong/ernie-2.0-en-subword-0622181152_1655892712.228775
#MODEL_DIR=SpanBERT/spanbert-base-cased-subword-0622170814_1655888894.8568664
#MODEL_DIR=SpanBERT/spanbert-base-cased-triple-0622170845_1655888925.0427754

#MODEL_DIR=roberta-large-triple-1009161447_1665332087.0228014
#MODEL_DIR=bert-large-uncased-triple-1009133704_1665322624.1819515
#MODEL_DIR=nghuyong/ernie-2.0-large-en-triple-0112204200_1641991320.82681
MODEL_DIR=SpanBERT/spanbert-large-cased-triple-1009145517_1665327317.497042

#list_layer="0 2 4 6 8 10 12 14 16 18 20 22 24"
#list_layer="0,2,6,10,12"
list_layer="0,4,12,20,24"
list_std="1"
list_rep="mean max"
list_sep="sentence none sep"
#list_sep="sentence none sep split"
list_model=($DATA_DIR/pretrain/$MODEL_DIR/ckpt*)
echo ${list_model[@]};

for MODEL in ${list_model[@]};
do
for STD in $list_std;
do
for LAYER in $list_layer;
do
for SEP in $list_sep;
do
for REP in $list_rep;
do

echo "========== MODEL" $MODEL " LAYER " $LAYER " STD " $STD " SEP " $SEP " REP " $REP "=============="
CUDA_VISIBLE_DEVICES=$GPU ~/jiangchy/anaconda3/envs/jiangchy-pytorch/bin/python main.py --standardize $STD --embedding contextual --plm_layer $LAYER \
--sep_strategy $SEP --mode $MODE --save_dir $SAVE_DIR --plm_model $MODEL --pca 0 --rep_strategy $REP

done
done
done
done
done

mkdir  $DATA_DIR/$SAVE_DIR
cp scripts/run_pretrained.sh $DATA_DIR/$SAVE_DIR


