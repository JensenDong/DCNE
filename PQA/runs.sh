#!/bin/sh

python -u -c 'import torch; print(torch.__version__); print(torch.cuda.device_count())'

SETTING=$1

CODE_PATH=$SETTING/codes
DATA_PATH=data
SAVE_PATH=$SETTING/models

#The first four parameters must be provided
MODE=$2
MODEL=$3
DATASET=$4
GPU_DEVICE=$5
SAVE_ID=$6

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$7
NEGATIVE_SAMPLE_SIZE=$8
HIDDEN_DIM=$9
GAMMA=${10}
ALPHA=${11}
LEARNING_RATE=${12}
MAX_STEPS=${13}
REG=${14}
P=${15}

if [ $MODE == "train" ]
then

echo "Start Training......"


CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_train \
            --do_valid \
            --do_test \
            --data_path $FULL_DATA_PATH \
            --model $MODEL \
            -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
            -g $GAMMA -a $ALPHA \
            -lr $LEARNING_RATE --max_steps $MAX_STEPS \
            -save $SAVE \
			-reg $REG --p $P  \
			${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_valid -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_test -init $SAVE

else
   echo "Unknown MODE" $MODE
fi