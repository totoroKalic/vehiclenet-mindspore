
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_SIZE"
echo "For example: bash run_distribute.sh 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

RANK_SIZE=$1
export RANK_SIZE

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf train
mkdir train
cd ./train
for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../train.py ./device$i
    cp ../src/*.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --run_distribute True --device_num $1 --is_modelarts False --eval_training True > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../