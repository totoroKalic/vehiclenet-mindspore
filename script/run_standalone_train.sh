echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DEVICE_ID"
echo "For example: bash run_standalone_train.sh device_id"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

set -e

DEVICE_ID=$1
export DEVICE_ID

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf single_train
mkdir single_train
cd ./single_train
mkdir src
cd ../
cp ./train.py ./single_train
cp ./src/*.py ./single_train/src
cd ./single_train

env > env0.log

echo "Standalone train begin."
python train.py --run_distribute False --device_id $1 --device_num 1 --is_modelarts False > ./train_alone.log 2>&1 &

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../