#!/usr/bin/env bash -l
set -e

env_name="advCP"

echo "########################## Conda Env ##########################"
if conda env list | grep -qE "^$env_name\s"; then
    echo "Conda environment '$env_name' exists."
else
    echo "Conda environment '$env_name' does not exist. creating ..."
    conda env create --name $env_name --file environment.yml
fi

echo "Installing pytorch, cuda, and related packages ..."
conda install -y --name $env_name pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=10.2 -c pytorch
conda run --live-stream -n $env_name pip install spconv-cu102

echo "########################## Dependency Setup ##########################"
echo "Setting up IoU cuda operator ..."
cd mvp/perception/cuda_op 
conda run --live-stream -n $env_name python setup.py install
cd -

# Must run before setting up OpenCOOD
# export CC=$(which gcc)
# export CXX=$(which g++)
# export CUDAHOSTCXX=$(which g++)
# export LDSHARED="$(which gcc) -shared"
# unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE LIBRARY_PATH
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="$(dirname $(gcc -print-file-name=libgomp.so.1)):$LD_LIBRARY_PATH"
# export CFLAGS="-I/usr/include -I$CONDA_PREFIX/include"
# export CXXFLAGS="-I/usr/include -I$CONDA_PREFIX/include"
# export TORCH_CUDA_ARCH_LIST="8.0"

echo "Setting up OpenCOOD ..."
cd third_party/OpenCOOD
conda run --live-stream -n $env_name python opencood/utils/setup.py build_ext --inplace
conda run --live-stream -n $env_name python opencood/pcdet_utils/setup.py build_ext --inplace
cd -

echo "The environment is set and you should be able to run the evaluation. Commands"
echo "    conda activate advCP"
echo "    python scripts/evaluate.py"
