# Must run before setting up OpenCOOD
module load micromamba
module swap gnu13 gnu12/12.2.0
module load cuda11
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export LDSHARED="$(which gcc) -shared"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(dirname $(gcc -print-file-name=libgomp.so.1)):$LD_LIBRARY_PATH"
export CFLAGS="-I/usr/include -I$CONDA_PREFIX/include"
export CXXFLAGS="-I/usr/include -I$CONDA_PREFIX/include"
export TORCH_CUDA_ARCH_LIST="8.0"
