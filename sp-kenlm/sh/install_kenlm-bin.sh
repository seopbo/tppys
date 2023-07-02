# install kenlm
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev

SCRIPT_PATH=$BASH_SOURCE
cd $(dirname $SCRIPT_PATH)/..

# If you want to train language model by kenlm, you will build kenlm
if [ -d tmp/kenlm ]; then
    echo "kenlm is already built."
else
    echo "build kenlm"
    git clone https://github.com/kpu/kenlm tmp/kenlm
    cd tmp/kenlm

    mkdir -p build
    cd build
    cmake ..
    make -j 4
fi