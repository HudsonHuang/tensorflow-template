#prepare environment and run default experiment

source activate tensorflow
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
pip install -U pip
#conda install pytorch torchvision cuda80 -c soumith
pip install -U pip
pip install -r requirements.txt
python setup.py install
bash download.sh
bash experiment_name1.sh