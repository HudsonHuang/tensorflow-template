#prepare environment and run default experiment

#prepare environment 
# source activate tensorflow
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
# pip install -U pip
# conda install pytorch torchvision cuda80 -c soumith
pip install -U pip
pip install -r requirements.txt
# python setup.py install

#run default experiment
bash ./experiment/Deep_mnist.sh