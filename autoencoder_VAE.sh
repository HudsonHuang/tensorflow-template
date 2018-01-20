#path or other
python dataset_prepare.py \
    --dataset_name MNIST \
    --base_url http://yann.lecun.com/exdb/mnist/
python main.py \
    --model autoencoder_vae \
    --total_epoch 10000