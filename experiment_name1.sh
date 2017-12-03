#path or other
echo "To checkout tensorboard, run `tensorboard --logdir=./logdir/experiment_name1`"
python prepare_features.py
python main.py
echo "Please find your results in `./generated/experiment_name1`"