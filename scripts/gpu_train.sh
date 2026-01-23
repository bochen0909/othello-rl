python scripts/train_othello.py \
  --num-iterations 200 \
  --num-workers 12 \
  --num-gpus 1 \
  --num-cpus 16 \
  --train-batch-size 16000 \
  --minibatch-size 512 \
  --num-sgd-iter 20 \
  --checkpoint-freq 20