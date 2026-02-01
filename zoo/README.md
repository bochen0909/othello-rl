checkpoint_20260131 was trained by:
python scripts/train_othello.py \
  --start-player random \
  --num-iterations 200 \
  --num-workers 8 \
  --num-gpus 1 \
  --num-cpus 12 \
  --train-batch-size 16000 \
  --minibatch-size 512 \
  --num-sgd-iter 20 \
  --checkpoint-freq 20 \
  --opponent "random,greedy"


python scripts/train_othello.py \
  --start-player random \
  --num-iterations 300 \
  --sample-timeout-s 300 \
  --num-workers 4 \
  --num-gpus 1 \
  --num-cpus 12 \
  --train-batch-size 16000 \
  --minibatch-size 512 \
  --num-sgd-iter 20 \
  --checkpoint-freq 20 \
  --resume-checkpoint zoo/checkpoint_20260131/ \
  --opponent "drohh,aelskels"
  #--opponent "random,greedy,nealetham,drohh,aelskels"
