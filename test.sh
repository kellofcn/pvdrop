CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/fern/ -m output/fern/ -r 8 --eval  --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/fern/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/flower/ -m output/flower/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/flower/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/fortress/ -m output/fortress/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/fortress/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/horns/ -m output/horns/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/horns/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/leaves/ -m output/leaves/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/leaves/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/trex/ -m output/trex/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/trex/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/orchids/ -m output/orchids/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/orchids/

CUDA_VISIBLE_DEVICES=1 python train.py -s data/llff/room/ -m output/room/ -r 8 --eval --n_views 3
CUDA_VISIBLE_DEVICES=1 python render.py -m output/room/
