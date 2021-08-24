#!/bin/sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python reduction.py --seed 3 --interest 8 --perp 85 --epsi 200 --step 3000 --select UMAP
CUDA_VISIBLE_DEVICES=0 python reduction.py --seed 3 --interest 12 --perp 85 --epsi 200 --step 3000 --select UMAP
CUDA_VISIBLE_DEVICES=0 python reduction.py --seed 3 --interest 16 --perp 85 --epsi 200 --step 3000 --select UMAP