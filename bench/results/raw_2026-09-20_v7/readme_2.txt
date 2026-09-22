so: /root/.cache/torch_extensions/py312_cu128/attention_forward_cuda/attention_forward_cuda.so
====================================================================================================
Custom CUDA (+L) vs PyTorch SDPA backends — shape set 'readme', 10 paired reps, FP16
GPU: NVIDIA GeForce RTX 4060 Ti  torch 2.10.0+cu128
====================================================================================================
B=1 H=8/8 N= 1024 D= 64 dense : ours 0.0590ms (36.4 TF)  flash 0.0573ms (37.5 TF)  cudnn 0.0545ms (39.4 TF)
        paired gap vs flash +3.06% [+2.78, +6.15]  vs cudnn +8.59% [+6.72, +11.22]  (positive = ours slower)  maxdiff vs flash 2.4e-04
B=1 H=8/8 N= 1024 D= 64 causal: ours 0.0513ms (20.9 TF)  flash 0.0505ms (21.3 TF)  cudnn 0.0498ms (21.5 TF)
        paired gap vs flash +1.53% [+1.14, +3.48]  vs cudnn +3.16% [+2.64, +5.08]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 dense : ours 0.2158ms (39.8 TF)  flash 0.2158ms (39.8 TF)  cudnn 0.2045ms (42.0 TF)
        paired gap vs flash +0.03% [-1.71, +0.90]  vs cudnn +5.44% [+4.39, +5.77]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 causal: ours 0.1351ms (31.8 TF)  flash 0.1491ms (28.8 TF)  cudnn 0.1395ms (30.8 TF)
        paired gap vs flash -9.21% [-9.72, -8.05]  vs cudnn -3.12% [-3.35, -1.77]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 4096 D= 64 dense : ours 0.8348ms (41.2 TF)  flash 0.8473ms (40.6 TF)  cudnn 0.7982ms (43.0 TF)
        paired gap vs flash -1.20% [-3.34, +0.37]  vs cudnn +3.67% [+2.42, +4.67]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=8/8 N= 4096 D= 64 causal: ours 0.4672ms (36.8 TF)  flash 0.4955ms (34.7 TF)  cudnn 0.4880ms (35.2 TF)
        paired gap vs flash -5.75% [-6.25, -4.66]  vs cudnn -4.34% [-5.32, -3.88]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D= 64 dense : ours 0.8177ms (42.0 TF)  flash 0.8488ms (40.5 TF)  cudnn 0.7997ms (43.0 TF)
        paired gap vs flash -3.65% [-3.93, -2.20]  vs cudnn -0.10% [-1.09, +2.38]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D= 64 causal: ours 0.4466ms (38.5 TF)  flash 0.4696ms (36.6 TF)  cudnn 0.4665ms (36.8 TF)
        paired gap vs flash -4.87% [-5.07, -4.46]  vs cudnn -4.56% [-4.85, -2.88]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D= 64 dense : ours 3.2313ms (42.5 TF)  flash 3.2957ms (41.7 TF)  cudnn 3.0877ms (44.5 TF)
        paired gap vs flash -2.04% [-2.31, +0.58]  vs cudnn +4.09% [+0.76, +5.13]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D= 64 causal: ours 1.7010ms (40.4 TF)  flash 1.7525ms (39.2 TF)  cudnn 1.7004ms (40.4 TF)
        paired gap vs flash -2.87% [-4.07, +0.54]  vs cudnn -0.04% [-2.83, +1.16]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D=128 dense : ours 1.6794ms (40.9 TF)  flash 1.6784ms (40.9 TF)  cudnn 1.5606ms (44.0 TF)
        paired gap vs flash +0.12% [-1.27, +1.13]  vs cudnn +6.94% [+6.54, +7.66]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D=128 causal: ours 0.9533ms (36.0 TF)  flash 0.8659ms (39.7 TF)  cudnn 0.8438ms (40.7 TF)
        paired gap vs flash +10.41% [+9.86, +11.01]  vs cudnn +13.30% [+12.30, +14.18]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D=128 dense : ours 6.5228ms (42.1 TF)  flash 6.3939ms (43.0 TF)  cudnn 6.1459ms (44.7 TF)
        paired gap vs flash +1.90% [+1.38, +2.56]  vs cudnn +6.19% [+4.69, +7.29]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D=128 causal: ours 3.5499ms (38.7 TF)  flash 3.2819ms (41.9 TF)  cudnn 3.1791ms (43.2 TF)
        paired gap vs flash +8.13% [+6.87, +8.65]  vs cudnn +11.51% [+10.45, +12.38]  (positive = ours slower)  maxdiff vs flash 4.9e-04
====================================================================================================
