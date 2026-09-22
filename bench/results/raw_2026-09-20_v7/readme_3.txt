so: /root/.cache/torch_extensions/py312_cu128/attention_forward_cuda/attention_forward_cuda.so
====================================================================================================
Custom CUDA (+L) vs PyTorch SDPA backends — shape set 'readme', 10 paired reps, FP16
GPU: NVIDIA GeForce RTX 4060 Ti  torch 2.10.0+cu128
====================================================================================================
B=1 H=8/8 N= 1024 D= 64 dense : ours 0.0593ms (36.2 TF)  flash 0.0576ms (37.3 TF)  cudnn 0.0545ms (39.4 TF)
        paired gap vs flash +2.73% [+0.93, +3.82]  vs cudnn +8.84% [+4.23, +9.99]  (positive = ours slower)  maxdiff vs flash 2.4e-04
B=1 H=8/8 N= 1024 D= 64 causal: ours 0.0512ms (21.0 TF)  flash 0.0505ms (21.3 TF)  cudnn 0.0499ms (21.5 TF)
        paired gap vs flash +1.34% [+1.31, +7.05]  vs cudnn +2.57% [-2.30, +2.91]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 dense : ours 0.2178ms (39.4 TF)  flash 0.2155ms (39.9 TF)  cudnn 0.2044ms (42.0 TF)
        paired gap vs flash +1.53% [-0.40, +2.69]  vs cudnn +6.13% [+4.17, +6.43]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 causal: ours 0.1353ms (31.7 TF)  flash 0.1488ms (28.9 TF)  cudnn 0.1396ms (30.8 TF)
        paired gap vs flash -8.94% [-9.29, -7.85]  vs cudnn -3.08% [-3.23, -2.02]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 4096 D= 64 dense : ours 0.8377ms (41.0 TF)  flash 0.8508ms (40.4 TF)  cudnn 0.8003ms (42.9 TF)
        paired gap vs flash -1.63% [-2.05, +1.19]  vs cudnn +4.59% [+2.34, +4.88]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=8/8 N= 4096 D= 64 causal: ours 0.4687ms (36.7 TF)  flash 0.4978ms (34.5 TF)  cudnn 0.4887ms (35.2 TF)
        paired gap vs flash -5.91% [-6.52, -4.54]  vs cudnn -4.19% [-5.25, -3.26]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D= 64 dense : ours 0.8213ms (41.8 TF)  flash 0.8527ms (40.3 TF)  cudnn 0.8119ms (42.3 TF)
        paired gap vs flash -3.66% [-3.76, -1.39]  vs cudnn +1.02% [-0.87, +2.83]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D= 64 causal: ours 0.4482ms (38.3 TF)  flash 0.4736ms (36.3 TF)  cudnn 0.4587ms (37.4 TF)
        paired gap vs flash -5.29% [-6.38, -2.36]  vs cudnn -2.81% [-5.66, -1.76]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D= 64 dense : ours 3.2378ms (42.4 TF)  flash 3.3091ms (41.5 TF)  cudnn 3.0895ms (44.5 TF)
        paired gap vs flash -1.79% [-2.45, -0.04]  vs cudnn +3.95% [+0.43, +5.18]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D= 64 causal: ours 1.7029ms (40.4 TF)  flash 1.7550ms (39.2 TF)  cudnn 1.7002ms (40.4 TF)
        paired gap vs flash -2.53% [-4.96, -0.93]  vs cudnn -0.06% [-3.05, +1.01]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D=128 dense : ours 1.6813ms (40.9 TF)  flash 1.6841ms (40.8 TF)  cudnn 1.5632ms (44.0 TF)
        paired gap vs flash +0.15% [-1.16, +1.14]  vs cudnn +7.09% [+4.86, +7.92]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D=128 causal: ours 0.9562ms (35.9 TF)  flash 0.8677ms (39.6 TF)  cudnn 0.8476ms (40.5 TF)
        paired gap vs flash +10.46% [+9.32, +11.12]  vs cudnn +13.28% [+12.34, +14.16]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D=128 dense : ours 6.5303ms (42.1 TF)  flash 6.4325ms (42.7 TF)  cudnn 6.1439ms (44.7 TF)
        paired gap vs flash +2.00% [+1.33, +2.96]  vs cudnn +6.18% [+4.13, +7.49]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D=128 causal: ours 3.5526ms (38.7 TF)  flash 3.2876ms (41.8 TF)  cudnn 3.1897ms (43.1 TF)
        paired gap vs flash +8.63% [+6.90, +9.84]  vs cudnn +11.79% [+11.51, +12.28]  (positive = ours slower)  maxdiff vs flash 4.9e-04
====================================================================================================
