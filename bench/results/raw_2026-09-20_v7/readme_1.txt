so: /root/.cache/torch_extensions/py312_cu128/attention_forward_cuda/attention_forward_cuda.so
====================================================================================================
Custom CUDA (+L) vs PyTorch SDPA backends — shape set 'readme', 10 paired reps, FP16
GPU: NVIDIA GeForce RTX 4060 Ti  torch 2.10.0+cu128
====================================================================================================
B=1 H=8/8 N= 1024 D= 64 dense : ours 0.0601ms (35.8 TF)  flash 0.0603ms (35.6 TF)  cudnn 0.0555ms (38.7 TF)
        paired gap vs flash +2.61% [-7.26, +4.15]  vs cudnn +8.74% [+3.75, +11.53]  (positive = ours slower)  maxdiff vs flash 2.4e-04
B=1 H=8/8 N= 1024 D= 64 causal: ours 0.0517ms (20.8 TF)  flash 0.0534ms (20.1 TF)  cudnn 0.0504ms (21.3 TF)
        paired gap vs flash +0.07% [-18.66, +1.45]  vs cudnn +2.47% [-19.07, +3.01]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 dense : ours 0.2186ms (39.3 TF)  flash 0.2175ms (39.5 TF)  cudnn 0.2074ms (41.4 TF)
        paired gap vs flash +0.37% [-0.23, +1.76]  vs cudnn +5.18% [+4.07, +6.29]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 2048 D= 64 causal: ours 0.1371ms (31.3 TF)  flash 0.1513ms (28.4 TF)  cudnn 0.1414ms (30.4 TF)
        paired gap vs flash -9.32% [-9.70, -8.12]  vs cudnn -3.12% [-3.32, -1.62]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=8/8 N= 4096 D= 64 dense : ours 0.8438ms (40.7 TF)  flash 0.8563ms (40.1 TF)  cudnn 0.8112ms (42.4 TF)
        paired gap vs flash -1.06% [-1.82, +0.04]  vs cudnn +3.77% [+2.86, +4.49]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=8/8 N= 4096 D= 64 causal: ours 0.4745ms (36.2 TF)  flash 0.5030ms (34.2 TF)  cudnn 0.4968ms (34.6 TF)
        paired gap vs flash -5.72% [-6.43, -4.04]  vs cudnn -4.48% [-5.30, -3.83]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D= 64 dense : ours 0.8276ms (41.5 TF)  flash 0.8549ms (40.2 TF)  cudnn 0.8135ms (42.2 TF)
        paired gap vs flash -3.25% [-3.76, -2.19]  vs cudnn +0.61% [-0.36, +1.55]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D= 64 causal: ours 0.4548ms (37.8 TF)  flash 0.4758ms (36.1 TF)  cudnn 0.4672ms (36.8 TF)
        paired gap vs flash -4.80% [-5.31, -3.61]  vs cudnn -3.39% [-4.67, -2.29]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D= 64 dense : ours 3.2777ms (41.9 TF)  flash 3.3294ms (41.3 TF)  cudnn 3.1365ms (43.8 TF)
        paired gap vs flash -1.52% [-1.95, +0.35]  vs cudnn +4.47% [+3.12, +4.81]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D= 64 causal: ours 1.7316ms (39.7 TF)  flash 1.7668ms (38.9 TF)  cudnn 1.7075ms (40.2 TF)
        paired gap vs flash -1.83% [-3.02, +0.05]  vs cudnn +1.48% [-3.25, +2.72]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 2048 D=128 dense : ours 1.6976ms (40.5 TF)  flash 1.6993ms (40.4 TF)  cudnn 1.5833ms (43.4 TF)
        paired gap vs flash -0.60% [-1.19, +0.72]  vs cudnn +6.85% [+6.18, +7.34]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 2048 D=128 causal: ours 0.9614ms (35.7 TF)  flash 0.8762ms (39.2 TF)  cudnn 0.8567ms (40.1 TF)
        paired gap vs flash +9.94% [+8.58, +10.82]  vs cudnn +12.51% [+11.99, +13.75]  (positive = ours slower)  maxdiff vs flash 4.9e-04
B=1 H=32/8 N= 4096 D=128 dense : ours 6.5930ms (41.7 TF)  flash 6.4854ms (42.4 TF)  cudnn 6.2066ms (44.3 TF)
        paired gap vs flash +1.85% [+1.01, +2.21]  vs cudnn +5.99% [+4.03, +7.09]  (positive = ours slower)  maxdiff vs flash 1.2e-04
B=1 H=32/8 N= 4096 D=128 causal: ours 3.5747ms (38.4 TF)  flash 3.3242ms (41.3 TF)  cudnn 3.2266ms (42.6 TF)
        paired gap vs flash +7.81% [+7.09, +8.44]  vs cudnn +10.95% [+10.62, +12.02]  (positive = ours slower)  maxdiff vs flash 4.9e-04
====================================================================================================
