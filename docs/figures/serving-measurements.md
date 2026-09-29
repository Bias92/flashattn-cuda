# Serving Measurements

Median of three accepted runs per configuration and setting. TTFT and TPOT
are per-run request p50s, not pooled request percentiles. These are the
September 20-21 archived serving results, not measurements of the current build.

Times below are in milliseconds; throughput is total output tokens/s.
The long-context TTFT figure converts milliseconds to seconds.

| Label | Attention configuration |
|---|---|
| Native Flash | Native prefill and decode |
| Custom decode-only | Native prefill, custom decode |
| Custom prefill + decode | Custom prefill and decode |

## Low Latency

### Eager

**p50 TTFT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 28.988 | 30.042 | 22.476 |
| 512 | 34.638 | 34.423 | 32.979 |
| 1,024 | 62.057 | 61.798 | 60.389 |
| 1,920 | 115.672 | 115.836 | 112.395 |

**p50 TPOT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 10.848 | 10.520 | 10.673 |
| 512 | 11.268 | 10.715 | 10.940 |
| 1,024 | 11.279 | 10.366 | 10.688 |
| 1,920 | 11.882 | 10.464 | 10.650 |

**Output tokens/s**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 89.327 | 92.241 | 90.195 |
| 512 | 85.203 | 90.536 | 88.792 |
| 1,024 | 83.665 | 90.915 | 90.101 |
| 1,920 | 78.803 | 87.218 | 84.875 |

### CUDA Graphs

**p50 TTFT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 16.874 | 20.619 | 20.048 |
| 512 | 33.827 | 34.323 | 34.194 |
| 1,024 | 61.585 | 62.645 | 61.220 |
| 1,920 | 115.753 | 117.562 | 114.051 |

**p50 TPOT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 8.255 | 8.313 | 8.302 |
| 512 | 8.261 | 8.361 | 8.355 |
| 1,024 | 8.282 | 8.432 | 8.433 |
| 1,920 | 8.382 | 8.495 | 8.517 |

**Output tokens/s**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 128 | 119.611 | 118.688 | 118.983 |
| 512 | 117.483 | 116.363 | 116.472 |
| 1,024 | 114.332 | 112.767 | 112.906 |
| 1,920 | 108.055 | 106.837 | 106.904 |

## High Throughput

### Eager

**p50 TTFT (ms)**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 63.331 | 60.366 | 58.149 |
| 4 | 116.792 | 112.779 | 114.803 |
| 8 | 125.869 | 174.908 | 184.660 |
| 16 | 201.973 | 279.310 | 242.652 |
| 32 | 327.706 | 330.313 | 307.466 |

**p50 TPOT (ms)**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 12.088 | 11.280 | 11.179 |
| 4 | 13.087 | 10.806 | 11.013 |
| 8 | 13.459 | 11.869 | 11.749 |
| 16 | 14.664 | 13.205 | 12.507 |
| 32 | 17.429 | 15.893 | 15.537 |

**Output tokens/s**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 156.728 | 169.132 | 170.685 |
| 4 | 287.798 | 332.920 | 327.418 |
| 8 | 528.461 | 598.362 | 612.112 |
| 16 | 952.792 | 1026.217 | 1102.025 |
| 32 | 1599.307 | 1717.310 | 1779.234 |

### CUDA Graphs

**p50 TTFT (ms)**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 59.179 | 59.065 | 58.245 |
| 4 | 113.153 | 114.227 | 113.418 |
| 8 | 146.042 | 172.791 | 144.378 |
| 16 | 226.528 | 183.113 | 206.148 |
| 32 | 321.360 | 310.367 | 291.955 |

**p50 TPOT (ms)**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 8.538 | 8.544 | 8.533 |
| 4 | 8.618 | 8.630 | 8.642 |
| 8 | 9.456 | 9.522 | 9.533 |
| 16 | 10.888 | 11.236 | 11.017 |
| 32 | 14.446 | 15.133 | 14.486 |

**Output tokens/s**

| Concurrent requests | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2 | 224.548 | 225.102 | 224.256 |
| 4 | 418.972 | 419.205 | 423.398 |
| 8 | 744.520 | 735.773 | 756.012 |
| 16 | 1250.666 | 1215.820 | 1261.472 |
| 32 | 1855.617 | 1803.594 | 1875.279 |

## Long Context

### Eager

**p50 TTFT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 63.116 | 62.581 | 60.782 |
| 4,096 | 128.980 | 128.150 | 122.323 |
| 8,192 | 285.976 | 285.753 | 274.318 |
| 16,384 | 722.764 | 724.001 | 689.729 |
| 32,640 | 2063.087 | 2062.667 | 1957.973 |

**p50 TPOT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 14.406 | 13.124 | 13.162 |
| 4,096 | 14.828 | 12.970 | 13.113 |
| 8,192 | 14.293 | 12.848 | 13.054 |
| 16,384 | 14.703 | 12.941 | 13.224 |
| 32,640 | 14.371 | 12.650 | 13.095 |

**Output tokens/s**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 67.674 | 73.424 | 75.310 |
| 4,096 | 63.445 | 70.202 | 71.143 |
| 8,192 | 61.986 | 65.979 | 65.178 |
| 16,384 | 49.627 | 53.355 | 53.208 |
| 32,640 | 32.783 | 34.768 | 35.312 |

### CUDA Graphs

**p50 TTFT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 61.750 | 61.714 | 60.411 |
| 4,096 | 125.214 | 126.621 | 121.813 |
| 8,192 | 283.007 | 286.885 | 275.709 |
| 16,384 | 718.777 | 725.985 | 692.903 |
| 32,640 | 2054.669 | 2088.384 | 1982.867 |

**p50 TPOT (ms)**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 4.616 | 4.606 | 4.602 |
| 4,096 | 4.700 | 4.655 | 4.656 |
| 8,192 | 4.868 | 4.826 | 4.832 |
| 16,384 | 5.300 | 5.198 | 5.195 |
| 32,640 | 5.896 | 5.906 | 5.918 |

**Output tokens/s**

| Input tokens | Native Flash | Custom decode-only | Custom prefill + decode |
|---:|---:|---:|---:|
| 2,048 | 196.032 | 197.501 | 197.989 |
| 4,096 | 176.097 | 178.820 | 179.973 |
| 8,192 | 141.445 | 142.537 | 144.289 |
| 16,384 | 91.710 | 92.061 | 94.639 |
| 32,640 | 45.527 | 45.013 | 46.792 |

Serving percentage comparisons in the root README use these same medians
before rounding. Historical per-run ratios remain in the campaign report
and data.json.

[Raw runs and campaign report](../serving/prefill_campaign_2026-09-20/REPORT.md)
| [Unrounded values and source hashes](data.json)
