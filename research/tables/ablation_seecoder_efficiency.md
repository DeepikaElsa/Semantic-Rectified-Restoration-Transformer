# Computational cost — measured values only

| Metric | Restormer | Restormer + SeeCoder |
|---|---:|---:|
| Parameters | 26,126,644 | 26,560,329 |
| Train time (h) | 6.08 | 1.77 |
| Epochs completed | 60 | 18 |
| Median epoch time (min) | 5.82 | 5.67 |
| Peak VRAM, train (MB) | 5557 | 5463 |
| Peak RAM, train (MB) | 3438 | 2428 |
| Inference time / image (s) | 0.2183 | unavailable |
| Peak VRAM, eval (MB) | 6784 | unavailable |
| GFLOPs | unavailable | unavailable |

**GFLOPs:** unavailable — neither `thop` nor `fvcore` is installed. Never estimated; an estimated FLOP count would be indistinguishable from a measured one in this table.

* **Restormer**: 2 epoch(s) >1500s excluded as system standby, not compute: ep9 (17303s), ep12 (5515s)
* **Restormer + SeeCoder**: not evaluated yet (no summary.json under results/ablation_seecoder)

*Train time and median epoch time are wall-clock on one laptop GPU under WDDM and are not a controlled benchmark; they are comparable to each other because both runs used the same machine and configuration.*
