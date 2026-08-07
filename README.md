# GSSFormer paper-aligned code

This package corresponds to the manuscript titled **GSSFormer: A geometry-conditioned state-space network for semantic segmentation of full-scene traditional-village point clouds**.

The default training protocol is fixed at 300 epochs. The training script does not provide an epoch override, and the evaluation script accepts only a final checkpoint saved at epoch 300.
## Files

- `config.py`: datasets, the 300-epoch protocol, and the ablation variants reported in Table 3
- `dataset.py`: 10-dimensional input construction, 16-point local attributes, 16-neighbor boundary labels, spatial-block sampling, and cache validation
- `models.py`: four-stage PTv3-C backbone, GSSM, CAMP, and executable ablation switches
- `train.py`: AdamW training, five-epoch warm-up, cosine annealing, four-step gradient accumulation, and boundary auxiliary supervision
- `evaluate.py`: 10 m overlapping-window inference and three-run metric averaging

## Reported variants

| Variant | Paper configuration |
|---|---|
| `ptv3_c` | PTv3-C |
| `ptv3_c_camp` | PTv3-C + CAMP |
| `ptv3_c_gssm` | PTv3-C + GSSM |
| `gssformer` | GSSFormer |
| `ssm_standard` | Multi-order bidirectional SSM without conditioning |
| `ssm_geometry` | SSM + geometry modulation |
| `ssm_boundary` | SSM + boundary-response modulation |
| `gssm_main_head` | Main segmentation head |
| `gssm_shared_fusion` | Shared multi-scale fusion |

## Training

```bash
python train.py --dataset village --variant gssformer --data-root /path/to/village --output outputs/village/gssformer --seed 3407
```

```bash
python train.py --dataset sensaturban --variant gssformer --data-root /path/to/sensaturban --output outputs/sensaturban/gssformer --seed 3407
```

Use the same seed and protocol for all ablation variants. The comparison methods outside this package must also be configured for 300 epochs in their respective public implementations.

## Evaluation

```bash
python evaluate.py --dataset village --data-root /path/to/village --checkpoint outputs/village/gssformer/final_model.pth --output outputs/village/gssformer/test --seed 3407
```

```bash
python evaluate.py --dataset sensaturban --data-root /path/to/sensaturban --checkpoint outputs/sensaturban/gssformer/final_model.pth --output outputs/sensaturban/gssformer/test --seed 3407
```

Evaluation always performs three independent full-scene inference runs. The individual results, run seeds, mean metrics, and standard deviations are written to `summary.json`.
