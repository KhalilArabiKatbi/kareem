# Worker task

Implement the coordinate below as `model.py`. Every coordinate value must be realised faithfully;
the dimension definitions explain what each value means.

## Coordinate (next_coordinate.json)

```json
{{COORDINATE}}
```

## Dimension definitions

```json
{{DIMENSIONS}}
```

## Reference documents (ai_docs/, read-only)

{{AI_DOCS}}
{{RETRY_ERROR}}
## Output

Return JSON with:

- `model_py`: the complete Python source of `model.py` (module-level `FEATURE_NAMES`,
  `featurize`, `build_model`, and `TRAIN_CONFIG` reflecting every training-related coordinate
  value; optional `loss_fn`).
- `design_notes`: <=2500 characters mapping each coordinate key to how the code realises it.

Checklist before you answer:

- Only allowed imports; no file, network, process or dynamic-code access; no `torch.manual_seed`,
  `.cuda()`, `torch.compile`, `torch.load/save`; no `while True`.
- `len(featurize(history)) == len(FEATURE_NAMES)` for every history length, including 1.
- `featurize` costs well under 0.3 ms per call; bound any windows (e.g. `history[-8:]`).
- `build_model(input_dim, n_actions)` returns an `nn.Module` whose forward maps `[B, input_dim]`
  to `[B, n_actions]` logits and works in both train and eval mode for batch size 1.
