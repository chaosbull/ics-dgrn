# Checkpoints

Author: [chaosbull](https://github.com/chaosbull)  
License: Apache-2.0

This release does not ship trained `.pt` / `.pth` weights.
Tables and figures under `result/` were saved from training runs; state dicts were not exported.

To save weights yourself after training:

```python
torch.save(pack["model"].state_dict(), "checkpoints/ics_dgrn_pems08.pt")
```
