# Checkpoints

Author: chaosbull  
License: Apache-2.0  

This release does **not** ship trained `.pt` / `.pth` weights.  
Metrics and figures under `result/` were produced during training without exporting state dicts.

To save weights yourself after training:

```python
torch.save(pack["model"].state_dict(), "checkpoints/ics_dgrn_pems08.pt")
```
