# F01 Global Endpoint 基座

本目录三组实验均固定历史 `PCS_RC_LC_R1_FF_R3_GLOBAL_ENDPOINT` 的：
- PCS-RC source + recurrent proposal context
- adaptive-g FF-R3 graph-level H3 translation path
- `LOSS_MODE=foldflow_r3_global_endpoint`
- clean Endpoint supervision
- `SAMPLER_MODE=bridge`
- `R3_TRANSPORT_FRACTION=0.05`
- `R3_PATH_MIN_SIGMA=0`
- `COORD_PEP_AS_CONDITION=on`
- `SEQ_INPUT_MODE=pep_condition`
- `PROPOSAL_ADAPTER_START_ROUND=1`
- `FINAL_READOUT_MODE=integrated_endpoint`
- SATC/DSM/pathflow/SF2M auxiliary losses all off
- batch=56, max_epoch=200, EMA=.999

实验树不是线性的：
- F01 -> G00：验证 AbX-inspired interface pair-time；
- F01 -> G01：验证 explicit global Flow pair semantics；
- G01 -> G02：验证 Score 在 Flow 之上的增量。
