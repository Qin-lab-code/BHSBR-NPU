# BHSBR
This is our implementation for paper "Behavior Habits Enhanced Intention Learning for Session-Based Recommendation"

## Contributions

- Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)

## Project support

由人工智能与数字经济广东省实验室（深圳）开放课题资助（编号：GML-KF-24-22）

This research was financially supported by the Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), under Grant No. GML-KF-24-22.

## Ascend NPU migration

The training code supports Ascend NPU through `torch_npu`, while retaining CUDA and CPU fallbacks. In automatic mode the runtime preference is NPU, CUDA, then CPU.

```bash
# Ascend NPU (recommended on an Ascend host)
python main.py --device npu --device-id 0 --dataset Applicances

# Automatic fallback or CPU smoke run
python main.py --device auto --dataset Applicances
python main.py --device cpu --dataset sample
```

Install a `torch_npu` build compatible with the host PyTorch and CANN versions before selecting `--device npu`.

On an Ascend host, install the common Python dependencies and validate CANN/HCCL/NPU availability first:

```bash
bash setup.sh
bash run_npu.sh --dataset Computers --batch-size 128
```

`setup.sh` also handles clean shells where `PYTHONPATH` and `LD_LIBRARY_PATH` have not been defined before sourcing CANN's `set_env.sh`.
