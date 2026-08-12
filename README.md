# BHSBR - MindSpore Ascend Migration

MindSpore implementation of "Behavior Habits Enhanced Intention Learning for Session-Based Recommendation".

## Contributions

- Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)

## Project support

由人工智能与数字经济广东省实验室（深圳）开放课题资助（编号：GML-KF-24-22）

This research was financially supported by the Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), under Grant No. GML-KF-24-22.

## Framework

The training path is implemented with native MindSpore APIs:

- `mindspore.nn.Cell` model components
- `mindspore.value_and_grad` automatic differentiation
- MindSpore Adam and CrossEntropyLoss
- Ascend device context
- NumPy/SciPy runtime dataset files

PyTorch and `torch_npu` are not imported by model training. PyTorch is only an optional one-time dependency for converting trusted historical `.pt` datasets.

## Ascend setup

The default setup targets MindSpore 2.9.0 and CANN 8.3.RC1 or newer compatible CANN releases.

```bash
bash setup.sh
```

Run training:

```bash
bash run_mindspore.sh --dataset Computers --batch-size 128
bash run_mindspore.sh --dataset Trivago --batch-size 128
```

For a short smoke run:

```bash
bash run_mindspore.sh --dataset Computers --batch-size 4 --epochs 1
```

## Dataset format

MindSpore training reads the following files from each dataset directory:

```text
adj.pickle
train_sets_mindspore.pkl
valid_sets_mindspore.pkl
test_sets_mindspore.pkl
metadata.json (optional)
```

Convert a trusted legacy PyTorch dataset once:

```bash
python data/convert_pt_to_mindspore.py --dataset-dir data/Computers
python data/convert_pt_to_mindspore.py --dataset-dir data/Trivago
```

After conversion, training no longer needs PyTorch.

The included JD preprocessing script now writes the MindSpore runtime format directly:

```bash
python data/prepare_jd.py --dataset Applicances --input data/Applicances/JD_Applicances_sample
python data/prepare_jd.py --dataset Computers --input data/Computers/JD_Computers_sample
```

## Evaluation

The program reports Recall/HR, MRR and NDCG at 5, 10 and 20. A single epoch is selected by validation HR@20 + MRR@20, and the corresponding test metrics are retained. Results and the best MindSpore checkpoint are written to `results/`.
