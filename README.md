# BHSBR - MindSpore Implementation

MindSpore implementation of "Behavior Habits Enhanced Intention Learning for Session-Based Recommendation".

## Contributions

- Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)

## Project support

由人工智能与数字经济广东省实验室（深圳）开放课题资助（编号：GML-KF-24-22）

This research was financially supported by the Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), under Grant No. GML-KF-24-22.

## Setup

The default setup targets MindSpore 2.9.0 and CANN 8.3.RC1 or newer compatible CANN releases.

```bash
bash setup.sh
```

Run training:

```bash
bash run_mindspore.sh --dataset Computers --batch-size 128
bash run_mindspore.sh --dataset Trivago --batch-size 128
```
