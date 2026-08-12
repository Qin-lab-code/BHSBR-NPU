"""Train BHSBR with the native MindSpore framework on Ascend or CPU."""

import argparse
import datetime
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import mindspore as ms
from mindspore import Tensor, nn, ops
import numpy as np

from BHSBR import BHSBR
from metric import evaluate_topk


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="MindSpore BHSBR training")
    parser.add_argument("--device-target", choices=("Ascend", "CPU"), default="Ascend")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dataset", choices=("Applicances", "Computers", "Trivago"), default="Computers")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def setup_runtime(args):
    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def pad_1d(values, pad_value=0, dtype=np.int32):
    width = max(len(value) for value in values)
    result = np.full((len(values), width), pad_value, dtype=dtype)
    for index, value in enumerate(values):
        result[index, : len(value)] = value
    return result


def collate_batch(batch):
    item_ids, alias, adjacency, lengths, macro, micro, micro_len, actions, pairs, positions, targets = zip(*batch)
    items = pad_1d(item_ids)
    aliases = pad_1d(alias, pad_value=-1)
    macros = pad_1d(macro)
    action_array = pad_1d(actions)
    position_array = pad_1d(positions)
    max_nodes = items.shape[1]
    max_edges = macros.shape[1]
    max_action_len = aliases.shape[1]

    adjacency_batch = np.zeros((len(batch), max_nodes, max_edges * 2), dtype=np.float32)
    micro_batch = np.zeros((len(batch), max_edges, max_action_len), dtype=np.int32)
    micro_len_batch = np.ones((len(batch), max_edges), dtype=np.int32)
    pair_batch = np.zeros((len(batch), action_array.shape[1], action_array.shape[1]), dtype=np.int32)

    for index, (matrix, edge_actions, edge_lengths, pair) in enumerate(zip(adjacency, micro, micro_len, pairs)):
        node_count = matrix.shape[0] // 2
        edge_count = matrix.shape[1]
        adjacency_batch[index, :node_count, :edge_count] = matrix[:node_count]
        adjacency_batch[index, :node_count, max_edges:max_edges + edge_count] = matrix[node_count:]
        for edge_index, edge_action in enumerate(edge_actions):
            micro_batch[index, edge_index, : len(edge_action)] = edge_action
        micro_len_batch[index, : len(edge_lengths)] = edge_lengths
        pair_batch[index, : pair.shape[0], : pair.shape[1]] = pair

    arrays = (
        items, adjacency_batch, aliases, np.asarray(lengths, np.int32), macros,
        micro_batch, micro_len_batch, action_array, pair_batch, position_array,
        np.asarray(targets, np.int32),
    )
    dtypes = (ms.int32, ms.float32, ms.int32, ms.int32, ms.int32, ms.int32, ms.int32, ms.int32, ms.int32, ms.int32, ms.int32)
    return tuple(Tensor(array, dtype) for array, dtype in zip(arrays, dtypes))


class BatchIterator:
    def __init__(self, examples, batch_size, shuffle=False, seed=1):
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.examples))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        for start in range(0, len(indices), self.batch_size):
            yield collate_batch([self.examples[index] for index in indices[start:start + self.batch_size]])


def load_split(dataset_dir, name):
    path = dataset_dir / f"{name}_sets_mindspore.pkl"
    if not path.exists():
        legacy = dataset_dir / f"{name}_sets_EMBSR.pt"
        if legacy.exists():
            raise FileNotFoundError(
                f"{path.name} is missing. Convert legacy data first: "
                f"python data/convert_pt_to_mindspore.py --dataset-dir {dataset_dir}"
            )
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_data(dataset_name, batch_size, seed):
    dataset_dir = BASE_DIR / "data" / dataset_name
    train = load_split(dataset_dir, "train")
    valid = load_split(dataset_dir, "valid")
    test = load_split(dataset_dir, "test")
    with (dataset_dir / "adj.pickle").open("rb") as stream:
        adjacency = pickle.load(stream)
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vocab_size = int(metadata["item_vocab_size"])
        max_position = int(metadata["max_position"])
    else:
        defaults = {"Applicances": (77259, 40), "Computers": (96291, 30), "Trivago": (183562, 50)}
        vocab_size, max_position = defaults[dataset_name]
    return (
        BatchIterator(train, batch_size, True, seed),
        BatchIterator(valid, batch_size),
        BatchIterator(test, batch_size),
        vocab_size,
        max_position,
        adjacency,
    )


def create_model(dataset, vocab_size, max_position, adjacency):
    if dataset == "Trivago":
        return BHSBR(vocab_size + 1, adjacency, 7, max_position + 1, 50, dropout_rate=0.5, alpha=12.0)
    return BHSBR(vocab_size + 1, adjacency, 11, max_position + 1, 101, dropout_rate=0.1 if dataset == "Computers" else 0.2, alpha=12.0, step=4)


def train_epoch(iterator, model, loss_fn, optimizer, epoch):
    model.set_train(True)

    def forward_fn(*batch):
        logits = model(*batch[:-1])
        return loss_fn(logits, batch[-1] - 1)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    total = 0.0
    for step, batch in enumerate(iterator, 1):
        loss, gradients = grad_fn(*batch)
        optimizer(gradients)
        value = float(loss.asnumpy())
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {step}: {value}")
        total += value
        if step % 100 == 0:
            logging.info("Epoch %d step %d/%d mean_loss %.4f", epoch, step, len(iterator), total / step)
    return total / max(len(iterator), 1)


def evaluate(iterator, model):
    model.set_train(False)
    predictions = {5: [], 10: [], 20: []}
    targets = []
    for batch in iterator:
        scores = model(*batch[:-1])
        _, top20 = ops.topk(scores, 20)
        top20 = top20.asnumpy()
        for k in predictions:
            predictions[k].append(top20[:, :k])
        targets.append(batch[-1].asnumpy() - 1)
    targets = np.concatenate(targets)
    return {k: evaluate_topk(np.concatenate(value), targets) for k, value in predictions.items()}


def score(metrics):
    return metrics[20][0] + metrics[20][1]


def main():
    args = parse_args()
    setup_runtime(args)
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = results_dir / f"mindspore_BHSBR_{args.dataset}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("MindSpore %s, target=%s:%d", ms.__version__, args.device_target, args.device_id)
    train, valid, test, vocab_size, max_position, adjacency = load_data(args.dataset, args.batch_size, args.seed)
    model = create_model(args.dataset, vocab_size, max_position, adjacency)
    learning_rate = 0.0003 if args.dataset == "Computers" else 0.001
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, use_amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()

    best_score = -1.0
    best_epoch = -1
    best_test = None
    patience = 0
    for epoch in range(args.epochs):
        mean_loss = train_epoch(train, model, loss_fn, optimizer, epoch)
        valid_metrics = evaluate(valid, model)
        test_metrics = evaluate(test, model)
        logging.info("epoch=%d loss=%.4f valid=%s test=%s", epoch, mean_loss, valid_metrics, test_metrics)
        current = score(valid_metrics)
        if current > best_score:
            best_score = current
            best_epoch = epoch
            best_test = test_metrics
            patience = 0
            ms.save_checkpoint(model, str(results_dir / f"best_{args.dataset}.ckpt"))
        else:
            patience += 1
        if patience > args.patience:
            break

    logging.info("best_epoch=%d best_test=%s", best_epoch, best_test)


if __name__ == "__main__":
    main()
