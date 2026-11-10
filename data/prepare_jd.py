"""Prepare JD micro-behavior sessions for BHSBR.

The raw format is one session per line. Each whitespace-separated event is:
``item_id+behavior_id+category_id+time_interval+dwell_time``.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp


DATASET_NAMES = ("Applicances", "Computers")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a JD dataset for BHSBR")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--min-item-support", type=int, default=1)
    return parser.parse_args()


def read_sessions(path):
    sessions = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            events = []
            for token in line.split():
                fields = token.split("+")
                if len(fields) != 5:
                    raise ValueError(f"invalid event at line {line_number}: {token}")
                events.append((int(fields[0]), int(fields[1])))
            if events:
                sessions.append(events)
    return sessions


def filter_and_map_sessions(sessions, max_length, min_item_support):
    support = Counter(item for session in sessions for item, _ in session)
    filtered = []
    for session in sessions:
        session = [event for event in session if support[event[0]] >= min_item_support]
        while len(session) > 1 and session[-1][0] == session[-2][0]:
            session.pop()
        if len(session) > 1:
            filtered.append(session[-max_length:])

    item_ids = sorted({item for session in filtered for item, _ in session})
    item_map = {old_id: new_id for new_id, old_id in enumerate(item_ids, 1)}
    mapped = [
        [(item_map[item], action) for item, action in session]
        for session in filtered
    ]
    return mapped, item_map


def split_sessions(sessions):
    train_end = int(len(sessions) * 0.7)
    valid_end = int(len(sessions) * 0.8)
    return sessions[:train_end], sessions[train_end:valid_end], sessions[valid_end:]


def build_example(session, action_count=10):
    target_item, target_action = session[-1]
    inputs = session[:-1]
    input_items = [item for item, _ in inputs]
    input_actions = [action for _, action in inputs]

    nodes = np.unique(input_items)
    node_lookup = {int(item): index for index, item in enumerate(nodes)}
    alias_inputs = np.asarray([node_lookup[item] for item in input_items], dtype=np.int32)

    boundaries = [
        index for index in range(len(input_items) - 1)
        if input_items[index] != input_items[index + 1]
    ] + [len(input_items) - 1]
    macro_items = [input_items[index] for index in boundaries]

    edge_count = len(macro_items)
    adjacency_in = np.zeros((len(nodes), edge_count), dtype=np.int64)
    adjacency_out = np.zeros((len(nodes), edge_count), dtype=np.int64)
    for edge_index in range(edge_count - 1):
        adjacency_out[node_lookup[macro_items[edge_index]], edge_index] = 1
        adjacency_in[node_lookup[macro_items[edge_index + 1]], edge_index] = 1
    adjacency = np.concatenate((adjacency_in, adjacency_out), axis=0).astype(np.float32)

    micro_actions = []
    micro_lengths = []
    start = 0
    for boundary in boundaries:
        actions = np.asarray(input_actions[start:boundary + 1], dtype=np.int32)
        micro_actions.append(actions)
        micro_lengths.append(len(actions))
        start = boundary + 1

    # The original EMBSR preprocessing rotates the target operation to the
    # front to align it with the prepended star node.
    actions = [target_action] + input_actions
    positions = [len(session)] + list(range(1, len(session)))
    pairs = np.zeros((len(actions), len(actions)), dtype=np.int64)
    for row, left in enumerate(actions):
        for col, right in enumerate(actions):
            pairs[row, col] = (left - 1) * action_count + right

    return (
        np.asarray(nodes, dtype=np.int32),
        alias_inputs,
        adjacency,
        len(alias_inputs),
        np.asarray(macro_items, dtype=np.int32),
        micro_actions,
        np.asarray(micro_lengths, dtype=np.int32),
        np.asarray(actions, dtype=np.int32),
        pairs.astype(np.int32),
        np.asarray(positions, dtype=np.int32),
        target_item,
    )


def build_global_adjacency(train_sessions, item_count):
    transitions = Counter()
    for session in train_sessions:
        items = [item for item, _ in session]
        for item in items:
            transitions[(item, item)] += 1
        for source, target in zip(items, items[1:]):
            transitions[(target, source)] += 1
            transitions[(source, target)] += 1

    rows = np.fromiter((key[0] for key in transitions), dtype=np.int64)
    cols = np.fromiter((key[1] for key in transitions), dtype=np.int64)
    values = np.fromiter(transitions.values(), dtype=np.float32)
    adjacency = sp.coo_matrix(
        (values, (rows, cols)), shape=(item_count + 1, item_count + 1)
    ).tocsr()
    row_sums = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    return (sp.diags(inverse) @ adjacency).tocoo()


def main():
    args = parse_args()
    sessions = read_sessions(args.input)
    sessions, item_map = filter_and_map_sessions(
        sessions, args.max_length, args.min_item_support
    )
    train, valid, test = split_sessions(sessions)
    if not train or not valid or not test:
        raise ValueError("not enough retained sessions for a 70/10/20 split")

    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in (("train", train), ("valid", valid), ("test", test)):
        with (output_dir / f"{split_name}_sets_mindspore.pkl").open("wb") as destination:
            pickle.dump(
                [build_example(session) for session in split],
                destination,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    adjacency = build_global_adjacency(train, len(item_map))
    with (output_dir / "adj.pickle").open("wb") as destination:
        pickle.dump(adjacency, destination, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "dataset": args.dataset,
        "source_file": args.input.name,
        "source_sessions": len(read_sessions(args.input)),
        "retained_sessions": len(sessions),
        "train_sessions": len(train),
        "valid_sessions": len(valid),
        "test_sessions": len(test),
        "item_vocab_size": len(item_map),
        "max_position": args.max_length,
        "min_item_support": args.min_item_support,
        "action_count": 10,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "item_map.json").write_text(
        json.dumps(item_map, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
