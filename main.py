import numpy as np
import torch
from torch import optim
from metric import get_mrr, get_recall, get_ndcg
import datetime
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
from torch.nn.utils.rnn import pad_sequence
from BHSBR import BHSBR
from device import resolve_device
import datetime
# from Model.PairRNN import PairSelfAttentionLayer
import argparse



import logging
import time
import sys
import os
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Logging setup: log to file + console
def setup_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger




class ListDataset(Dataset):
    def __init__(self, *datalist):
        assert all(len(datalist[0]) == len(data) for data in datalist)
        self.datalist = datalist

    def __getitem__(self, index):
        return tuple(data[index] for data in self.datalist)

    def __len__(self):
        return len(self.datalist[0])


def load_legacy_dataset(path):
    """Load the trusted bundled legacy dataset on old and new PyTorch releases."""
    main_module = sys.modules.get('__main__')
    if main_module is not None and not hasattr(main_module, 'ListDataset'):
        setattr(main_module, 'ListDataset', ListDataset)
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def batch_padding(batch):
    # 根据信息进行padding
    # print(len(batch[0]))
    item_ids, alias_inputs, A, seq_real_len, macro_items, micro_actions, micro_len, action, pairs, poses, y = zip(
        *batch)
    item_ids = pad_sequence(item_ids, batch_first=True).long()
    alias_inputs = pad_sequence(alias_inputs, batch_first=True, padding_value=-1).long()
    max_action_len = alias_inputs.size(1)
    batch_max_length = item_ids.size(1)
    macro_items = pad_sequence(macro_items, batch_first=True, padding_value=0)  # n_edges
    batch_edge_length = macro_items.size(1)
    new_A = []
    for a in A:
        node_len, edge_len = int(a.size(0) / 2), a.size(1)
        a_in, a_out = a[:node_len, :], a[node_len:, :]
        pad_items_length = batch_max_length - node_len
        pad_edges_length = batch_edge_length - edge_len
        pad_tuple = (0, pad_edges_length, 0, pad_items_length)
        a_in, a_out = F.pad(a_in, pad_tuple), F.pad(a_out, pad_tuple)
        new_A.append(torch.cat((a_in, a_out), 1).tolist())
    new_micro_actions = []
    for ac in micro_actions:
        new_ac = pad_sequence(ac, batch_first=True, padding_value=0)
        pad_edge_size = batch_edge_length - new_ac.size(0)
        pad_action_size = max_action_len - new_ac.size(1)
        new_ac = F.pad(new_ac, (0, pad_action_size, 0, pad_edge_size))
        new_micro_actions.append(new_ac.tolist())
    micro_len = pad_sequence(micro_len, batch_first=True, padding_value=1)  # 这里需要注意，padding的内容最后是用不上的
    action = pad_sequence(action, batch_first=True)
    poses = pad_sequence(poses, batch_first=True)
    new_pairs = []
    seq_len = action.size(1)
    for pair in pairs:
        length = pair.size(0)
        pad_length = seq_len - length
        pad_tuple = (0, pad_length, 0, pad_length)
        new_pair = F.pad(pair, pad_tuple)
        new_pairs.append(new_pair.tolist())
    return item_ids, alias_inputs, torch.LongTensor(new_A), torch.LongTensor(seq_real_len), torch.LongTensor(
        macro_items), torch.LongTensor(new_micro_actions), torch.LongTensor(micro_len), action, torch.Tensor(
        new_pairs), poses, torch.LongTensor(y)

def preprocess_data(data_name, batch_size=128):
    dataset_dir = BASE_DIR / 'data' / data_name
    train_sets = load_legacy_dataset(dataset_dir / 'train_sets_EMBSR.pt')
    valid_sets = load_legacy_dataset(dataset_dir / 'valid_sets_EMBSR.pt')
    test_sets = load_legacy_dataset(dataset_dir / 'test_sets_EMBSR.pt')
    # if data_name == 'Trivago' or data_name == 'sample':
    #     train_sets = torch.load('../data/%s/train_sets_EMBSR.pt'% data_name)
    #     valid_sets = torch.load('../data/%s/valid_sets_EMBSR.pt'% data_name)
    #     test_sets = torch.load('../data/%s/test_sets_EMBSR.pt'% data_name)
    # else:
    #     train_sets = torch.load('../data/%s/train_sets_EMBSR.pt'% data_name)
    #     valid_sets = torch.load('../data/%s/valid_sets_EMBSR.pt'% data_name)
    #     test_sets = torch.load('../data/%s/test_sets_EMBSR.pt'% data_name)
        #train_sets, valid_sets, test_sets = train_dataload.dataset, valid_dataload.dataset, test_dataload.dataset
    with (dataset_dir / 'adj.pickle').open('rb') as adj_file:
        adj = pickle.load(adj_file)
    train_dataload = DataLoader(train_sets, batch_size=batch_size, shuffle=True,collate_fn=batch_padding, num_workers=0)
    valid_dataload = DataLoader(valid_sets, batch_size=batch_size, shuffle=False,collate_fn=batch_padding, num_workers=0)
    test_dataload = DataLoader(test_sets, batch_size=batch_size, shuffle=False,collate_fn=batch_padding, num_workers=0)
    metadata_path = dataset_dir / 'metadata.json'
    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
        item_vocab_size = metadata['item_vocab_size']
        max_position = metadata['max_position']
    elif data_name == 'Applicances':
        item_vocab_size = 77259
        max_position = 20
    elif data_name == 'Computers':
        item_vocab_size = 96290 + 1
        max_position = 20
    elif data_name == 'sample':
        item_vocab_size = 2154
        max_position = 50
    else:
        item_vocab_size = 183561 + 1
        max_position = 50
    return train_dataload, valid_dataload, test_dataload, item_vocab_size, max_position, adj


def train_process(train_data, model, criterion, opti, epoch, device):
    losses = 0
    steps = len(train_data)
    for step, (x_items, x_alias, x_A, x_len, x_macro, x_micro, x_micro_len, x_action, x_pairs, x_poses, y_train) in enumerate(train_data):
        # new_index = [ i for i in range(1, x_poses.size(1))] + [0]
        # x_action, x_poses, x_pairs = x_action[:,new_index], x_poses[:, new_index], x_pairs[:, new_index, new_index]
        opti.zero_grad()
        q = model(
            x_items.to(device), x_A.float().to(device), x_alias.to(device),
            x_len.to(device), x_macro.to(device), x_micro.to(device),
            x_micro_len.to(device), x_action.long().to(device),
            x_pairs.long().to(device), x_poses.long().to(device)
        )
        target_items = y_train
        loss = criterion(q, target_items.to(device)-1)
        loss.backward()
        opti.step()
        losses += loss.item()
        if (step + 1) % 100 == 0:
            logging.info("[Epoch %02d/%d] Step %03d/%d - Mean Loss: %.4f", epoch, 30, step, steps, losses / (step + 1))

def valid_process(valid_data, model, valid_test, device):
    y_pre_item_all = torch.empty((0, 20), dtype=torch.long, device=device)
    y_pre_item_all_5 = torch.empty((0, 5), dtype=torch.long, device=device)
    y_pre_item_all_10 = torch.empty((0, 10), dtype=torch.long, device=device)
    valid_target_item = torch.empty(0, dtype=torch.long, device=device)
    for x_test_items, x_test_alias, x_test_A, x_test_len, x_test_macro, x_test_micro, x_test_micro_len, x_action_test, x_pairs_test, x_poses_test, y_test in valid_data:
        with torch.no_grad():
            # if valid_test == 'test':
            #     x_action_test, x_pairs_test, x_poses_test = x_action_test[:,:-1], x_pairs_test[:, :-1, :-1], x_poses_test[:, :-1]
            scores = model(
                x_test_items.to(device), x_test_A.float().to(device),
                x_test_alias.to(device), x_test_len.to(device),
                x_test_macro.to(device), x_test_micro.to(device),
                x_test_micro_len.to(device), x_action_test.long().to(device),
                x_pairs_test.long().to(device), x_poses_test.long().to(device)
            )
            pre_items = scores.topk(20, dim=1)[1]
            pre_items_5 = pre_items[:, :5]
            pre_items_10 = pre_items[:, :10]
            y_pre_item_all_5 = torch.cat((y_pre_item_all_5, pre_items_5), 0)
            y_pre_item_all_10 = torch.cat((y_pre_item_all_10, pre_items_10), 0)
            y_pre_item_all = torch.cat((y_pre_item_all, pre_items), 0)
        valid_target_item = torch.cat((valid_target_item, y_test.to(device)))
    items_recall_5 = get_recall(y_pre_item_all_5, valid_target_item.unsqueeze(1)-1)
    items_mrr_5 = get_mrr(y_pre_item_all_5, valid_target_item.unsqueeze(1)-1)
    items_ndcg_5 = get_ndcg(y_pre_item_all_5.cpu(), valid_target_item.unsqueeze(1).cpu()-1)
    items_recall_10 = get_recall(y_pre_item_all_10, valid_target_item.unsqueeze(1)-1)
    items_mrr_10 = get_mrr(y_pre_item_all_10, valid_target_item.unsqueeze(1)-1)
    items_ndcg_10 = get_ndcg(y_pre_item_all_10.cpu(), valid_target_item.unsqueeze(1).cpu()-1)
    items_recall = get_recall(y_pre_item_all, valid_target_item.unsqueeze(1)-1)
    items_mrr = get_mrr(y_pre_item_all, valid_target_item.unsqueeze(1)-1)
    items_ndcg = get_ndcg(y_pre_item_all.cpu(), valid_target_item.unsqueeze(1).cpu()-1)
    logging.info('%s Result: HR@20: %.4f  MRR@20: %.4f  NDCG@20: %.4f',
             valid_test, items_recall, items_mrr.tolist(), items_ndcg.tolist())
    
    if valid_test == 'test':
        logging.info('%s Result: HR@5: %.4f  MRR@5: %.4f  NDCG@5: %.4f',
                     valid_test, items_recall_5, items_mrr_5.tolist(), items_ndcg_5.tolist())
        logging.info('%s Result: HR@10: %.4f  MRR@10: %.4f  NDCG@10: %.4f',
                     valid_test, items_recall_10, items_mrr_10.tolist(), items_ndcg_10.tolist())

        return items_recall, items_mrr, [items_recall_5, items_mrr_5, items_ndcg_5,items_recall_10, items_mrr_10, items_ndcg_10,items_recall, items_mrr, items_ndcg]
    else:
        return items_recall, items_mrr, [items_recall, items_mrr, items_ndcg]




def get_model(model_name, item_vocab_size, max_position, config, drop_, alpha_, adj, device):
    item_embedding_size, behavior_embedding_size, hidden_size = config[0], config[1], config[2]
    if model_name == 'EMBSR':
        model = BHSBR(item_vocab_size + 1, adj, 11, max_position+1, 101, item_embedding_size, behavior_embedding_size, 100, hidden_size , drop_, alpha_, step=4)
    elif model_name == 'EMBSR_trivago':
        model = BHSBR(item_vocab_size + 1, adj, 7, max_position+1, 50, item_embedding_size, behavior_embedding_size, 100, hidden_size , drop_, alpha_)
    return model.to(device)


# def run(train_data, valid_data, test_data, item_vocab_size, max_position, model_name, config, result_file_name, lr_, drop_, alpha, adj):
#     model = get_model(model_name, item_vocab_size, max_position, config, drop_, alpha, adj)
#     opti = optim.Adam(model.parameters(), lr=lr_, weight_decay=0, amsgrad=True)

#     best_hr, best_mrr = 0, 0
#     top_K = [5, 10, 20]
#     best_results = {}
#     metrics = {}
#     for K in top_K:
#         best_results['epoch%d' % K] = [0, 0, 0]
#         best_results['metric%d' % K] = [0, 0, 0]

#     best_test_list = []
#     stop_number = 0
#     best_epoch = 0

#     log_path = result_file_name + ".log"
#     setup_logger(log_path)
#     logging.info("Training started. Model: %s", model_name)

#     for epoch in range(50):
#         if stop_number > 10:
#             break
#         epoch_start = time.time()
#         logging.info("Epoch %d started", epoch)

#         model.train()
#         train_process(train_data, model, criterion, opti, epoch)

#         model.eval()
#         with torch.no_grad():
#             valid_hr, valid_mrr, valid_results = valid_process(valid_data, model, 'valid')
#             test_hr, test_mrr, test_results = valid_process(test_data, model, 'test')

#             metrics['hr5'] = test_results[0]
#             metrics['hr10'] = test_results[3]
#             metrics['hr20'] = test_results[6]
#             metrics['mrr5'] = test_results[1].tolist()
#             metrics['mrr10'] = test_results[4].tolist()
#             metrics['mrr20'] = test_results[7].tolist()
#             metrics['ndcg5'] = test_results[2].tolist()
#             metrics['ndcg10'] = test_results[5].tolist()
#             metrics['ndcg20'] = test_results[8].tolist()

#         for K in top_K:
#             if epoch == 0:
#                 best_results['metric%d' % K][0] = metrics['hr%d' % K]
#                 best_results['metric%d' % K][1] = metrics['mrr%d' % K]
#                 best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
#             else:
#                 if best_results['metric%d' % K][0] < metrics['hr%d' % K]:
#                     best_results['metric%d' % K][0] = metrics['hr%d' % K]
#                     best_results['epoch%d' % K][0] = epoch
#                 if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
#                     best_results['metric%d' % K][1] = metrics['mrr%d' % K]
#                     best_results['epoch%d' % K][1] = epoch
#                 if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
#                     best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
#                     best_results['epoch%d' % K][2] = epoch

#         best_ = (valid_hr - best_hr) / best_hr + (valid_mrr - best_mrr) / best_mrr if best_hr > 0 and best_mrr > 0 else 1
#         if best_ > 0:
#             stop_number = 0
#             best_hr, best_mrr = valid_hr, valid_mrr
#             best_test_hr, best_test_mrr = test_hr, test_mrr
#             best_test_list = test_results
#             best_epoch = epoch
#         else:
#             stop_number += 1

#         logging.info("Validation HR@20: %.4f, MRR@20: %.4f, NDCG@20: %.4f", valid_results[0], valid_results[1], valid_results[2])
#         logging.info("Test HR@20: %.4f, MRR@20: %.4f, NDCG@20: %.4f", test_results[6], test_results[7], test_results[8])
#         logging.info("Best HR so far: %.4f, Best MRR so far: %.4f, Best Epoch: %d", best_hr, best_mrr, best_epoch)

#         for K in top_K:
#             logging.info("Best @%d: HR: %.4f (ep%d), MRR: %.4f (ep%d), NDCG: %.4f (ep%d)",
#                          K,
#                          best_results['metric%d' % K][0], best_results['epoch%d' % K][0],
#                          best_results['metric%d' % K][1], best_results['epoch%d' % K][1],
#                          best_results['metric%d' % K][2], best_results['epoch%d' % K][2])

#         epoch_end = time.time()
#         elapsed = epoch_end - epoch_start
#         logging.info("Epoch %d finished in %.2f seconds (%.2f minutes)", epoch, elapsed, elapsed / 60)

#     logging.info("Training finished.")
#     logging.info("Best Validation HR@20: %.4f, MRR@20: %.4f", best_hr, best_mrr)
#     logging.info("Best Test HR@20: %.4f, MRR@20: %.4f", best_test_hr, best_test_mrr)
#     logging.info("Full best test result list: %s", str(best_test_list))

def run(train_data, valid_data, test_data, item_vocab_size, max_position, model_name, config, result_file_name, lr_, drop_, alpha, adj, device, epochs=50, patience=10):
    model = get_model(model_name, item_vocab_size, max_position, config, drop_, alpha, adj, device)
    criterion = nn.CrossEntropyLoss().to(device)
    opti = optim.Adam(model.parameters(), lr=lr_, weight_decay=0, amsgrad=True)

    best_hr, best_mrr = 0, 0
    top_K = [5, 10, 20]
    best_results = {}
    metrics = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    best_test_list = []
    stop_number = 0
    best_epoch = 0

    log_path = result_file_name + ".log"
    setup_logger(log_path)
    logging.info("Training started. Model: %s", model_name)

    for epoch in range(epochs):
        if stop_number > patience:
            break

        epoch_start = time.time()
        logging.info("Epoch %d started", epoch)

        # 训练阶段
        train_start = time.time()
        model.train()
        train_process(train_data, model, criterion, opti, epoch, device)
        train_end = time.time()

        # 推理阶段（验证 + 测试）
        inference_start = time.time()
        model.eval()
        with torch.no_grad():
            valid_hr, valid_mrr, valid_results = valid_process(valid_data, model, 'valid', device)
            test_hr, test_mrr, test_results = valid_process(test_data, model, 'test', device)

            metrics['hr5'] = test_results[0]
            metrics['hr10'] = test_results[3]
            metrics['hr20'] = test_results[6]
            metrics['mrr5'] = test_results[1].tolist()
            metrics['mrr10'] = test_results[4].tolist()
            metrics['mrr20'] = test_results[7].tolist()
            metrics['ndcg5'] = test_results[2].tolist()
            metrics['ndcg10'] = test_results[5].tolist()
            metrics['ndcg20'] = test_results[8].tolist()
        inference_end = time.time()

        for K in top_K:
            if epoch == 0:
                best_results['metric%d' % K][0] = metrics['hr%d' % K]
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
            else:
                if best_results['metric%d' % K][0] < metrics['hr%d' % K]:
                    best_results['metric%d' % K][0] = metrics['hr%d' % K]
                    best_results['epoch%d' % K][0] = epoch
                if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                    best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                    best_results['epoch%d' % K][1] = epoch
                if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                    best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                    best_results['epoch%d' % K][2] = epoch

        best_ = (valid_hr - best_hr) / best_hr + (valid_mrr - best_mrr) / best_mrr if best_hr > 0 and best_mrr > 0 else 1
        if best_ > 0:
            stop_number = 0
            best_hr, best_mrr = valid_hr, valid_mrr
            best_test_hr, best_test_mrr = test_hr, test_mrr
            best_test_list = test_results
            best_epoch = epoch
        else:
            stop_number += 1

        logging.info("Validation HR@20: %.4f, MRR@20: %.4f, NDCG@20: %.4f", valid_results[0], valid_results[1], valid_results[2])
        logging.info("Test HR@20: %.4f, MRR@20: %.4f, NDCG@20: %.4f", test_results[6], test_results[7], test_results[8])
        logging.info("Best HR so far: %.4f, Best MRR so far: %.4f, Best Epoch: %d", best_hr, best_mrr, best_epoch)

        for K in top_K:
            logging.info("Best @%d: HR: %.4f (ep%d), MRR: %.4f (ep%d), NDCG: %.4f (ep%d)",
                         K,
                         best_results['metric%d' % K][0], best_results['epoch%d' % K][0],
                         best_results['metric%d' % K][1], best_results['epoch%d' % K][1],
                         best_results['metric%d' % K][2], best_results['epoch%d' % K][2])

        # 记录耗时
        epoch_end = time.time()
        train_time = train_end - train_start
        inference_time = inference_end - inference_start
        total_time = epoch_end - epoch_start
        logging.info("Epoch %d finished: Train Time = %.2fs, Inference Time = %.2fs, Total Time = %.2fs",
                     epoch, train_time, inference_time, total_time)

    logging.info("Training finished.")
    logging.info("Best Validation HR@20: %.4f, MRR@20: %.4f", best_hr, best_mrr)
    logging.info("Best Test HR@20: %.4f, MRR@20: %.4f", best_test_hr, best_test_mrr)
    logging.info("Full best test result list: %s", str(best_test_list))






def parse_args():
    parser = argparse.ArgumentParser(description='Train BHSBR on Ascend NPU, CUDA, or CPU')
    parser.add_argument('--device', choices=['auto', 'npu', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--dataset', choices=['Applicances', 'Computers', 'Trivago', 'sample'], default='Applicances')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device, args.device_id)
    logging.info('Using device: %s', device)
    parameters = [100, 100, 100]
    file_name = args.dataset
    model_name = 'EMBSR'
    process_train, process_valid, process_test, item_vocab_size, max_position, adj = preprocess_data(file_name, args.batch_size)
    alpha = 12
    if file_name =='Applicances':
        lr_ = 0.0001
        drop_ = 0.2
    elif file_name == 'Computers':
        lr_ = 0.0003
        drop_ = 0.1
    else:
        lr_ = 0.001
        drop_ = 0.5
        model_name = 'EMBSR_trivago'
    np.random.seed(1)
    torch.manual_seed(1)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device.type == 'npu':
        torch.npu.manual_seed_all(1)
    nowtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = BASE_DIR / 'results'
    results_dir.mkdir(exist_ok=True)
    result_file = str(results_dir / ('result_%s_%s_%s' % (model_name, file_name, nowtime)))
    # process_train, process_valid, process_test, item_vocab_size, max_position = preprocess_data(file_name)
    run(process_train, process_valid, process_test, item_vocab_size, max_position, model_name, parameters, result_file, lr_, drop_, alpha, adj, device, args.epochs, args.patience)
    # process_train, process_valid, process_test = 0, 0, 0


if __name__ == '__main__':
    main()
