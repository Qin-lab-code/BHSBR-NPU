"""MindSpore implementation of BHSBR for Ascend training."""

import math

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import Normal


class HGNNConv(nn.Cell):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.02), (hidden_size, hidden_size), ms.float32)
        )
        self.bias = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.02), (hidden_size,), ms.float32)
        )

    def construct(self, x, graph):
        return ops.matmul(graph, ops.matmul(x, self.weight) + self.bias)


class HGNN(nn.Cell):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.conv1 = HGNNConv(hidden_size)
        self.conv2 = HGNNConv(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, graph):
        first = self.dropout(self.conv1(x, graph))
        second = self.dropout(self.conv2(first, graph))
        return (first + second) / 2.0


class ItemConv(nn.Cell):
    """Global item propagation without constructing an Ascend sparse tensor."""

    def __init__(self, adjacency, layers=2, embedding_size=100):
        super().__init__()
        adjacency = adjacency.tocoo()
        self.rows = Tensor(adjacency.row.astype("int32"), ms.int32)
        self.cols = Tensor(adjacency.col.astype("int32"), ms.int32)
        self.values = Tensor(adjacency.data.astype("float32"), ms.float32)
        self.layers = nn.CellList(
            [nn.Dense(embedding_size, embedding_size, has_bias=False) for _ in range(layers)]
        )
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

    def construct(self, embeddings):
        propagated = embeddings
        outputs = [ops.expand_dims(embeddings, 0)]
        scatter_indices = ops.expand_dims(self.rows, 1)
        for layer in self.layers:
            propagated = layer(propagated)
            messages = ops.gather(propagated, self.cols, 0)
            messages = messages * ops.expand_dims(self.values, 1)
            propagated = ops.tensor_scatter_add(ops.zeros_like(propagated), scatter_indices, messages)
            outputs.append(ops.expand_dims(self.normalize(propagated), 0))
        return ops.mean(ops.concat(outputs, axis=0), axis=0)


class HierarchicalGNN(nn.Cell):
    def __init__(self, embedding_size, step=1):
        super().__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.w_ih = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.1), (embedding_size * 3, embedding_size * 2), ms.float32)
        )
        self.w_hh = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.1), (embedding_size * 3, embedding_size), ms.float32)
        )
        self.b_ih = ms.Parameter(ms.common.initializer.initializer(Normal(0.1), (embedding_size * 3,), ms.float32))
        self.b_hh = ms.Parameter(ms.common.initializer.initializer(Normal(0.1), (embedding_size * 3,), ms.float32))
        self.b_in = ms.Parameter(ms.common.initializer.initializer(Normal(0.1), (embedding_size,), ms.float32))
        self.b_out = ms.Parameter(ms.common.initializer.initializer(Normal(0.1), (embedding_size,), ms.float32))
        self.edge_in = nn.Dense(embedding_size * 2, embedding_size)
        self.edge_out = nn.Dense(embedding_size * 2, embedding_size)

    def cell(self, adjacency, hidden, macro, micro, micro_len):
        batch, edges, actions, dim = micro.shape
        flat = micro.reshape((batch * edges, actions, dim))
        gru_output, _ = self.gru(flat)
        last_index = ops.maximum(micro_len.reshape((-1,)).astype(ms.int32) - 1, 0)
        gather_index = last_index.reshape((-1, 1, 1))
        gather_index = ops.tile(gather_index, (1, 1, dim))
        micro_hidden = ops.gather_d(gru_output, 1, gather_index).squeeze(1)
        micro_hidden = micro_hidden.reshape((batch, edges, dim))

        macro_hidden = ops.concat((macro, micro_hidden), axis=2)
        incoming = self.edge_in(macro_hidden)
        outgoing = self.edge_out(macro_hidden)
        inputs = ops.concat(
            (
                ops.matmul(adjacency[:, :, :edges], incoming) + self.b_in,
                ops.matmul(adjacency[:, :, edges:], outgoing) + self.b_out,
            ),
            axis=2,
        )
        gi = ops.matmul(inputs, self.w_ih.T) + self.b_ih
        gh = ops.matmul(hidden, self.w_hh.T) + self.b_hh
        i_r, i_i, i_n = ops.split(gi, split_size_or_sections=self.embedding_size, axis=2)
        h_r, h_i, h_n = ops.split(gh, split_size_or_sections=self.embedding_size, axis=2)
        reset = ops.sigmoid(i_r + h_r)
        update = ops.sigmoid(i_i + h_i)
        candidate = ops.tanh(i_n + reset * h_n)
        return (1.0 - update) * hidden + update * candidate

    def construct(self, adjacency, hidden, macro, micro, micro_len):
        for _ in range(self.step):
            hidden = self.cell(adjacency, hidden, macro, micro, micro_len)
        return hidden


class BHSBR(nn.Cell):
    def __init__(
        self, n_items, adjacency, n_actions, n_positions, n_action_pairs,
        item_dim=100, action_dim=100, position_dim=100, hidden_dim=100,
        dropout_rate=0.0, alpha=12.0, step=1,
    ):
        super().__init__()
        self.item_embeddings = nn.Embedding(n_items, item_dim, padding_idx=0)
        self.action_embeddings = nn.Embedding(n_actions, action_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(n_positions, position_dim, padding_idx=0)
        self.action_pair_embeddings = nn.Embedding(n_action_pairs, action_dim, padding_idx=0)
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.step = step
        self.alpha = alpha

        self.item_graph = ItemConv(adjacency, layers=2, embedding_size=item_dim)
        self.hgnn = HGNN(item_dim)
        self.gnn = HierarchicalGNN(hidden_dim)
        self.w_q1 = nn.Dense(hidden_dim, hidden_dim)
        self.w_k1 = nn.Dense(hidden_dim, hidden_dim)
        self.w_q2 = nn.Dense(hidden_dim, hidden_dim)
        self.w_k2 = nn.Dense(hidden_dim, hidden_dim)
        self.highway = nn.Dense(hidden_dim * 2, hidden_dim)
        self.attn_weights = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.1), (hidden_dim, hidden_dim), ms.float32)
        )
        self.attn = ms.Parameter(
            ms.common.initializer.initializer(Normal(0.1), (1, hidden_dim), ms.float32)
        )
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

    def update_items(self, hidden, star, adjacency, macro, micro, micro_len):
        hidden = self.gnn(adjacency, hidden, macro, micro, micro_len)
        alpha = ops.bmm(self.w_q1(hidden), ops.expand_dims(self.w_k1(star), 2))
        alpha = alpha / math.sqrt(self.hidden_dim)
        new_hidden = (1.0 - alpha) * hidden + alpha * ops.expand_dims(star, 1)
        beta = ops.bmm(self.w_k2(new_hidden), ops.expand_dims(self.w_q2(star), 2))
        beta = ops.softmax(beta / math.sqrt(self.hidden_dim), axis=1)
        new_star = ops.bmm(ops.transpose(beta, (0, 2, 1)), new_hidden)
        return new_hidden, new_star

    def build_hypergraph(self, actions):
        valid = actions > 0
        indices = ops.maximum(actions.astype(ms.int32) - 1, 0)
        counts = ops.one_hot(indices, self.n_actions, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        incidence = ops.sum(counts * ops.expand_dims(valid.astype(ms.float32), -1), dim=1) + 0.0001
        dv = ops.sum(incidence, dim=1)
        de = ops.sum(incidence, dim=0)
        return ops.diag(1.0 / dv) @ incidence @ ops.diag(1.0 / de) @ incidence.T

    def construct(self, items, adjacency, alias, seq_len, macro_items, micro_actions, micro_len, actions, pairs, positions):
        global_items = self.item_graph(self.item_embeddings.embedding_table)
        sequence_size = alias.shape[1] + 1
        actions = actions[:, :sequence_size]
        positions = positions[:, :sequence_size]
        pairs = pairs[:, :sequence_size, :sequence_size]

        hidden = self.item_embeddings(items)
        initial = hidden
        macro_embedding = self.item_embeddings(macro_items)
        micro_embedding = self.action_embeddings(micro_actions)
        length = ops.sum((items > 0).astype(ms.float32), dim=1, keepdim=True)
        star = ops.sum(hidden, dim=1) / ops.maximum(length, 1.0)

        for _ in range(self.step):
            hidden, star = self.update_items(hidden, star, adjacency, macro_embedding, micro_embedding, micro_len)
            if star.ndim == 3:
                star = star.squeeze(1)
        gate = ops.sigmoid(self.highway(ops.concat((initial, hidden), axis=2)))
        hidden = gate * initial + (1.0 - gate) * hidden

        safe_alias = ops.maximum(alias.astype(ms.int32), 0)
        gather_index = ops.expand_dims(safe_alias, 2)
        gather_index = ops.tile(gather_index, (1, 1, self.hidden_dim))
        sequence_hidden = ops.gather_d(hidden, 1, gather_index)
        sequence_hidden = ops.concat((ops.expand_dims(star, 1), sequence_hidden), axis=1)
        sequence_input = sequence_hidden + self.action_embeddings(actions)
        last_index = seq_len.astype(ms.int32).reshape((-1, 1, 1))
        last_index = ops.tile(last_index, (1, 1, self.hidden_dim))
        last_hidden = ops.gather_d(sequence_input, 1, last_index).squeeze(1)

        hgnn_outputs = []
        for batch_index in range(macro_items.shape[0]):
            count = int(ops.sum((macro_items[batch_index] > 0).astype(ms.int32)).asnumpy())
            count = max(count, 1)
            selected = ops.gather(global_items, macro_items[batch_index, :count].astype(ms.int32), 0)
            graph = self.build_hypergraph(micro_actions[batch_index, :count])
            hgnn_outputs.append(ops.mean(self.hgnn(selected, graph), axis=0))
        hgnn_hidden = ops.stack(hgnn_outputs, axis=0)

        combined = ops.stack((last_hidden, hgnn_hidden), axis=0)
        weights = ops.sum(ops.matmul(combined, self.attn_weights) * self.attn, dim=-1)
        weights = ops.expand_dims(ops.softmax(weights, axis=0), -1)
        session = ops.sum(combined * weights, dim=0)
        session = self.normalize(session)
        candidates = self.normalize(self.item_embeddings.embedding_table[1:])
        return self.alpha * ops.matmul(session, candidates.T)
