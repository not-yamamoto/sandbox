from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import networkx as nx
import matplotlib.pyplot as plt


Task = Literal["node", "edge", "graph"]

NODE = "n"
EDGE = "e"


# ============================================================
# 1) 合成データ（固定グラフ + ノード時系列） + タスク別ラベル
# ============================================================
@dataclass(frozen=True)
class SyntheticData:
    g_nx: nx.DiGraph
    src: np.ndarray          # (E,)
    dst: np.ndarray          # (E,)
    x: np.ndarray            # (T, N, F)
    y_node: np.ndarray       # (T, N, 1)
    y_edge: np.ndarray       # (T, E, 1)
    y_graph: np.ndarray      # (T, 1)


def _build_random_digraph(num_nodes: int, num_edges: int, seed: int = 0) -> nx.DiGraph:
    g = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=True)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return nx.DiGraph(g)


def generate_synthetic(
    num_nodes: int = 30,
    num_edges: int = 80,
    feat_dim: int = 4,
    timesteps: int = 240,
    seed: int = 0,
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    g_nx = _build_random_digraph(num_nodes, num_edges, seed=seed)

    edges = list(g_nx.edges())
    src = np.array([u for u, _ in edges], dtype=np.int32)
    dst = np.array([v for _, v in edges], dtype=np.int32)

    nbrs = [[] for _ in range(num_nodes)]
    for u, v in edges:
        nbrs[u].append(v)

    X = np.zeros((timesteps, num_nodes, feat_dim), dtype=np.float32)
    X[0] = rng.normal(size=(num_nodes, feat_dim)).astype(np.float32)

    for t in range(1, timesteps):
        prev = X[t - 1]
        diff = np.zeros_like(prev)
        for i in range(num_nodes):
            if nbrs[i]:
                mean_n = prev[nbrs[i]].mean(axis=0)
                diff[i] = mean_n - prev[i]
        noise = 0.05 * rng.normal(size=prev.shape).astype(np.float32)
        X[t] = prev + 0.25 * diff + noise
        X[t, :, 0] = np.tanh(X[t, :, 0])

    # 次時刻を当てる（x_inがt、y_*がt+1）
    y_node = X[1:, :, 0:1].astype(np.float32)     # (T-1, N, 1)
    x_in = X[:-1].astype(np.float32)              # (T-1, N, F)

    x0_next = X[1:, :, 0]                         # (T-1, N)
    y_edge = np.abs(x0_next[:, src] - x0_next[:, dst])[..., None].astype(np.float32)  # (T-1, E, 1)

    y_graph = y_node.mean(axis=1).astype(np.float32)  # (T-1, 1)

    return SyntheticData(
        g_nx=g_nx, src=src, dst=dst,
        x=x_in, y_node=y_node, y_edge=y_edge, y_graph=y_graph
    )


# ============================================================
# 2) GraphTensor 化（固定構造 + node feat 差し替え）
# ============================================================
def make_graphtensor(src: np.ndarray, dst: np.ndarray, node_feat: np.ndarray) -> tfgnn.GraphTensor:
    n = int(node_feat.shape[0])
    e = int(src.shape[0])
    return tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(features={}),
        node_sets={
            NODE: tfgnn.NodeSet.from_fields(
                sizes=tf.constant([n], tf.int32),  # (1,)
                features={"feat": tf.convert_to_tensor(node_feat, tf.float32)},  # (N,F)
            )
        },
        edge_sets={
            EDGE: tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([e], tf.int32),  # (1,)
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(NODE, tf.convert_to_tensor(src, tf.int32)),  # (E,)
                    target=(NODE, tf.convert_to_tensor(dst, tf.int32)),  # (E,)
                ),
                features={},
            )
        }
    )


def make_window_sample(src: np.ndarray, dst: np.ndarray, x_window: np.ndarray) -> Tuple[tfgnn.GraphTensor, ...]:
    W = int(x_window.shape[0])
    return tuple(make_graphtensor(src, dst, x_window[k]) for k in range(W))


# ============================================================
# 3) Dataset（window）※ここで batch>1 対応
# ============================================================
@dataclass(frozen=True)
class WindowedDataset:
    ds: tf.data.Dataset
    window: int
    num_nodes: int
    num_edges: int


def build_windowed_dataset(
    data: SyntheticData,
    window: int = 8,
    task: Task = "node",
    start: int = 0,
    end: int | None = None,
    shuffle: bool = True,
    batch_size: int = 8,
) -> WindowedDataset:
    x = data.x
    T, N, F = x.shape
    E = int(data.src.shape[0])

    if end is None:
        end = T
    if window < 2:
        raise ValueError("window must be >= 2")
    if not (0 <= start <= end <= T):
        raise ValueError(f"Invalid range: start={start}, end={end}, T={T}")

    if task == "node":
        y_all = data.y_node
        y_spec = tf.TensorSpec(shape=(N, 1), dtype=tf.float32)
    elif task == "edge":
        y_all = data.y_edge
        y_spec = tf.TensorSpec(shape=(E, 1), dtype=tf.float32)
    elif task == "graph":
        y_all = data.y_graph
        y_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)
    else:
        raise ValueError(f"Unknown task: {task}")

    sample_gt = make_graphtensor(data.src, data.dst, x[0])
    gt_spec = sample_gt.spec
    inp_spec = tuple(gt_spec for _ in range(window))

    t0 = max(start, window - 1)
    t1 = end
    if t0 >= t1:
        raise ValueError("No samples in the given range.")

    def gen() -> Iterable[Tuple[Tuple[tfgnn.GraphTensor, ...], tf.Tensor]]:
        idxs = np.arange(t0, t1, dtype=np.int32)
        if shuffle:
            np.random.shuffle(idxs)
        for t in idxs:
            x_window = x[t - window + 1: t + 1]  # (W,N,F)
            inputs = make_window_sample(data.src, data.dst, x_window)
            y = tf.convert_to_tensor(y_all[t], tf.float32)
            yield inputs, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=(inp_spec, y_spec))
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return WindowedDataset(ds=ds, window=window, num_nodes=N, num_edges=E)


# ============================================================
# 4) モデル（batch対応）
# ============================================================
def build_gnn_backbone_from_spec(
    gt_spec: tfgnn.GraphTensorSpec,
    hidden_dim: int = 64,
    num_layers: int = 2,
) -> tf.keras.Model:
    """
    入力: batched GraphTensor（Bグラフ）
    出力: node embedding（RaggedTensor: (B, N_i, H)）
    """
    inp = tf.keras.layers.Input(type_spec=gt_spec)

    gt = tfgnn.keras.layers.MapFeatures(
        node_sets_fn={
            NODE: tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(hidden_dim),
            ])
        }
    )(inp)

    for _ in range(num_layers):
        gt = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                NODE: tfgnn.keras.layers.NodeSetUpdate(
                    edge_set_inputs={
                        EDGE: tfgnn.keras.layers.SimpleConv(
                            message_fn=tf.keras.layers.Dense(hidden_dim, activation="relu"),
                            reduce_type="sum",
                        )
                    },
                    next_state=tf.keras.Sequential([
                        tf.keras.layers.Dense(hidden_dim, activation="relu"),
                        tf.keras.layers.Dense(hidden_dim),
                    ]),
                )
            }
        )(gt)

    h_flat = gt.node_sets[NODE]["feat"]        # (sum_nodes, H)
    sizes = gt.node_sets[NODE].sizes           # (B,)
    h = tf.RaggedTensor.from_row_lengths(h_flat, sizes)  # (B, N_i, H)
    return tf.keras.Model(inp, h, name="gnn_backbone")


def build_switchable_temporal_model(
    gt_spec: tfgnn.GraphTensorSpec,
    window: int,
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    task: Task,
    hidden_dim: int = 64,
    gnn_layers: int = 2,
    rnn_units: int = 64,
) -> tf.keras.Model:
    """
    inputs: (gt_{t-W+1}, ..., gt_t) のタプル
      各 gt は batched GraphTensor（Bグラフ）
    出力:
      node:  (B,N,1)
      edge:  (B,E,1)
      graph: (B,1)
    """
    backbone = build_gnn_backbone_from_spec(gt_spec, hidden_dim=hidden_dim, num_layers=gnn_layers)
    inputs = tuple(tf.keras.layers.Input(type_spec=gt_spec) for _ in range(window))

    # 各時刻: Ragged (B, N_i, H) → 固定Nなら Dense (B,N,H)
    hs_dense = []
    for inp_t in inputs:
        h_ragged = backbone(inp_t)
        # 固定グラフ前提：全サンプルで N_i == num_nodes を期待
        tf.debugging.assert_equal(h_ragged.row_lengths(), tf.fill([tf.shape(h_ragged.row_lengths())[0]], num_nodes))
        hs_dense.append(h_ragged.to_tensor())  # (B,N,H)

    # (B, W, N, H)
    h_seq = tf.stack(hs_dense, axis=1)

    # per-node RNN: (B*N, W, H) にしてGRU → (B,N,U)
    B = tf.shape(h_seq)[0]
    H = tf.shape(h_seq)[3]
    h_seq_bn = tf.reshape(h_seq, [B * num_nodes, window, H])
    h_last_bn = tf.keras.layers.GRU(rnn_units)(h_seq_bn)  # (B*N, U)
    h_last = tf.reshape(h_last_bn, [B, num_nodes, rnn_units])  # (B,N,U)

    if task == "node":
        out = tf.keras.layers.Dense(1)(h_last)  # (B,N,1)

    elif task == "edge":
        src_t = tf.constant(src, dtype=tf.int32)  # (E,)
        dst_t = tf.constant(dst, dtype=tf.int32)  # (E,)

        send = tf.gather(h_last, src_t, axis=1)   # (B,E,U)
        recv = tf.gather(h_last, dst_t, axis=1)   # (B,E,U)
        pair = tf.concat([send, recv, send * recv, tf.abs(send - recv)], axis=-1)  # (B,E,4U)
        out = tf.keras.layers.Dense(1)(pair)      # (B,E,1)

    elif task == "graph":
        g = tf.reduce_mean(h_last, axis=1)        # (B,U)
        out = tf.keras.layers.Dense(1)(g)         # (B,1)

    else:
        raise ValueError(f"Unknown task: {task}")

    return tf.keras.Model(inputs=list(inputs), outputs=out, name=f"tfgnn_temporal_{task}_batched")


# ============================================================
# 5) 学習 / 推論
# ============================================================
def compile_and_train(model: tf.keras.Model, ds: tf.data.Dataset, epochs: int = 10, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    model.fit(ds, epochs=epochs)


def predict_batch(model: tf.keras.Model, inputs: Tuple[tfgnn.GraphTensor, ...]) -> tf.Tensor:
    return model(inputs, training=False)


# ============================================================
# 6) 可視化（最初のバッチの先頭サンプルだけ）
# ============================================================
def plot_node_pred_one_graph(g_nx: nx.DiGraph, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = np.abs(y_true - y_pred)

    pos = nx.spring_layout(g_nx, seed=0)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(g_nx, pos, alpha=0.25)
    sizes = 250 + 2200 * (err / (err.max() + 1e-6))
    nx.draw_networkx_nodes(g_nx, pos, node_size=sizes)
    topk = np.argsort(-err)[:8]
    labels = {int(i): f"T:{y_true[i]:+.2f}\nP:{y_pred[i]:+.2f}" for i in topk}
    nx.draw_networkx_labels(g_nx, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Node regression (True vs Pred)")
    plt.show()


def plot_edge_pred_one_graph(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = np.abs(y_true - y_pred)

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=30)
    plt.title(title + " | abs error histogram")
    plt.xlabel("|True - Pred|")
    plt.ylabel("count")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Edge regression (True vs Pred)")
    plt.show()


def plot_graph_pred_one_graph(y_true: float, y_pred: float, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.bar(["true", "pred"], [y_true, y_pred])
    plt.title(title)
    plt.ylabel("value")
    plt.show()


# ============================================================
# Main
# ============================================================
def main(task: Task = "node", batch_size: int = 8) -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    data = generate_synthetic(num_nodes=30, num_edges=80, feat_dim=4, timesteps=260, seed=0)

    window = 10
    T = data.x.shape[0]
    train_end = int(T * 0.8)

    # 「test入力にtrain過去が混ざらない」時系列評価
    test_start = train_end + window - 1
    if test_start >= T:
        raise ValueError("Not enough timesteps for the chosen split/window.")

    train_pack = build_windowed_dataset(
        data, window=window, task=task, start=0, end=train_end, shuffle=True, batch_size=batch_size
    )
    test_pack = build_windowed_dataset(
        data, window=window, task=task, start=test_start, end=None, shuffle=False, batch_size=batch_size
    )

    # batched spec は dataset の element_spec から取る（これが一番確実）
    # element_spec: ( (gt,...gt), y )
    batched_gt_spec = train_pack.ds.element_spec[0][0]

    model = build_switchable_temporal_model(
        gt_spec=batched_gt_spec,
        window=window,
        num_nodes=train_pack.num_nodes,
        src=data.src,
        dst=data.dst,
        task=task,
        hidden_dim=64,
        gnn_layers=2,
        rnn_units=64,
    )
    model.summary()

    compile_and_train(model, train_pack.ds, epochs=12, lr=1e-3)

    # 推論：テストの最初のバッチから「先頭の1グラフ」を可視化
    for (x_inputs, y_true) in test_pack.ds.take(1):
        y_pred = predict_batch(model, x_inputs).numpy()
        y_true_np = y_true.numpy()

        # バッチ先頭だけ取り出して表示
        if task == "node":
            plot_node_pred_one_graph(
                data.g_nx,
                y_true_np[0],      # (N,1)
                y_pred[0],         # (N,1)
                f"Temporal TF-GNN (NODE) | batch_size={batch_size}"
            )
        elif task == "edge":
            plot_edge_pred_one_graph(
                y_true_np[0],      # (E,1)
                y_pred[0],         # (E,1)
                f"Temporal TF-GNN (EDGE) | batch_size={batch_size}"
            )
        elif task == "graph":
            plot_graph_pred_one_graph(
                float(y_true_np[0, 0]),
                float(y_pred[0, 0]),
                f"Temporal TF-GNN (GRAPH) | batch_size={batch_size}"
            )
        break


if __name__ == "__main__":
    main(task="node", batch_size=16)
    # main(task="edge", batch_size=16)
    # main(task="graph", batch_size=16)
