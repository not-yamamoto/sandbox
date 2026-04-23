from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import networkx as nx
import matplotlib.pyplot as plt


Task = Literal["node", "edge", "graph"]

NODE = "n"
EDGE = "e"


# ============================================================
# 1) グラフ生成（固定グラフ + ノード時系列） + タスク別ラベル生成
# ============================================================
@dataclass(frozen=True)
class SyntheticData:
    g_nx: nx.DiGraph
    src: np.ndarray          # (E,)
    dst: np.ndarray          # (E,)
    x: np.ndarray            # (T, N, F) 入力（時系列）
    y_node: np.ndarray       # (T, N, 1) 次時刻の node target
    y_edge: np.ndarray       # (T, E, 1) 次時刻の edge target
    y_graph: np.ndarray      # (T, 1)    次時刻の graph target


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
    src = np.array([u for u, v in edges], dtype=np.int32)
    dst = np.array([v for u, v in edges], dtype=np.int32)

    # 近傍（生成過程用）
    nbrs = [[] for _ in range(num_nodes)]
    for u, v in edges:
        nbrs[u].append(v)

    # ノード時系列特徴
    X = np.zeros((timesteps, num_nodes, feat_dim), dtype=np.float32)
    X[0] = rng.normal(size=(num_nodes, feat_dim)).astype(np.float32)

    for t in range(1, timesteps):
        prev = X[t - 1].copy()
        diff = np.zeros_like(prev)
        for i in range(num_nodes):
            if not nbrs[i]:
                continue
            mean_n = prev[nbrs[i]].mean(axis=0)
            diff[i] = mean_n - prev[i]
        noise = 0.05 * rng.normal(size=prev.shape).astype(np.float32)
        X[t] = prev + 0.25 * diff + noise
        X[t, :, 0] = np.tanh(X[t, :, 0])  # 少し非線形

    # “予測したいもの”を定義（次時刻 t+1 の値を当てる）
    # node target: 次時刻の feature0
    y_node = X[1:, :, 0:1].astype(np.float32)       # (T-1, N, 1)
    x_in = X[:-1].astype(np.float32)                # (T-1, N, F)

    # edge target: 次時刻の |x0(src)-x0(dst)| （例：相互作用強度っぽいもの）
    x0_next = X[1:, :, 0]                            # (T-1, N)
    y_edge = np.abs(x0_next[:, src] - x0_next[:, dst])[..., None].astype(np.float32)  # (T-1, E, 1)

    # graph target: 次時刻の node target の平均（例：ウェハ平均膜厚みたいに）
    y_graph = y_node.mean(axis=1).astype(np.float32)  # (T-1, 1)

    return SyntheticData(
        g_nx=g_nx, src=src, dst=dst,
        x=x_in, y_node=y_node, y_edge=y_edge, y_graph=y_graph
    )


# ============================================================
# 2) GraphTensor化（固定構造で node feat だけ差し替え）
# ============================================================
def make_graphtensor(src: np.ndarray, dst: np.ndarray, node_feat: np.ndarray) -> tfgnn.GraphTensor:
    """
    node_feat: (N, F)
    """
    n = int(node_feat.shape[0])
    e = int(src.shape[0])
    return tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(features={}),
        node_sets={
            NODE: tfgnn.NodeSet.from_fields(
                sizes=tf.constant([n], tf.int32),
                features={"feat": tf.convert_to_tensor(node_feat, tf.float32)},
            )
        },
        edge_sets={
            EDGE: tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([e], tf.int32),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(NODE, tf.convert_to_tensor(src, tf.int32)),
                    target=(NODE, tf.convert_to_tensor(dst, tf.int32)),
                ),
                features={},
            )
        }
    )


def make_window_sample(
    src: np.ndarray,
    dst: np.ndarray,
    x_window: np.ndarray,  # (W, N, F)
) -> Tuple[tfgnn.GraphTensor, ...]:
    # 入力を (GraphTensor, ... GraphTensor) の固定長タプルにする
    return tuple(make_graphtensor(src, dst, x_window[k]) for k in range(x_window.shape[0]))


# ============================================================
# 3) Dataset（時系列窓 W をちゃんと使う）
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
) -> WindowedDataset:
    """
    各サンプル:
      inputs: (gt(t-W+1), ..., gt(t)) の W 個
      label:  y(t) （= 次時刻 t+1 を表すラベルが data 側で用意済み）
    ※ここでの t は data.x の時刻index。
    """
    x = data.x
    T, N, F = x.shape
    E = data.src.shape[0]
    if end is None:
        end = T
    assert window >= 2

    # ラベル配列
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
        raise ValueError(task)

    # GraphTensorSpec（1時刻分）
    sample_gt = make_graphtensor(data.src, data.dst, x[0])
    gt_spec = sample_gt.spec
    inp_spec = tuple(gt_spec for _ in range(window))

    # 生成器（batch_size=1前提で扱いやすく）
    def gen() -> Iterable[Tuple[Tuple[tfgnn.GraphTensor, ...], tf.Tensor]]:
        idxs = np.arange(max(start, window - 1), end)
        if shuffle:
            np.random.shuffle(idxs)
        for t in idxs:
            x_window = x[t - window + 1 : t + 1]  # (W, N, F)
            inputs = make_window_sample(data.src, data.dst, x_window)
            y = tf.convert_to_tensor(y_all[t], tf.float32)
            yield inputs, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(inp_spec, y_spec),
    )

    # まずは確実に動く優先で batch(1)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return WindowedDataset(ds=ds, window=window, num_nodes=N, num_edges=E)


# ============================================================
# 4) モデル（共通GNN backbone + task head切替 + 時系列エンコーダ）
# ============================================================
def build_gnn_backbone(hidden_dim: int = 64, num_layers: int = 2) -> tf.keras.Model:
    """
    入力: GraphTensor（1時刻）
    出力: ノード埋め込み h (total_nodes, hidden_dim)
    """
    inp = tf.keras.layers.Input(type_spec=tfgnn.GraphTensorSpec.from_piece_specs(
        context_spec=tfgnn.ContextSpec.from_field_specs({}),
        node_sets_spec={
            NODE: tfgnn.NodeSetSpec.from_field_specs(
                features_spec={"feat": tf.TensorSpec([None, None], tf.float32)},
                sizes_spec=tf.TensorSpec([None], tf.int32),
            )
        },
        edge_sets_spec={
            EDGE: tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={},
                sizes_spec=tf.TensorSpec([None], tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    source_node_set=NODE, target_node_set=NODE,
                    index_spec=tf.TensorSpec([None, None], tf.int32),
                ),
            )
        }
    ))

    gt = inp
    gt = tfgnn.keras.layers.MapFeatures(
        node_sets_fn={
            NODE: tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(hidden_dim),
            ])
        }
    )(gt)

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

    h = gt.node_sets[NODE]["feat"]  # (total_nodes, hidden_dim)
    return tf.keras.Model(inp, h, name="gnn_backbone")


def build_switchable_temporal_model(
    window: int,
    num_nodes: int,
    num_edges: int,
    task: Task,
    hidden_dim: int = 64,
    gnn_layers: int = 2,
    rnn_units: int = 64,
) -> tf.keras.Model:
    """
    inputs: (gt_{t-W+1}, ..., gt_t) のタプル
    1) 各時刻で GNN(backbone) -> node embedding
    2) node embedding列を RNNで時系列統合（per-node）
    3) taskに応じて node/edge/graph head を切替
    """
    backbone = build_gnn_backbone(hidden_dim=hidden_dim, num_layers=gnn_layers)

    inputs = tuple(tf.keras.layers.Input(type_spec=backbone.input_spec) for _ in range(window))
    # 上は少しトリッキーなので、正しくは backbone.input の type_spec を流用する形が安全
    # ただ Keras の内部都合で InputSpec が扱いにくいことがあるため、次の方法で作ります:
    inputs = tuple(tf.keras.layers.Input(type_spec=backbone.input.type_spec) for _ in range(window))

    # (W,) それぞれの GraphTensor を backbone に通して node embedding を得る
    hs = [backbone(inp_t) for inp_t in inputs]  # 各 (N, hidden_dim) ※batch=1前提で N固定
    # (W, N, H) に積む
    h_seq = tf.stack(hs, axis=0)  # (W, N, H)

    # per-node 時系列RNN: (N, W, H) にしてから GRU
    h_seq = tf.transpose(h_seq, perm=[1, 0, 2])  # (N, W, H)
    h_last = tf.keras.layers.GRU(rnn_units)(h_seq)  # (N, rnn_units)

    if task == "node":
        out = tf.keras.layers.Dense(1)(h_last)  # (N, 1)

    elif task == "edge":
        # エッジ毎に sender/receiver の埋め込みでスコア/回帰
        # 入力グラフは固定なので、最初の時刻入力から adjacency を取り出して使う
        # （値は同じなのでどの時刻でもOK）
        adj = inputs[0].edge_sets[EDGE].adjacency
        send = tf.gather(h_last, adj.source)  # (E, rnn_units)
        recv = tf.gather(h_last, adj.target)  # (E, rnn_units)
        pair = tf.concat([send, recv, send * recv, tf.abs(send - recv)], axis=-1)
        out = tf.keras.layers.Dense(1)(pair)  # (E, 1)

    elif task == "graph":
        # ノードをプールしてグラフ表現（batch=1想定でmean）
        g = tf.reduce_mean(h_last, axis=0, keepdims=True)  # (1, rnn_units)
        out = tf.keras.layers.Dense(1)(g)                  # (1, 1)
        out = tf.squeeze(out, axis=0)                      # (1,)

    else:
        raise ValueError(task)

    return tf.keras.Model(inputs=list(inputs), outputs=out, name=f"tfgnn_temporal_{task}")


# ============================================================
# 5) 学習 / 推論
# ============================================================
def compile_and_train(model: tf.keras.Model, ds: tf.data.Dataset, epochs: int = 10, lr: float = 1e-3) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    model.fit(ds, epochs=epochs)


@tf.function
def predict_one(model: tf.keras.Model, inputs: Tuple[tfgnn.GraphTensor, ...]) -> tf.Tensor:
    return model(inputs, training=False)


# ============================================================
# 6) 可視化（task別）
# ============================================================
def plot_node_pred(g_nx: nx.DiGraph, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
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
    plt.scatter(y_true, y_pred)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Node regression (True vs Pred)")
    plt.show()


def plot_edge_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
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


def plot_graph_pred(y_true: float, y_pred: float, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.bar(["true", "pred"], [y_true, y_pred])
    plt.title(title)
    plt.ylabel("value")
    plt.show()


# ============================================================
# Main: 生成 -> 学習 -> 推論 -> 可視化
# ============================================================
def main(task: Task = "node") -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    data = generate_synthetic(num_nodes=30, num_edges=80, feat_dim=4, timesteps=260, seed=0)

    window = 10
    train_end = int(data.x.shape[0] * 0.8)

    train_pack = build_windowed_dataset(
        data, window=window, task=task, start=0, end=train_end, shuffle=True
    )
    test_pack = build_windowed_dataset(
        data, window=window, task=task, start=train_end, end=None, shuffle=False
    )

    model = build_switchable_temporal_model(
        window=window,
        num_nodes=train_pack.num_nodes,
        num_edges=train_pack.num_edges,
        task=task,
        hidden_dim=64,
        gnn_layers=2,
        rnn_units=64,
    )
    model.summary()

    compile_and_train(model, train_pack.ds, epochs=12, lr=1e-3)

    # 推論：テストの先頭サンプルで評価・可視化
    # dsから1つ取り出す
    for (x_inputs, y_true) in test_pack.ds.take(1):
        # batch(1)なので外す
        x_inputs = tuple(t[0] for t in x_inputs)  # W個のGraphTensor
        y_true_np = y_true.numpy()[0]

        y_pred = predict_one(model, x_inputs).numpy()
        # node/edge は (N,1)/(E,1) で返る。graph は (1,)
        if task == "node":
            plot_node_pred(data.g_nx, y_true_np, y_pred, "Temporal TF-GNN (NODE)")
        elif task == "edge":
            plot_edge_pred(y_true_np, y_pred, "Temporal TF-GNN (EDGE)")
        elif task == "graph":
            plot_graph_pred(float(y_true_np[0]), float(y_pred[0]), "Temporal TF-GNN (GRAPH)")
        break


if __name__ == "__main__":
    # "node" / "edge" / "graph" を切り替えて実行
    main(task="node")
    # main(task="edge")
    # main(task="graph")
