import time

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from datasets import build_dataloader, get_num_classes
from evaluate import predict_all
from train import build_criterion, build_model, build_optimizer, resolve_device, train

st.set_page_config(page_title="CV 对比实验", page_icon="🔍", layout="wide")

BACKBONES = ["linear", "mlp", "cnn"]
OPTIMIZERS = ["sgd", "adam"]
CRITERIA = ["cross_entropy", "mse"]
DEVICES = ["auto", "cuda", "cpu", "mps"]
MAX_EXAMPLES = 12


@st.cache_resource(show_spinner="正在准备数据集（首次运行需要下载）…")
def load_data(dataset, batch_size):
    return build_dataloader(dataset, batch_size)


def to_display(image):
    """Undo the [-1, 1] normalization and return a viewable numpy image."""
    image = (image * 0.5 + 0.5).clamp(0, 1).numpy()
    if image.shape[0] == 1:
        return image[0]
    return np.transpose(image, (1, 2, 0))


def build_label(config, existing):
    base = (
        f"{config['dataset']}-{config['backbone']}-{config['optimizer']}"
        f"-{config['criterion']}-lr{config['lr']:g}"
    )
    label = base
    index = 2
    while label in existing:
        label = f"{base} #{index}"
        index += 1
    return label


def confusion_matrix(preds, labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        matrix[true, pred] += 1
    return matrix


def plot_confusion(matrix, num_classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(range(num_classes), fontsize=8)
    ax.set_yticklabels(range(num_classes), fontsize=8)
    plt.tight_layout()
    return fig


st.title("🔍 CV 模型对比实验")
st.caption("选一组超参数训练一次，再换一组，所有结果会自动叠在一起对比。")

st.session_state.setdefault("runs", [])

with st.sidebar:
    st.header("实验配置")
    dataset = st.selectbox("数据集", ["mnist", "cifar10", "cifar100"])
    backbone = st.selectbox("模型", BACKBONES, index=2)
    optimizer = st.selectbox("优化器", OPTIMIZERS)
    criterion = st.selectbox("损失函数", CRITERIA)
    lr = st.number_input("学习率", min_value=0.0001, max_value=1.0, value=0.0100, step=0.0010, format="%.4f")
    batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    epochs = st.slider("Epochs", 1, 20, 5)
    device_name = st.selectbox("设备", DEVICES)

    start = st.button("开始训练", type="primary", width="stretch")
    st.divider()
    clear = st.button("清空对比结果", width="stretch")

if clear:
    st.session_state.runs = []
    st.rerun()

config = {
    "dataset": dataset,
    "backbone": backbone,
    "optimizer": optimizer,
    "criterion": criterion,
    "lr": lr,
    "batch_size": batch_size,
    "epochs": epochs,
}

if "cifar100" == dataset and device_name in ("auto", "cpu"):
    st.warning("CIFAR100 + CPU 训练会很慢，建议先用 MNIST 或 CIFAR10 演示。")

# --- Dataset preview -------------------------------------------------------
try:
    train_loader, test_loader = load_data(dataset, batch_size)
except Exception as exc:  # download / network problems
    st.error(f"数据集加载失败：{exc}")
    st.stop()

num_classes = get_num_classes(dataset)
sample_batch = next(iter(train_loader))
first_image = sample_batch[0][0]

st.subheader("① 数据长什么样")
st.caption(
    f"{dataset}：{len(train_loader.dataset)} 张训练图、{len(test_loader.dataset)} 张测试图，"
    f"共 {num_classes} 类，尺寸 {first_image.shape[1]}×{first_image.shape[2]}。"
)

preview = train_loader.dataset
columns = st.columns(8)
for i in range(8):
    image, label = preview[i]
    with columns[i]:
        st.image(to_display(image), caption=f"标签 {label}", width="stretch")

# --- Training --------------------------------------------------------------
st.subheader("② 训练模型")

if start:
    device = resolve_device(device_name)
    model = build_model(
        backbone,
        in_channels=first_image.shape[0],
        size=first_image.shape[1],
        classes=num_classes,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer_obj = build_optimizer(optimizer, model.parameters(), lr)
    criterion_obj = build_criterion(criterion)

    existing = [run["name"] for run in st.session_state.runs]
    run_name = build_label(config, existing)

    live = {"loss": [], "acc": []}
    progress_bar = st.progress(0.0, text="准备训练…")
    chart_slot = st.empty()

    def on_epoch_end(epoch, loss, acc):
        live["loss"].append(loss)
        live["acc"].append(acc)
        progress_bar.progress(
            (epoch + 1) / epochs,
            text=f"Epoch {epoch + 1}/{epochs} · loss {loss:.4f} · test acc {acc:.2%}",
        )
        frame = pd.DataFrame(
            {"训练 Loss": live["loss"], "测试准确率": live["acc"]},
            index=range(1, len(live["loss"]) + 1),
        )
        chart_slot.line_chart(frame, height=260)

    with st.status(f"在 {device} 上训练 {run_name}…", expanded=True):
        t0 = time.time()
        loss_history, acc_history = train(
            model,
            train_loader,
            test_loader,
            optimizer_obj,
            criterion_obj,
            epochs=epochs,
            device=device,
            on_epoch_end=on_epoch_end,
            progress=False,
        )
        seconds = time.time() - t0
        st.write(f"参数数量：{num_params:,} · 用时：{seconds:.1f}s")

        preds, labels, _ = predict_all(model, test_loader, device)
        matrix = confusion_matrix(preds, labels, num_classes)

        wrong = np.where(preds != labels)[0][:MAX_EXAMPLES]
        examples = []
        for idx in wrong:
            image, true_label = test_loader.dataset[int(idx)]
            examples.append((to_display(image), int(true_label), int(preds[idx])))

        st.session_state.runs.append(
            {
                "name": run_name,
                "config": dict(config),
                "loss": loss_history,
                "acc": acc_history,
                "params": num_params,
                "seconds": seconds,
                "confusion": matrix,
                "num_classes": num_classes,
                "examples": examples,
            }
        )

    progress_bar.progress(1.0, text="训练完成")
    st.success(f"{run_name} 完成，最终测试准确率 {acc_history[-1]:.2%}")
else:
    st.info("在左侧选好参数后点击「开始训练」。")

# --- Comparison ------------------------------------------------------------
runs = st.session_state.runs

if runs:
    st.subheader("③ 多组实验对比")

    summary = pd.DataFrame(
        [
            {
                "实验": run["name"],
                "数据集": run["config"]["dataset"],
                "模型": run["config"]["backbone"],
                "优化器": run["config"]["optimizer"],
                "损失": run["config"]["criterion"],
                "lr": run["config"]["lr"],
                "Epochs": run["config"]["epochs"],
                "参数量": run["params"],
                "最终准确率": run["acc"][-1],
                "用时(s)": round(run["seconds"], 1),
            }
            for run in runs
        ]
    )
    st.dataframe(
        summary.style.format({"最终准确率": "{:.2%}", "参数量": "{:,}", "lr": "{:g}"}),
        width="stretch",
        hide_index=True,
    )

    # Runs may have different epoch counts, so build per-run Series to align them
    acc_frame = pd.DataFrame(
        {
            run["name"]: pd.Series(run["acc"], index=range(1, len(run["acc"]) + 1))
            for run in runs
        }
    )
    acc_frame.index.name = "Epoch"
    loss_frame = pd.DataFrame(
        {
            run["name"]: pd.Series(run["loss"], index=range(1, len(run["loss"]) + 1))
            for run in runs
        }
    )
    loss_frame.index.name = "Epoch"

    left, right = st.columns(2)
    with left:
        st.caption("测试准确率")
        st.line_chart(acc_frame, height=320)
    with right:
        st.caption("训练 Loss")
        st.line_chart(loss_frame, height=320)

    st.subheader("④ 错误分析")
    selected = st.selectbox("选择一组实验", [run["name"] for run in runs], index=len(runs) - 1)
    run = next(item for item in runs if item["name"] == selected)

    left, right = st.columns([1, 1])
    with left:
        if run["num_classes"] <= 20:
            st.caption("混淆矩阵（横轴=预测，纵轴=真实）")
            st.pyplot(plot_confusion(run["confusion"], run["num_classes"]))
        else:
            st.caption("最易混淆的类别对")
            matrix = run["confusion"].copy()
            np.fill_diagonal(matrix, 0)
            pairs = sorted(
                [(matrix[i, j], i, j) for i in range(run["num_classes"]) for j in range(run["num_classes"])],
                reverse=True,
            )[:10]
            st.dataframe(
                pd.DataFrame(pairs, columns=["次数", "真实类别", "被预测为"]),
                hide_index=True,
                width="stretch",
            )
    with right:
        st.caption("被分错的样本")
        if not run["examples"]:
            st.write("这一组没有错分样本。")
        else:
            columns = st.columns(3)
            for i, (image, true_label, pred_label) in enumerate(run["examples"]):
                with columns[i % 3]:
                    st.image(
                        image,
                        caption=f"真实 {true_label} → 预测 {pred_label}",
                        width="stretch",
                    )
else:
    st.info("还没有实验记录。训练一组之后，这里会出现对比表格和曲线。")
