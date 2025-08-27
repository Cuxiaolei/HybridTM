import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os

# 设置字体和样式，确保高质量输出
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5  # 稍粗线条提高清晰度
plt.rcParams['figure.dpi'] = 600  # 超高分辨率
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['lines.markersize'] = 0  # 无标记
plt.style.use('seaborn-v0_8-whitegrid')

# 高质量输出参数
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1


def generate_sample_data(num_epochs=100, num_models=7):
    """生成7个模型的示例数据"""
    np.random.seed(42)

    models_data = {}

    for i in range(num_models):
        model_name = f"Model {i + 1}"

        # 生成有区分度的miou数据
        base_miou = 0.45 + (i * 0.03)
        final_miou = 0.8 + (i * 0.02)
        if final_miou > 0.95:
            final_miou = 0.95
        miou = np.linspace(base_miou, final_miou, num_epochs)
        miou += np.random.normal(0, 0.007, num_epochs).cumsum() * 0.1
        miou = np.clip(miou, 0, 1)

        # 生成对应的oa数据
        base_oa = base_miou + 0.05 + (i * 0.02)
        final_oa = final_miou + 0.05 + (i * 0.01)
        if final_oa > 0.98:
            final_oa = 0.98
        oa = np.linspace(base_oa, final_oa, num_epochs)
        oa += np.random.normal(0, 0.005, num_epochs).cumsum() * 0.1
        oa = np.clip(oa, 0, 1)

        models_data[model_name] = {
            'miou': miou,
            'oa': oa
        }

    return models_data


def plot_miou_curve(models_data, num_epochs=100, save_path=None,
                    title="mIoU Convergence Curves"):
    """绘制mIoU曲线（支持7个模型）"""
    plt.figure(figsize=(8, 6))

    # 7个清晰区分的颜色
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2'  # 粉色
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历7个模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, metrics['miou'], label=model_name, color=color, linestyle='-')

    # 设置图表属性
    plt.title(title)
    plt.xlabel('Training Epochs')
    plt.ylabel('mIoU')
    plt.xlim(1, num_epochs)
    plt.ylim(0.4, 1.0)  # 适当调整y轴范围，突出差异
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pil_kwargs=dict(quality=95))
        print(f"mIoU图表已保存至: {save_path}")

    plt.show()


def plot_oa_curve(models_data, num_epochs=100, save_path=None,
                  title="OA Convergence Curves"):
    """绘制OA曲线（支持7个模型）"""
    plt.figure(figsize=(8, 6))

    # 与mIoU保持一致的7种颜色
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2'  # 粉色
    ]

    epochs = np.arange(1, num_epochs + 1)

    # 遍历7个模型绘制曲线
    for i, (model_name, metrics) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, metrics['oa'], label=model_name, color=color, linestyle='-')

    # 设置图表属性
    plt.title(title)
    plt.xlabel('Training Epochs')
    plt.ylabel('OA')
    plt.xlim(1, num_epochs)
    plt.ylim(0.5, 1.0)  # 适当调整y轴范围
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pil_kwargs=dict(quality=95))
        print(f"OA图表已保存至: {save_path}")

    plt.show()


def load_data_from_csv(file_path):
    """从CSV加载7个模型的数据"""
    df = pd.read_csv(file_path)

    models_data = {}

    # 获取所有模型名称
    model_columns = [col.split('_')[0] for col in df.columns if '_miou' in col]
    model_names = list(set(model_columns))

    # 确保读取到7个模型
    if len(model_names) != 7:
        print(f"警告: 检测到{len(model_names)}个模型，而不是预期的7个")

    for model in model_names:
        miou_col = f"{model}_miou"
        oa_col = f"{model}_oa"

        if miou_col in df.columns and oa_col in df.columns:
            models_data[model] = {
                'miou': df[miou_col].values,
                'oa': df[oa_col].values
            }

    return models_data, len(df)


def main():
    # 方式1：使用7个模型的示例数据
    # num_epochs = 100
    # models_data = generate_sample_data(num_epochs, num_models=7)

    # 方式2：从CSV加载7个模型的数据
    csv_path = "train_metrics.csv"  # 替换为你的CSV路径
    models_data, num_epochs = load_data_from_csv(csv_path)

    # 绘制图表
    plot_miou_curve(
        models_data,
        num_epochs,
        save_path="./miou_7_models.png",
        title="mIoU Convergence Curves (7 Models)"
    )

    plot_oa_curve(
        models_data,
        num_epochs,
        save_path="./oa_7_models.png",
        title="OA Convergence Curves (7 Models)"
    )


if __name__ == "__main__":
    main()
