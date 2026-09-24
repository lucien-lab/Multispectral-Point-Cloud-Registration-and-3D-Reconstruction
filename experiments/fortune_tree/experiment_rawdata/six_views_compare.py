"""六视角彩色点云 + 两种方法配准结果对比图（3D / RGB）

上排 (a)-(f)：6 个视角各自的带颜色点云（未配准，保持扫描仪自身坐标系）
下排 (g)-(h)：M_ref（纯几何基准）与 M_joint（几何+光谱联合，本文方法）的配准结果

颜色：由 2048 波段光谱转 RGB —— 取标定波段 957/730/552（1-based）
      的标定反射率 ×4 并截断到 [0,1]，再全局 p95 拉伸 + gamma 0.8。
      全局只算一个拉伸因子，保证 8 个子图颜色可比。

用法：python six_views_compare.py
输出：visualization/six_views_registration_compare.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "visualization"

DEGS = [0, 60, 120, 180, 240, 300]
RGB_BANDS_1BASED = (957, 730, 552)   # R,G,B（MATLAB 1-based，与 convert_to_txt 一致）
ELEV, AZIM = 18, -60                 # 8 个子图统一相机，保证可比
PT_RAW, PT_MERGED = 1.7, 0.8          # 点径默认值（实际由 draw() 按底色传入）

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "STSong", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def load_frame(deg):
    z = np.load(CACHE / f"frame_{deg:03d}.npz")
    return z["xyz"].astype(float), z["refl"].astype(float)


def rgb_from_spectra(refl):
    """标定反射率 → RGB。refl 已是 (raw-dark)/(white-dark)，此处只需 ×4。"""
    idx = [b - 1 for b in RGB_BANDS_1BASED]
    return np.clip(refl[:, idx] * 4.0, 0, 1)


# ---------------------------------------------------------------- ① 读 6 帧并统一增强
data = {}
for d in DEGS:
    xyz, refl = load_frame(d)
    data[d] = {"xyz": xyz, "rgb": rgb_from_spectra(refl)}

stack = np.vstack([data[d]["rgb"] for d in DEGS])
scale = float(np.percentile(stack.max(axis=1), 95)) or 1.0
print(f"6 帧合计 {len(stack)} 点；RGB(×4) 亮度均值 {stack.mean():.4f}；"
      f"全局 p95 亮度 {scale:.4f}")
for d in DEGS:
    data[d]["base"] = np.clip(data[d]["rgb"] / max(scale, 1e-6), 0, 1)
base_all = np.vstack([data[d]["base"] for d in DEGS])
print(f"p95 拉伸后亮度均值 {base_all.mean():.4f}  通道均值 {base_all.mean(axis=0).round(3)}"
      f"  （绿植应绿通道偏高）；gamma 按底色在 draw() 里施加")

# ---------------------------------------------------------------- ② 两方法配准结果
reg = np.load(CACHE / "registration.npz")


def merged(key):
    P, C = [], []
    for d in DEGS:
        T = np.eye(4) if d == 0 else reg[f"{key}_{d}"]
        P.append((T[:3, :3] @ data[d]["xyz"].T).T + T[:3, 3])
        C.append(data[d]["base"])
    P, C = np.vstack(P), np.vstack(C)
    u = np.unique(np.round(P, 6), axis=0, return_index=True)[1]
    return P[u], C[u], len(P) - len(u)


m_ref, c_ref, dup_ref = merged("T_ref")
m_joint, c_joint, dup_joint = merged("T_joint")
print(f"M_ref  合并后 {len(m_ref)} 点（坐标去重 {dup_ref} 点）  "
      f"bbox z[{m_ref[:,2].min():+.3f},{m_ref[:,2].max():+.3f}]")
print(f"M_joint 合并后 {len(m_joint)} 点（坐标去重 {dup_joint} 点）  "
      f"bbox z[{m_joint[:,2].min():+.3f},{m_joint[:,2].max():+.3f}]")
for d in DEGS:
    c = data[d]["xyz"].mean(0)
    print(f"  {d:3d}° 质心 ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
          f"z[{data[d]['xyz'][:,2].min():+.3f},{data[d]['xyz'][:,2].max():+.3f}]")


def lim(arrays, pad=0.25):
    a = np.vstack(arrays)
    lo, hi = a.min(0), a.max(0)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    r = r.max() / 2 * (1 + pad)
    return [(c[i] - r, c[i] + r) for i in range(3)]


lim_single = lim([data[d]["xyz"] for d in DEGS])
lim_merged = lim([m_ref, m_joint])

# ---------------------------------------------------------------- ③ 出图
def draw(bg: str, fg: str, cap: str, gamma: float, pt_raw: float, pt_merged: float,
         path: Path) -> None:
    """bg 底色；fg 标题色；cap 图注色；gamma 色调映射（<1 提亮，>1 压暗增饱和）。

    白底需压暗（gamma>1）才能看清，黑底需提亮（gamma<1）才不会糊成一片黑。
    """
    vis = {d: data[d]["base"] ** gamma for d in DEGS}
    stack_v = np.vstack([vis[d] for d in DEGS])
    print(f"  [{bg}] gamma={gamma} → 亮度均值 {stack_v.mean():.4f} "
          f"通道 {stack_v.mean(axis=0).round(3)}")
    fig = plt.figure(figsize=(16.5, 15.2), facecolor=bg)
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.0, 1.32],
                  left=0.012, right=0.988, top=0.935, bottom=0.115,
                  wspace=0.04, hspace=0.07)

    # 上两排：(a)-(f) 六个视角，每排 3 个（每格宽 1/3 图宽）
    for i, d in enumerate(DEGS):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2], projection="3d", facecolor=bg)
        ax.scatter(data[d]["xyz"][:, 0], data[d]["xyz"][:, 1], data[d]["xyz"][:, 2],
                   s=pt_raw, c=vis[d], marker=".", linewidths=0, depthshade=False)
        ax.set_xlim(*lim_single[0]); ax.set_ylim(*lim_single[1]); ax.set_zlim(*lim_single[2])
        ax.view_init(ELEV, AZIM)
        ax.set_box_aspect((1, 1, 1), zoom=1.05)
        ax.set_axis_off()
        ax.set_title(f"({chr(97+i)}) {d}° 视角", color=fg, fontsize=13, pad=0)

    # 第三排：(g) M_ref 与 (h) M_joint，各占半宽
    for ax, (pts, cols, lab) in zip(
            [fig.add_subplot(gs[2, 0:3], projection="3d", facecolor=bg),
             fig.add_subplot(gs[2, 3:6], projection="3d", facecolor=bg)],
            [(m_ref, c_ref, "(g) M_ref：纯几何基准（FPFH+RANSAC+point-to-plane ICP）"),
             (m_joint, c_joint, "(h) M_joint：几何+光谱联合（本文方法）")]):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=pt_merged, c=cols ** gamma,
                   marker=".", linewidths=0, depthshade=False)
        ax.set_xlim(*lim_merged[0]); ax.set_ylim(*lim_merged[1]); ax.set_zlim(*lim_merged[2])
        ax.view_init(ELEV, AZIM)
        ax.set_box_aspect((1, 1, 1), zoom=1.05)
        ax.set_axis_off()
        ax.set_title(lab, color=fg, fontsize=14, pad=0)

    fig.suptitle("发财树多光谱点云：六个视角的彩色点云（a-f）与两种方法的配准结果（g-h）",
                 color=fg, fontsize=17, y=0.972)
    fig.text(0.5, 0.082,
             f"颜色由每点 2048 波段光谱转出（957/730/552 → 650.1/560.3/490.4 nm 反射率 ×4，"
             f"全局 p95 拉伸 + gamma {gamma:g}）；(a)-(f) 为未配准的单视角点云（每视角 4152～4495 点），"
             "(g)-(h) 为配准后合并点云（6 视角全部点，共 25218 点）",
             color=cap, fontsize=10.5, ha="center", va="center")
    fig.text(0.5, 0.048,
             "8 个子图同一相机（俯仰 18°、方位 −60°）；(g)-(h) 共用同一坐标范围以便直接对比"
             "　　5 帧平均（20 mm 口径）—— M_ref：fitness 0.9824 / RMSE 7.03 mm / SAM 10.82°　"
             "M_joint：fitness 0.9602 / RMSE 7.48 mm / SAM 9.08°",
             color=cap, fontsize=10.5, ha="center", va="center")

    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)
    print(f"[保存] {path}")


# 白底为主交付（gamma 1.55 压暗增饱和，暗点云在白底上对比强烈）；黑底版保留备用
draw("white", "0.08", "0.38", gamma=1.55, pt_raw=4.5, pt_merged=2.0,
     path=OUT / "six_views_registration_compare.png")
draw("black", "white", "0.78", gamma=0.8, pt_raw=1.3, pt_merged=0.55,
     path=OUT / "six_views_registration_compare_darkbg.png")
