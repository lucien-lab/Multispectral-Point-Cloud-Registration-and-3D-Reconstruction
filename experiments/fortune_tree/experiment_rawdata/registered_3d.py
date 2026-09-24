"""两张配准结果图（M_ref / M_joint）的 3D 视角交付

产出两件东西：
  1. visualization/registered_3d_interactive.html  —— Plotly 可自由旋转的 3D 交互图
     （鼠标左键拖动旋转、滚轮缩放、双击复位；双击 HTML 即可打开，不需要 Python 环境）
  2. visualization/registered_multiview.png        —— 2 行 × 4 方位的静态多角度图
     （方位 0/90/180/270°，俯仰 20°，两方法每列同相机、同一坐标范围）

用法：python registered_3d.py
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
RGB_BANDS_1BASED = (957, 730, 552)
GAMMA = 1.55            # 白底压暗增饱和（与六视角图的白底版一致）
PT_MERGED = 2.0
AZIMUTHS = [0, 90, 180, 270]
ELEV = 20

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "STSong", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 载入与配色
data = {}
for d in DEGS:
    z = np.load(CACHE / f"frame_{d:03d}.npz")
    refl = z["refl"].astype(float)
    idx = [b - 1 for b in RGB_BANDS_1BASED]
    data[d] = {"xyz": z["xyz"].astype(float),
               "rgb": np.clip(refl[:, idx] * 4.0, 0, 1)}

stack = np.vstack([data[d]["rgb"] for d in DEGS])
scale = float(np.percentile(stack.max(axis=1), 95)) or 1.0
for d in DEGS:
    data[d]["vis"] = np.clip(data[d]["rgb"] / scale, 0, 1) ** GAMMA
print(f"全局 p95 亮度 {scale:.4f} → gamma {GAMMA} 后亮度均值 "
      f"{np.vstack([data[d]['vis'] for d in DEGS]).mean():.4f}")

reg = np.load(CACHE / "registration.npz")


def merged(key):
    P, C = [], []
    for d in DEGS:
        T = np.eye(4) if d == 0 else reg[f"{key}_{d}"]
        P.append((T[:3, :3] @ data[d]["xyz"].T).T + T[:3, 3])
        C.append(data[d]["vis"])
    return np.vstack(P), np.vstack(C)


m_ref, c_ref = merged("T_ref")
m_joint, c_joint = merged("T_joint")
print(f"M_ref {len(m_ref)} 点 | M_joint {len(m_joint)} 点")

# 统一坐标范围（两方法共用，便于直接对比）
# 用 0.2–99.8 百分位而非 min/max：配准残差产生的少量飞点会把包围盒撑大、使主体变小
allpts = np.vstack([m_ref, m_joint])
qlo, qhi = np.percentile(allpts, [0.2, 99.8], axis=0)
pad = (qhi - qlo) * 0.05
LIM = [(qlo[i] - pad[i], qhi[i] + pad[i]) for i in range(3)]
ASPECT = tuple(float(LIM[i][1] - LIM[i][0]) for i in range(3))   # 真实比例（等价 aspectmode=data）
print("等比范围(mm) " + " ".join("%s=%.1f" % (a, 1000 * (b[1] - b[0]))
                                  for a, b in zip("xyz", LIM)))

# ------------------------------------------------------------------ ① Plotly 交互 HTML
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def rgbstr(c):
        return ["rgb(%d,%d,%d)" % (int(255 * a), int(255 * b), int(255 * g))
                for a, b, g in c]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02,
                        specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("M_ref：纯几何基准",
                                        "M_joint：几何+光谱联合（本文方法）"))
    for col, (pts, cols, nm) in enumerate(
            [(m_ref, c_ref, "M_ref"), (m_joint, c_joint, "M_joint")], start=1):
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=nm,
            marker=dict(size=1.6, color=rgbstr(cols), opacity=0.9),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"),
            row=1, col=col)
    scene = dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                 aspectmode="data",
                 xaxis=dict(range=list(LIM[0])), yaxis=dict(range=list(LIM[1])),
                 zaxis=dict(range=list(LIM[2])),
                 camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)))
    fig.update_layout(template="plotly_white", height=900, width=1700,
                      title="发财树配准结果 3D 交互对比（左键拖动旋转 / 滚轮缩放 / 双击复位）"
                            "<br><sup>颜色由每点 2048 波段光谱转出；两图同一坐标比例（aspectmode=data）</sup>",
                      margin=dict(l=0, r=0, t=90, b=0),
                      legend=dict(orientation="h", y=1.06, x=0.42))
    fig.update_scenes(scene)
    p_html = OUT / "registered_3d_interactive.html"
    fig.write_html(str(p_html), include_plotlyjs="cdn")
    print(f"[保存] {p_html}  ({p_html.stat().st_size/1e6:.2f} MB，双击浏览器打开即可旋转)")
except Exception as e:                                    # pragma: no cover
    print(f"[跳过] Plotly 交互图生成失败：{type(e).__name__}: {e}")

# ------------------------------------------------------------------ ② 静态多角度图
fig = plt.figure(figsize=(16.5, 8.8), facecolor="white")
gs = GridSpec(2, 4, figure=fig, left=0.058, right=0.995, top=0.885, bottom=0.078,
              wspace=0.02, hspace=0.08)
rows = [(m_ref, c_ref, "M_ref（纯几何基准）"),
        (m_joint, c_joint, "M_joint（本文方法）")]
for r, (pts, cols, lab) in enumerate(rows):
    for c, az in enumerate(AZIMUTHS):
        ax = fig.add_subplot(gs[r, c], projection="3d", facecolor="white")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=PT_MERGED, c=cols,
                   marker=".", linewidths=0, depthshade=False)
        ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_zlim(*LIM[2])
        ax.view_init(ELEV, az)
        ax.set_box_aspect(ASPECT, zoom=1.02)
        ax.set_axis_off()
        if r == 0:
            ax.set_title(f"方位 {az}°", fontsize=13, color="0.15", pad=2)
    # 行侧标用图级文字，避免与相邻子图重叠
    yc = 0.74 if r == 0 else 0.30
    fig.text(0.020, yc, lab, rotation=90, ha="center", va="center",
             fontsize=13, color="0.10")
fig.suptitle("两种方法配准结果的 3D 多角度视图（方位 0°/90°/180°/270°，俯仰 20°）",
             fontsize=17, color="0.08", y=0.965)
fig.text(0.5, 0.048,
         "每行同一方法、每列同一相机与同一坐标范围（三轴按真实比例显示）；"
         "M_ref 为 FPFH+RANSAC+point-to-plane ICP 纯几何基准，M_joint 为几何+光谱联合（本文方法）",
         fontsize=10.5, color="0.38", ha="center")
fig.text(0.5, 0.018,
         "颜色由每点 2048 波段光谱转出（957/730/552 反射率 ×4，全局 p95 拉伸 + gamma 1.55）；"
         "6 视角全部点，各 25218 点；坐标范围按 0.2–99.8 百分位取，避免少量配准飞点把主体撑小",
         fontsize=10.5, color="0.38", ha="center")
p_png = OUT / "registered_multiview.png"
fig.savefig(p_png, dpi=150, facecolor="white")
plt.close(fig)
print(f"[保存] {p_png}")
