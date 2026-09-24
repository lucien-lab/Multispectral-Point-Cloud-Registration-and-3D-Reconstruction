# -*- coding: utf-8 -*-
"""专利图 5（v2，按 vision 评审意见重做）

评审结论（opencode-go/deepseek-v4-flash-vision-exp）：
  P1 下排点标过小 → 发灰；上排点标过大 → 网点感。要求「下排黑像素/上排 ≈ 1.2~1.6」。
  P1 下排盆体轮廓在黑白版不可辨（体素去重削薄壳）。
  P0 三臂在 50 mm 格宽下肉眼不可分 → 需局部放大插图；并要求 (i) 标注"本发明方法"。
实测后定参：
  上排 单视角 s=3.4 → 黑像素 1.95%
  下排 体素 4 mm（6020 点）+ s=4.0 → 黑像素 2.80%（比值 1.44，落入 1.2~1.6）
  取景 q=0.5%、pad=1.5%；面板缩放 zoom=1.10（物体占格更大）
  下排每格加「树干-盆口交界处」3 倍放大插图（同窗口、同比例），并在原格用细黑线框标出窗口。
"""
from __future__ import annotations

import base64
import html
import json
import os
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
ELEV, AZIM = 20, -60
S_VIEW, S_MERGED = 3.4, 4.0
MERGED_VOXEL = 0.004
PAD, QUANT, ZOOM = 0.015, 0.5, 1.10
GAMMA, SAT, SCALE_PCT, LO_PCT, HI_PCT = 2.6, 1.9, 95.0, 0.02, 0.98
WIN_CENTER_Z_PCT = 40.0      # 放大窗口中心的竖直分位（树干-盆口交界处）
WIN_HALF = 0.05              # 放大窗口半边长（m）→ 0.10 m 立方
INSET_XYWH = (0.635, 0.015, 0.355, 0.355)   # 插图在面板内的相对位置与大小
OUTNAME = os.environ.get("OUTNAME", "fig5_v2")
ARM_LABELS = [("A 几何", "几何配准"),
              ("B 几何+RGB", "几何与可见光融合配准"),
              ("C 几何+多光谱(本发明)", "几何与多光谱融合配准（本发明方法）")]

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "STSong", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

POSE = np.load(HERE / "abc_poses.npz")
XYZ, RGB = {}, {}
for d in DEGS:
    z = np.load(CACHE / f"frame_{d:03d}.npz")
    XYZ[d] = z["xyz"].astype(float)
    idx = [b - 1 for b in RGB_BANDS_1BASED]
    RGB[d] = np.clip(z["refl"].astype(float)[:, idx] * 4.0, 0, 1)
scale = float(np.percentile(np.vstack([RGB[d] for d in DEGS]).max(axis=1), SCALE_PCT)) or 1.0


def enhance(c):
    v = np.clip(c / scale, 0, 1)
    v = np.clip((v - LO_PCT) / max(HI_PCT - LO_PCT, 1e-6), 0, 1) ** GAMMA
    lum = (0.2126 * v[:, 0] + 0.7152 * v[:, 1] + 0.0722 * v[:, 2])[:, None]
    return np.clip(lum + SAT * (v - lum), 0, 1)


VIS = {d: enhance(RGB[d]) for d in DEGS}
M_VIS_FULL = np.vstack([VIS[d] for d in DEGS])


def merged(arm, voxel=MERGED_VOXEL):
    out = []
    for d in DEGS:
        T = np.eye(4) if d == 0 else POSE[f"{arm}|{d}"]
        out.append((T[:3, :3] @ XYZ[d].T).T + T[:3, 3])
    P = np.vstack(out)
    if voxel:
        key = np.floor(P / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        idx = np.sort(idx)
        return P[idx], M_VIS_FULL[idx]
    return P, M_VIS_FULL


MC = {a[0]: merged(a[0]) for a in ARM_LABELS}
M = {k: v[0] for k, v in MC.items()}
MCV = {k: v[1] for k, v in MC.items()}
M_RAW = {a[0]: merged(a[0], voxel=None)[0] for a in ARM_LABELS}


def lim(points, q=QUANT, pad=PAD):
    a = np.vstack(points)
    lo, hi = np.percentile(a, [q, 100 - q], axis=0)
    p = (hi - lo) * pad
    return [(lo[i] - p[i], hi[i] + p[i]) for i in range(3)]


LIM_VIEW = lim([XYZ[d] for d in DEGS])
LIM_MERGED = lim(list(M.values()))
# 放大窗口：树干-盆口交界（以合并点云的竖直分位定位，窗口为 0.10 m 立方）
allm = np.vstack(list(M.values()))
c = np.array([(LIM_MERGED[0][0] + LIM_MERGED[0][1]) / 2,
              (LIM_MERGED[1][0] + LIM_MERGED[1][1]) / 2,
              np.percentile(allm[:, 2], WIN_CENTER_Z_PCT)])
LIM_WIN = [(c[k] - WIN_HALF, c[k] + WIN_HALF) for k in range(3)]
AXIS_XY = (0.0009, 1.2505)
LIM_WIN[0] = (AXIS_XY[0] - WIN_HALF, AXIS_XY[0] + WIN_HALF)
LIM_WIN[1] = (AXIS_XY[1] - WIN_HALF, AXIS_XY[1] + WIN_HALF)

VIEW_PANELS = [(XYZ[d], VIS[d], f"({'abcdef'[i]}) {d}°") for i, d in enumerate(DEGS)]
ARM_PANELS = [(M[a[0]], MCV[a[0]], f"({'ghi'[i]}) {a[1]}") for i, a in enumerate(ARM_LABELS)]


def style3d(ax, L, zoom=ZOOM):
    for k, st in enumerate("xyz"):
        getattr(ax, "set_%slim" % st)(*L[k])
    ax.view_init(ELEV, AZIM)
    ax.set_box_aspect(tuple(float(L[k][1] - L[k][0]) for k in range(3)), zoom=zoom)
    ax.set_axis_off()


def draw_window_box(ax):
    """在合并结果面板内用细黑线画出放大窗口（0.10 m 立方），便于读者定位。"""
    (x0, x1), (y0, y1), (z0, z1) = LIM_WIN
    xs, ys, zs = [x0, x1], [y0, y1], [z0, z1]
    for i in (0, 1):
        for j in (0, 1):
            ax.plot([xs[i]] * 2, [ys[j]] * 2, zs, color="black", lw=0.7)
            ax.plot(xs, [ys[j]] * 2, [zs[i]] * 2, color="black", lw=0.7)
            ax.plot([xs[i]] * 2, ys, [zs[j]] * 2, color="black", lw=0.7)


def compose(mono, path):
    fig = plt.figure(figsize=(15, 15.8), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, left=0.010, right=0.990, top=0.974, bottom=0.010,
                  wspace=0.015, hspace=0.03)
    for i, (pts, cols, lab) in enumerate(VIEW_PANELS):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d", facecolor="white")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=S_VIEW, c=("black" if mono else cols),
                   marker=".", linewidths=0, depthshade=False, antialiased=not mono)
        style3d(ax, LIM_VIEW)
        ax.set_title(lab, fontsize=15 if mono else 14, color="black" if mono else "0.10", pad=-6)
    p0 = None
    for i, (pts, cols, lab) in enumerate(ARM_PANELS):
        k = 6 + i
        ax = fig.add_subplot(gs[k // 3, k % 3], projection="3d", facecolor="white")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=S_MERGED, c=("black" if mono else cols),
                   marker=".", linewidths=0, depthshade=False, antialiased=not mono)
        style3d(ax, LIM_MERGED)
        ax.set_title(lab, fontsize=15 if mono else 13.5, color="black" if mono else "0.10", pad=-6)
        draw_window_box(ax)
        box = ax.get_position()
        ax2 = fig.add_axes([box.x0 + INSET_XYWH[0] * box.width,
                            box.y0 + INSET_XYWH[1] * box.height,
                            INSET_XYWH[2] * box.width, INSET_XYWH[3] * box.height],
                           projection="3d", facecolor="white")
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=S_MERGED * 2.6,
                    c=("black" if mono else cols), marker=".", linewidths=0,
                    depthshade=False, antialiased=not mono)
        style3d(ax2, LIM_WIN, zoom=1.34)
        for sp in ax2.spines.values():
            sp.set_visible(True)
            sp.set_color("black")
        p0 = ax2
    _ = p0
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("[保存] %s" % path)


def save_panels():
    d = OUT / f"{OUTNAME}_panels"
    d.mkdir(parents=True, exist_ok=True)
    for i, (pts, cols, lab) in enumerate(VIEW_PANELS + ARM_PANELS):
        fig = plt.figure(figsize=(5, 5.27), facecolor="white")
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        s = S_VIEW if i < 6 else S_MERGED
        L = LIM_VIEW if i < 6 else LIM_MERGED
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=cols, marker=".",
                   linewidths=0, depthshade=False)
        style3d(ax, L)
        ax.set_title(lab, fontsize=14, color="0.10", pad=-6)
        if i >= 6:
            draw_window_box(ax)
            box = ax.get_position()
            ax2 = fig.add_axes([box.x0 + INSET_XYWH[0] * box.width,
                                box.y0 + INSET_XYWH[1] * box.height,
                                INSET_XYWH[2] * box.width, INSET_XYWH[3] * box.height],
                               projection="3d", facecolor="white")
            ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s * 2.6, c=cols, marker=".",
                        linewidths=0, depthshade=False)
            style3d(ax2, LIM_WIN, zoom=1.34)
        fig.savefig(d / f"panel_{'abcdefghi'[i]}.png", dpi=200, facecolor="white")
        plt.close(fig)
    print("[保存] 9 张单幅面板 → %s" % d)


compose(True, OUT / f"{OUTNAME}_bw.png")
compose(False, OUT / f"{OUTNAME}_color.png")
save_panels()

from PIL import Image

p = OUT / f"{OUTNAME}_bw.png"
im = Image.open(p).convert("L")
bw = im.point(lambda v: 0 if v < 128 else 255, mode="L")
bw.save(p)
a = np.asarray(bw)
print("[二值化] 纯黑 %.2f%%  纯白 %.2f%%  中间灰 %.3f%%"
      % (100 * (a == 0).mean(), 100 * (a == 255).mean(), 100 * ((a > 0) & (a < 255)).mean()))

# ---------- 每格黑像素占比（评审要求的验收量）----------
H, W = a.shape
ratios = {}
for i in range(9):
    r, c = divmod(i, 3)
    sub = a[int(H * r / 3):int(H * (r + 1) / 3), int(W * c / 3):int(W * (c + 1) / 3)]
    key = ("view_" + "abcdef"[i]) if i < 6 else ("arm_" + "ghi"[i - 6])
    ratios[key] = round(100 * float((sub == 0).mean()), 2)
top = float(np.mean([v for k, v in ratios.items() if k.startswith("view")]))
bot = float(np.mean([v for k, v in ratios.items() if k.startswith("arm")]))
print("[验收] 上排平均 %.2f%%  下排平均 %.2f%%  比值 %.2f（目标 1.2~1.6）" % (top, bot, bot / top))

# ---------- 可编辑合成 ----------
W_, H_, GAP, MARGIN = 900, 948, 6, 12
TW, TH = 3 * W_ + 2 * GAP + 2 * MARGIN, 3 * H_ + 2 * GAP + 2 * MARGIN
svg = ['<?xml version="1.0" encoding="UTF-8"?>',
       '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
       'width="%d" height="%d" viewBox="0 0 %d %d">' % (TW, TH, TW, TH),
       '<rect width="100%%" height="100%%" fill="#ffffff"/>']
dx = ['<mxfile host="app.diagrams.net"><diagram id="fig5v2" name="图5">',
      '<mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="0" tooltips="1" '
      'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" '
      'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' % (TW, TH)]
labels = [p[2] for p in VIEW_PANELS] + [p[2] for p in ARM_PANELS]
for i, lab in enumerate(labels):
    r, c = divmod(i, 3)
    x = MARGIN + c * (W_ + GAP)
    y = MARGIN + r * (H_ + GAP)
    b64 = base64.b64encode((OUT / f"{OUTNAME}_panels" / f"panel_{'abcdefghi'[i]}.png").read_bytes()).decode()
    svg.append('<image x="%d" y="%d" width="%d" height="%d" preserveAspectRatio="none" '
               'xlink:href="data:image/png;base64,%s"/>' % (x, y, W_, H_, b64))
    svg.append('<text x="%d" y="%d" font-family="Helvetica" font-size="34" fill="#000000">%s</text>'
               % (x + 12, y + 40, html.escape(lab)))
    dx.append('<mxCell id="p%d" value="%s" style="shape=image;imageAspect=0;aspect=fixed;'
              'verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png;base64,%s" '
              'vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>'
              '</mxCell>' % (i + 2, html.escape(lab), b64, x, y, W_, H_))
svg.append('</svg>')
dx.append('</root></mxGraphModel></diagram></mxfile>')
(OUT / f"{OUTNAME}.svg").write_text("\n".join(svg))
(OUT / f"{OUTNAME}.drawio").write_text("\n".join(dx))
print("[保存] %s.svg  %s.drawio" % (OUTNAME, OUTNAME))

meta = {"arm_labels": [a[1] for a in ARM_LABELS],
        "merge_points_raw": {k: int(len(v)) for k, v in M_RAW.items()},
        "merge_points_rendered": {k: int(len(v)) for k, v in M.items()},
        "merged_voxel_m": MERGED_VOXEL, "s_view": S_VIEW, "s_merged": S_MERGED,
        "pad": PAD, "quantile": QUANT, "zoom": ZOOM,
        "window": {"center_z_percentile": WIN_CENTER_Z_PCT, "half_m": WIN_HALF,
                   "lim": [[float(x) for x in l] for l in LIM_WIN]},
        "black_pixel_pct": ratios, "top_mean_pct": round(top, 2), "bottom_mean_pct": round(bot, 2),
        "bottom_over_top": round(bot / top, 2),
        "color_recipe": {"scale_pct": SCALE_PCT, "gamma": GAMMA, "saturation": SAT,
                         "stretch": [LO_PCT, HI_PCT]}}
(OUT / f"{OUTNAME}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("[保存] %s_meta.json" % OUTNAME)
