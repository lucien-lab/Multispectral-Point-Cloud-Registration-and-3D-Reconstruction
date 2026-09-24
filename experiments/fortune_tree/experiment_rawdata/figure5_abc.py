# -*- coding: utf-8 -*-
"""专利图 5：6 视角原始点云 + 三种融合方式的配准结果（3x3）

  (a)-(f) 6 个视角的原始点云（各扫描仪坐标系）
  (g) 几何配准结果
  (h) 几何与可见光融合配准结果
  (i) 几何与多光谱融合配准结果

输出：
  visualization/fig5_abc_bw.png        纯黑白版（插入专利附图，尺寸 3000x3160 与原文一致）
  visualization/fig5_abc_color.png     高对比度彩色版（供报告/论文）
  visualization/fig5_abc_panels/*.png  9 张单幅面板（可编辑合成的素材）
  visualization/fig5_abc.svg           可编辑合成（SVG，9 图 + 标签，可再编辑）
  visualization/fig5_abc.drawio        drawio 源文件（与本项目 picture/ 体例一致）
  visualization/fig5_abc_assembly.html 网页浏览版（可选）
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "visualization"
PANELS_DIR = OUT / "fig5_abc_panels"

DEGS = [0, 60, 120, 180, 240, 300]
RGB_BANDS_1BASED = (957, 730, 552)
ELEV, AZIM = 20, -60
import os
S_VIEW, S_MERGED = 4.5, 2.8
MERGED_VOXEL = 0.004
PAD, ZOOM = 0.015, 1.10          # 面板内边距与缩放（紧边放大，提高 5 cm 显示下的可辨度）
_ov = lambda k, v: type(v)(os.environ[k]) if os.environ.get(k) else v
S_VIEW, S_MERGED = _ov("S_VIEW", S_VIEW), _ov("S_MERGED", S_MERGED)
MERGED_VOXEL, PAD, ZOOM = _ov("VOXEL", MERGED_VOXEL), _ov("PAD", PAD), _ov("ZOOM", ZOOM)
OUTNAME = os.environ.get("OUTNAME", "fig5_abc")
PANELS_DIR = OUT / f"{OUTNAME}_panels"    # 合并云渲染前体素降采样（docx 中每格仅约 5 cm 宽，
                        # 2 万余点二值化后会糊成黑块；4 mm 体素下形态最清晰）
GAMMA = 2.6          # 白底压暗增饱和（加大对比度）
SAT = 1.9            # 饱和度增益
SCALE_PCT = 95.0     # 亮度归一化分位
LO_PCT, HI_PCT = 0.02, 0.98   # 通道线性拉伸的下上界（增强对比）

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "STSong", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ARMS = [("A 几何", "几何配准"), ("B 几何+RGB", "几何与可见光融合配准"),
        ("C 几何+多光谱(本发明)", "几何与多光谱融合配准")]

# ------------------------------------------------------------------ 数据
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
    """返回 (合并坐标, 对应颜色)；体素降采样时颜色同步抽取，保证一一对应。"""
    out = []
    for d in DEGS:
        T = np.eye(4) if d == 0 else POSE[f"{arm}|{d}"]
        out.append((T[:3, :3] @ XYZ[d].T).T + T[:3, 3])
    P = np.vstack(out)
    if voxel:
        key = np.floor(P / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)   # 每体素保留一个点
        idx = np.sort(idx)
        P = P[idx]
        return P, M_VIS_FULL[idx]
    return P, M_VIS_FULL


MC = {a[0]: merged(a[0]) for a in ARMS}
M = {k: v[0] for k, v in MC.items()}
MCV = {k: v[1] for k, v in MC.items()}
M_RAW = {a[0]: merged(a[0], voxel=None)[0] for a in ARMS}
print("合并点数（原始 → 渲染）: " + "  ".join(
    "%s=%d→%d" % (k, len(M_RAW[k]), len(M[k])) for k in M))
for k, v in M.items():
    print("  %-24s 外廓跨度 x=%.1f y=%.1f z=%.1f mm" % (
        k, 1000 * (np.percentile(v[:, 0], 99.8) - np.percentile(v[:, 0], 0.2)),
        1000 * (np.percentile(v[:, 1], 99.8) - np.percentile(v[:, 1], 0.2)),
        1000 * (np.percentile(v[:, 2], 99.8) - np.percentile(v[:, 2], 0.2))))


def lim(points, q=0.2, pad=PAD):
    a = np.vstack(points)
    lo, hi = np.percentile(a, [q, 100 - q], axis=0)
    p = (hi - lo) * pad
    return [(lo[i] - p[i], hi[i] + p[i]) for i in range(3)]


LIM_VIEW = lim([XYZ[d] for d in DEGS])
LIM_MERGED = lim(list(M.values()))

PANELS = [(XYZ[d], VIS[d], S_VIEW, LIM_VIEW, f"({'abcdef'[i]}) {d}°", f"panel_{'abcdef'[i]}")
          for i, d in enumerate(DEGS)]
PANELS += [(M[a[0]], MCV[a[0]], S_MERGED, LIM_MERGED, f"({'ghi'[i]}) {a[1]}", f"panel_{'ghi'[i]}")
           for i, a in enumerate(ARMS)]


def draw_axes(fig, gs, i, pts, cols, s, L, lab, mono):
    ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d", facecolor="white")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=("black" if mono else cols),
               marker=".", linewidths=0, depthshade=False, antialiased=not mono)
    ax.set_xlim(*L[0]); ax.set_ylim(*L[1]); ax.set_zlim(*L[2])
    ax.view_init(ELEV, AZIM)
    ax.set_box_aspect(tuple(float(L[k][1] - L[k][0]) for k in range(3)), zoom=1.03)
    ax.set_axis_off()
    ax.set_title(lab, fontsize=15 if mono else 14, color="black" if mono else "0.10", pad=-6)
    return ax


def compose(mono, path):
    fig = plt.figure(figsize=(15, 15.8), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, left=0.012, right=0.988, top=0.972, bottom=0.012,
                  wspace=0.015, hspace=0.03)
    for i, (pts, cols, s, L, lab, _) in enumerate(PANELS):
        draw_axes(fig, gs, i, pts, cols, s, L, lab, mono)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("[保存] %s" % path)


def save_panels():
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for pts, cols, s, L, lab, tag in PANELS:
        fig = plt.figure(figsize=(5, 5.27), facecolor="white")
        gs = GridSpec(1, 1, figure=fig, left=0.005, right=0.995, top=0.94, bottom=0.005)
        draw_axes(fig, gs, 0, pts, cols, s * 11.5, L, lab, False)
        p = PANELS_DIR / f"{tag}.png"
        fig.savefig(p, dpi=200, facecolor="white")
        plt.close(fig)
    print("[保存] %d 张单幅面板 → %s" % (len(PANELS), PANELS_DIR))


compose(True, OUT / f"{OUTNAME}_bw.png")
compose(False, OUT / f"{OUTNAME}_color.png")
save_panels()

# ---------------- 黑白版二值化（消除抗锯齿灰度，确保只用黑色） ----------------
from PIL import Image

p = OUT / f"{OUTNAME}_bw.png"
im = Image.open(p).convert("L")
bw = im.point(lambda v: 0 if v < 128 else 255, mode="L")
bw.save(p)
a = np.asarray(bw)
print("[二值化] %s → 纯黑 %.2f%%  纯白 %.2f%%  中间灰 %.3f%%"
      % (p.name, 100 * (a == 0).mean(), 100 * (a == 255).mean(),
         100 * ((a > 0) & (a < 255)).mean()))

# ---------------- 可编辑合成：SVG + drawio ----------------
def panel_files():
    return [(f"panel_{'abcdefghi'[i]}",
             f"{'abcdefghi'[i]}", PANELS[i][4]) for i in range(9)]


W, H, GAP, MARGIN = 900, 948, 6, 12
TW, TH = 3 * W + 2 * GAP + 2 * MARGIN, 3 * H + 2 * GAP + 2 * MARGIN
svg = ['<?xml version="1.0" encoding="UTF-8"?>',
       '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
       'width="%d" height="%d" viewBox="0 0 %d %d">' % (TW, TH, TW, TH),
       '<rect width="100%%" height="100%%" fill="#ffffff"/>']
dx = ['<mxfile host="app.diagrams.net"><diagram id="fig5abc" name="图5">',
      '<mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="0" tooltips="1" '
      'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" '
      'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' % (TW, TH)]
for i, (tag, letter, lab) in enumerate(panel_files()):
    r, c = divmod(i, 3)
    x = MARGIN + c * (W + GAP)
    y = MARGIN + r * (H + GAP)
    b64 = base64.b64encode((PANELS_DIR / f"{tag}.png").read_bytes()).decode()
    svg.append('<image x="%d" y="%d" width="%d" height="%d" preserveAspectRatio="none" '
               'xlink:href="data:image/png;base64,%s"/>' % (x, y, W, H, b64))
    svg.append('<text x="%d" y="%d" font-family="Helvetica" font-size="34" fill="#000000">%s</text>'
               % (x + 12, y + 40, html.escape(lab)))
    dx.append('<mxCell id="p%d" value="%s" style="shape=image;imageAspect=0;aspect=fixed;'
              'verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png;base64,%s" '
              'vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>'
              '</mxCell>' % (i + 2, html.escape(lab), b64, x, y, W, H))
svg.append('</svg>')
dx.append('</root></mxGraphModel></diagram></mxfile>')
(OUT / f"{OUTNAME}.svg").write_text("\n".join(svg))
(OUT / f"{OUTNAME}.drawio").write_text("\n".join(dx))
print("[保存] %s.svg  %s.drawio（可编辑合成，9 图 + 标签）" % (OUTNAME, OUTNAME))

meta = {"panels": [{"tag": t, "letter": l, "label": lab} for t, l, lab in panel_files()],
        "merge_points_raw": {k: int(len(v)) for k, v in M_RAW.items()},
        "merge_points_rendered": {k: int(len(v)) for k, v in M.items()},
        "merged_voxel_m": MERGED_VOXEL,
        "color_recipe": {"scale_p": scale, "scale_pct": SCALE_PCT, "gamma": GAMMA,
                         "saturation": SAT, "stretch": [LO_PCT, HI_PCT],
                         "rgb_bands_1based": list(RGB_BANDS_1BASED), "multiply": 4.0},
        "axes": {"elev": ELEV, "azim": AZIM, "s_view": S_VIEW, "s_merged": S_MERGED}}
(OUT / f"{OUTNAME}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("[保存] %s_meta.json" % OUTNAME)
