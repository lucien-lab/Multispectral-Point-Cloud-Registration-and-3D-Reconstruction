# -*- coding: utf-8 -*-
"""专利图 5（终版，3 列 × 4 行 = 11 格）

按 vision 评审结论重排（评审：三臂在 50 mm 格宽下肉眼不可分；缺"配准前"对照；
上排过重、下排过淡；须体现本发明方法），并把三方式差异改用**定量柱状图**呈现：

  (a)~(f) 6 个视角的原始点云（各扫描仪坐标系）
  (g) 未配准直接合并的对照           ← 恢复"配准前"，唯一能清楚辨出盆体的格子
  (h) 几何配准
  (i) 几何与可见光融合配准
  (j) 几何与多光谱融合配准（本发明方法）
  (k) 可辨识点集口径下 5 mm / 10 mm 几何一致性柱状图（4 种配置，3 轮重复误差棒）

点密度按实测验收：单视角 s=3.4（黑像素约 2.0%）、合并云体素 4 mm + s=4.0（约 2.8%），
下排/上排 ≈ 1.4（评审目标 1.2~1.6）；取景 q=0.5%、pad=1.5%、zoom=1.10。
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
OUTNAME = os.environ.get("OUTNAME", "fig5_final2")
ROWS, COLS = 4, 3
FIGW, FIGH = 15.0, 15.8 * 4 / 3

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


def voxel_idx(P, v):
    key = np.floor(P / v).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)


def transformed(arm):
    out = []
    for d in DEGS:
        T = np.eye(4) if d == 0 else POSE[f"{arm}|{d}"]
        out.append((T[:3, :3] @ XYZ[d].T).T + T[:3, 3])
    return np.vstack(out)


ARMS = [("A 几何", "几何配准"), ("B 几何+RGB", "几何与可见光融合配准"),
        ("C 几何+多光谱(本发明)", "几何与多光谱融合配准（本发明方法）")]
M_RAW = {a[0]: transformed(a[0]) for a in ARMS}
M_UNREG = transformed(None) if False else np.vstack([XYZ[d] for d in DEGS])   # 未配准直接合并
M = {k: v[voxel_idx(v, MERGED_VOXEL)] for k, v in M_RAW.items()}
M["未配准"] = M_UNREG[voxel_idx(M_UNREG, 0.0025)]   # 对照格用更细体素，使重影/错位更明显
MVIS = {k: M_VIS_FULL[voxel_idx(v, MERGED_VOXEL)] for k, v in M_RAW.items()}
MVIS["未配准"] = M_VIS_FULL[voxel_idx(M_UNREG, 0.0025)]


def lim(points, q=QUANT, pad=PAD):
    a = np.vstack(points)
    lo, hi = np.percentile(a, [q, 100 - q], axis=0)
    p = (hi - lo) * pad
    return [(lo[i] - p[i], hi[i] + p[i]) for i in range(3)]


LIM_VIEW = lim([XYZ[d] for d in DEGS])
LIM_MERGED = lim([M["未配准"]] + [M[a[0]] for a in ARMS])

# ---------------- 柱状图数据（可辨识点集口径，3 轮重复） ----------------
J = json.loads((HERE / "abc_three_way.json").read_text())
wl = np.load(CACHE / "frame_000.npz")["wavelength"]
k650, k800 = int(np.argmin(abs(wl - 650))), int(np.argmin(abs(wl - 800)))
NDVI_TH = 0.5
from scipy.spatial import cKDTree


def plant_metrics(T, d):
    f = np.load(CACHE / f"frame_{d:03d}.npz")
    b = np.load(CACHE / "frame_000.npz")
    rs, rt = np.clip(f["refl"].astype(float), 0, None), np.clip(b["refl"].astype(float), 0, None)
    nds = (rs[:, k800] - rs[:, k650]) / np.maximum(rs[:, k800] + rs[:, k650], 1e-9)
    ndt = (rt[:, k800] - rt[:, k650]) / np.maximum(rt[:, k800] + rt[:, k650], 1e-9)
    ms, mt = nds >= NDVI_TH, ndt >= NDVI_TH
    P = (T[:3, :3] @ f["xyz"][ms].T).T + T[:3, 3]
    kd = cKDTree(b["xyz"][mt])
    o = {}
    for t in (5, 10):
        dd, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(dd).mean())
    return o


CH = {"未配准": {"fit@5mm": [], "fit@10mm": []}}
for d in [60, 120, 180, 240, 300]:
    m = plant_metrics(np.eye(4), d)
    for k in m:
        CH["未配准"][k].append(m[k])
for lab in ("几何", "几何+可见光", "几何+多光谱（本发明）"):
    CH[lab] = {"fit@5mm": [], "fit@10mm": []}
for i, a in enumerate(ARMS):
    lab = ("几何", "几何+可见光", "几何+多光谱（本发明）")[i]
    CH[lab]["fit@5mm"] = [J[a[0]]["runs"][r]["plant_fit@5mm"] for r in range(3)]
    CH[lab]["fit@10mm"] = [J[a[0]]["runs"][r]["plant_fit@10mm"] for r in range(3)]
CH_MEAN = {k: {m: (float(np.mean(v[m])), float(np.std(v[m]))) for m in v} for k, v in CH.items()}
print("柱状图数据（可辨识点集口径，均值±标准差）：")
for k, v in CH_MEAN.items():
    print("  %-16s 5mm %.4f±%.4f  10mm %.4f±%.4f" % (k, v["fit@5mm"][0], v["fit@5mm"][1],
                                                    v["fit@10mm"][0], v["fit@10mm"][1]))


def draw_chart(ax, mono=True):
    names = list(CH_MEAN)
    hatches = ["", "///", "xxx", "...."]
    w, x = 0.2, np.arange(2)
    for i, nm in enumerate(names):
        mu = [CH_MEAN[nm]["fit@5mm"][0], CH_MEAN[nm]["fit@10mm"][0]]
        sd = [CH_MEAN[nm]["fit@5mm"][1], CH_MEAN[nm]["fit@10mm"][1]]
        ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=3,
               facecolor="white", edgecolor="black", linewidth=1.0,
               hatch=hatches[i], label=nm, error_kw={"ecolor": "black", "lw": 1.0})
        for xi, m, s_ in zip(x + (i - 1.5) * w, mu, sd):
            ax.annotate("%.4f" % m, (xi, m + s_ + 0.02), ha="center", va="bottom",
                        fontsize=7.2, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["5 mm 阈值", "10 mm 阈值"])
    ax.set_ylabel("可辨识点集几何一致性")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", ls=":", lw=0.6, color="0.55")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left", frameon=True, edgecolor="black")
    for sp in ax.spines.values():
        sp.set_color("black")


def panel_xyz(fig, gs_pos, pts, cols, s, L, mono=True):
    ax = fig.add_subplot(gs_pos, projection="3d", facecolor="white")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=("black" if mono else cols),
               marker=".", linewidths=0, depthshade=False, antialiased=not mono)
    for k, st in enumerate("xyz"):
        getattr(ax, "set_%slim" % st)(*L[k])
    ax.view_init(ELEV, AZIM)
    ax.set_box_aspect(tuple(float(L[k][1] - L[k][0]) for k in range(3)), zoom=ZOOM)
    ax.set_axis_off()
    return ax


def compose(mono, path):
    fig = plt.figure(figsize=(FIGW, FIGH), facecolor="white")
    gs = GridSpec(ROWS, COLS, figure=fig, left=0.010, right=0.990, top=0.982, bottom=0.008,
                  wspace=0.015, hspace=0.03)
    fs = 15 if mono else 14
    for i, d in enumerate(DEGS):
        ax = panel_xyz(fig, gs[i // 3, i % 3], XYZ[d], VIS[d], S_VIEW, LIM_VIEW, mono)
        ax.set_title("(%s) %d°" % ("abcdef"[i], d), fontsize=fs, color="black" if mono else "0.10", pad=-6)
    items = [("未配准", "未配准直接合并（对照）"), (ARMS[0][0], ARMS[0][1]),
             (ARMS[1][0], ARMS[1][1]), (ARMS[2][0], ARMS[2][1])]
    for i, (key, lab) in enumerate(items):
        k = 6 + i
        ax = panel_xyz(fig, gs[k // 3, k % 3], M[key], MVIS[key], S_MERGED, LIM_MERGED, mono)
        ax.set_title("(%s) %s" % ("ghij"[i], lab), fontsize=fs, color="black" if mono else "0.10", pad=-2)
    axc = fig.add_subplot(gs[3, 1:])
    draw_chart(axc, mono)
    axc.set_title("(k) 可辨识点集口径几何一致性（3 轮重复，误差棒为标准差）",
                  fontsize=fs - 1, color="black" if mono else "0.10", pad=14)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("[保存] %s" % path)


def save_panels():
    d = OUT / f"{OUTNAME}_panels"
    d.mkdir(parents=True, exist_ok=True)
    items = [("a", XYZ[0]), ("b", XYZ[60]), ("c", XYZ[120]), ("d", XYZ[180]),
             ("e", XYZ[240]), ("f", XYZ[300])]
    for tag, pts in items:
        fig = plt.figure(figsize=(5, 5.27), facecolor="white")
        ax = panel_xyz(fig, 111, pts, VIS[DEGS["abcdef".index(tag)]], S_VIEW, LIM_VIEW, False)
        fig.savefig(d / f"panel_{tag}.png", dpi=200, facecolor="white")
        plt.close(fig)
    keys = ["未配准", ARMS[0][0], ARMS[1][0], ARMS[2][0]]
    labs = ["未配准直接合并（对照）", ARMS[0][1], ARMS[1][1], ARMS[2][1]]
    for tag, key, lab in zip("ghij", keys, labs):
        fig = plt.figure(figsize=(5, 5.27), facecolor="white")
        ax = panel_xyz(fig, 111, M[key], MVIS[key], S_MERGED, LIM_MERGED, False)
        ax.set_title(lab, fontsize=13, color="0.10", pad=-6)
        fig.savefig(d / f"panel_{tag}.png", dpi=200, facecolor="white")
        plt.close(fig)
    fig = plt.figure(figsize=(6.2, 5.0), facecolor="white")
    draw_chart(fig.add_subplot(111), True)
    fig.savefig(d / "panel_k_chart.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("[保存] 11 张单幅面板 → %s" % d)


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

H, W = a.shape
ratios = {}
for i in range(6):
    r, c = divmod(i, 3)
    ratios["view_" + "abcdef"[i]] = round(100 * float((a[int(H * r / ROWS):int(H * (r + 1) / ROWS),
                                                         int(W * c / 3):int(W * (c + 1) / 3)] == 0).mean()), 2)
for i in range(4):
    r, c = 2 + i // 3, i % 3
    ratios["merge_" + "ghij"[i]] = round(100 * float((a[int(H * r / ROWS):int(H * (r + 1) / ROWS),
                                                         int(W * c / 3):int(W * (c + 1) / 3)] == 0).mean()), 2)
top = float(np.mean([v for k, v in ratios.items() if k.startswith("view")]))
bot = float(np.mean([v for k, v in ratios.items() if k.startswith("merge")]))
print("[验收] 上排 %.2f%%  合并排 %.2f%%  比值 %.2f（评审目标 1.2~1.6）" % (top, bot, bot / top))

# ---------------- 可编辑合成 ----------------
W_, H_, GAP, MARGIN = 900, 948, 6, 12
TW, TH = 3 * W_ + 2 * GAP + 2 * MARGIN, 4 * H_ + 3 * GAP + 2 * MARGIN
svg = ['<?xml version="1.0" encoding="UTF-8"?>',
       '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
       'width="%d" height="%d" viewBox="0 0 %d %d">' % (TW, TH, TW, TH),
       '<rect width="100%%" height="100%%" fill="#ffffff"/>']
dx = ['<mxfile host="app.diagrams.net"><diagram id="fig5f2" name="图5">',
      '<mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="0" tooltips="1" '
      'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" '
      'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' % (TW, TH)]
titles = [("(%s) %d°" % ("abcdef"[i], d)) for i, d in enumerate(DEGS)] + \
         ["(g) 未配准直接合并（对照）", "(h) 几何配准", "(i) 几何与可见光融合配准",
          "(j) 几何与多光谱融合配准（本发明方法）", "(k) 可辨识点集口径几何一致性（柱状图）"]
files = [f"panel_{c}.png" for c in "abcdefghij"] + ["panel_k_chart.png"]
for i, (lab, fn) in enumerate(zip(titles, files)):
    if i < 10:
        r, c = divmod(i, 3)
        ncol, nw = 1, W_
    else:
        r, c, ncol, nw = 3, 1, 2, W_ * 2 + GAP
    x = MARGIN + c * (W_ + GAP)
    y = MARGIN + r * (H_ + GAP)
    b64 = base64.b64encode((OUT / f"{OUTNAME}_panels" / fn).read_bytes()).decode()
    svg.append('<image x="%d" y="%d" width="%d" height="%d" preserveAspectRatio="none" '
               'xlink:href="data:image/png;base64,%s"/>' % (x, y, nw, H_, b64))
    svg.append('<text x="%d" y="%d" font-family="Helvetica" font-size="34" fill="#000000">%s</text>'
               % (x + 12, y + 40, html.escape(lab)))
    dx.append('<mxCell id="p%d" value="%s" style="shape=image;imageAspect=0;aspect=fixed;'
              'verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png;base64,%s" '
              'vertex="1" parent="1"><mxGeometry x="%d" y="%d" width="%d" height="%d" as="geometry"/>'
              '</mxCell>' % (i + 2, html.escape(lab), b64, x, y, nw, H_))
svg.append('</svg>')
dx.append('</root></mxGraphModel></diagram></mxfile>')
(OUT / f"{OUTNAME}.svg").write_text("\n".join(svg))
(OUT / f"{OUTNAME}.drawio").write_text("\n".join(dx))
print("[保存] %s.svg  %s.drawio" % (OUTNAME, OUTNAME))

meta = {"layout": "4x3 (a-f views, g unregistered, h 几何, i 几何+可见光, j 几何+多光谱(本发明), k chart)",
        "merge_points_raw": {k: int(len(v)) for k, v in M_RAW.items()},
        "merge_points_rendered": {k: int(len(v)) for k, v in M.items()},
        "unregistered_points_raw": int(len(M_UNREG)),
        "merged_voxel_m": MERGED_VOXEL, "s_view": S_VIEW, "s_merged": S_MERGED,
        "pad": PAD, "quantile": QUANT, "zoom": ZOOM,
        "black_pixel_pct": ratios, "view_mean_pct": round(top, 2),
        "merge_mean_pct": round(bot, 2), "merge_over_view": round(bot / top, 2),
        "chart_data": {k: {m: [round(v[m][0], 4), round(v[m][1], 4)] for m in v}
                       for k, v in CH_MEAN.items()},
        "color_recipe": {"scale_pct": SCALE_PCT, "gamma": GAMMA, "saturation": SAT,
                         "stretch": [LO_PCT, HI_PCT]}}
(OUT / f"{OUTNAME}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("[保存] %s_meta.json" % OUTNAME)
