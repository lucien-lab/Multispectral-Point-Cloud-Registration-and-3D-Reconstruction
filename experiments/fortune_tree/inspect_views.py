#!/usr/bin/env python3
"""查看 cripped/ 下各视角点云的范围/点数（8列：x,y,z,R,G,B,?,类别）"""
import numpy as np
from pathlib import Path

CRIPPED = Path(__file__).resolve().parent / "cripped"

for f in sorted(CRIPPED.glob("*.txt")):
    a = np.loadtxt(f, delimiter=",")
    xyz = a[:, :3]
    rgb = a[:, 3:6]
    print(f"{f.name}")
    print(f"   点数={len(a)}  范围 x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}] "
          f"y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}] z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}]")
    print(f"   中心=({xyz[:,0].mean():.3f},{xyz[:,1].mean():.3f},{xyz[:,2].mean():.3f}) "
          f"RGB均值={rgb.mean(axis=0).round(3)} 第7列=[{a[:,6].min():.2f},{a[:,6].max():.2f}]")