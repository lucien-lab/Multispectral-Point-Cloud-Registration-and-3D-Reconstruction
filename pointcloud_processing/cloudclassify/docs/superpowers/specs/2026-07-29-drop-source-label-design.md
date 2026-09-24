# 移除统一分类输出中的 source_label

## 目标

从 `cropped_point_cloud_classified.txt` 移除恒为 0 的 `source_label` 列，保留类别映射表不变。

## 输出格式

输出首行改为：

`point_id,x,y,z,r,g,b,class_id`

输出为 36,840 行、8 列。前 7 列与 `cropped_point_cloud.txt` 的 `point_id,X,Y,Z,R,G,B` 一一对应；第 8 列为 0–39 的 `class_id`。

## 验证

验证原 7 列逐值保持、`class_id` 范围为 0–39、所有 40 类仍出现，`class_id_mapping.csv` 内容不变。
