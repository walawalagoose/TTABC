# TTABC Project Page

《**What Drives Test-Time Adaptation for CLIP? A Controlled Empirical Study from an Update Perspective**》(NeurIPS 2026 D&B Track) 的项目展示主页。

一个纯静态、零依赖的单页学术主页，可直接在浏览器打开或用任意静态服务器托管（如 GitHub Pages）。

## 目录结构

```
project_page/
├── index.html          # 主页面
├── assets/
│   ├── style.css       # 深色学术风格样式（响应式）
│   ├── script.js       # 导航 / 复制 / 滚动动画 / 数字计数
│   └── img/
│       ├── empirical_study.jpg   # 研究总览图
│       └── taxonomy.jpg          # TTA4CLIP 分类法图
└── README.md
```

## 页面内容

- **Hero**：标题、作者、机构、Paper / arXiv / Code 链接、关键统计（20+ 方法、3 范式、4 类偏移、16 数据集）
- **Overview**：摘要与研究总览图
- **Taxonomy**：三大更新范式（parameter / state / inference-based）及其代表方法
- **Findings**：三个核心发现（Part 1/2/3）
- **Results**：主结果表（自然/损坏偏移、norm-layer 专长损坏）与各偏移的“获胜范式”
- **Benchmark**：TTABC 评测平台、四类偏移数据集、代码库结构树
- **Get Started**：安装 / 运行 / 新增方法
- **Citation**：BibTeX 与致谢

## 本地预览

方式一：直接双击 `index.html` 在浏览器打开。

方式二：启动静态服务器（推荐，避免个别浏览器的本地资源限制）：

```bash
cd project_page
python -m http.server 8000
# 浏览器访问 http://localhost:8000
```

## 发布到 GitHub Pages

本仓库已配置自动部署（`.github/workflows/deploy_page.yml`），推送到 `main` 分支且 `project_page/**` 有改动时自动发布。

首次使用需在仓库页面开启一次：

1. 打开仓库 → **Settings** → **Pages**
2. **Source** 选择 **GitHub Actions**
3. 保存后推送 `project_page/` 与 `.github/` 即可

发布后访问地址：`https://<用户名>.github.io/TTABC/`

> 仓库为 public 时 GitHub Pages 免费；private 仓库需要付费计划。
> 页面内所有资源均为相对路径，可直接部署在子路径 `/TTABC/` 下，无需修改。
> 如需自定义域名：在 `project_page/` 下添加 `CNAME` 文件（写入域名），并在 DNS 添加 CNAME 记录。

## 说明

- 页面数据（结果表、内存/耗时、相关系数等）均取自论文正文，仅选取代表性方法用于展示。
- 图片素材取自代码仓库 `TTABC/img/`。
- 论文与代码链接以 README 中的占位地址为准，正式发布时请替换为真实 arXiv / GitHub 地址。
- `.nojekyll` 文件用于跳过 Jekyll 处理，请勿删除。
