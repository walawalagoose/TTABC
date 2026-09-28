# TTABC Project Page

Project homepage for **"What Drives Test-Time Adaptation for CLIP? A Controlled Empirical Study from an Update Perspective"** (NeurIPS 2026 E&D Track).

A fully static, dependency-free single-page academic site. It can be opened directly in a browser or hosted on any static server (e.g., GitHub Pages).

## Directory Structure

```
project_page/
├── index.html          # Main page
├── assets/
│   ├── style.css       # Light academic theme (responsive)
│   ├── script.js       # Nav / copy / scroll reveal / count-up
│   └── img/
│       ├── empirical_study.jpg   # Overview of the empirical study
│       └── taxonomy.jpg          # TTA4CLIP taxonomy
└── README.md
```

## Page Sections

- **Hero**: title, authors, affiliations, Paper / arXiv / Code links, and key stats (20+ methods, 3 paradigms, 4 shift categories, 16 datasets)
- **Overview**: abstract and the empirical-study overview figure
- **Taxonomy**: the three update paradigms (parameter / state / inference-based) and their representative methods
- **Findings**: the three key findings (Part 1/2/3)
- **Results**: main result tables (natural / corruption shifts, norm-layer specialization) and the "winning paradigm" per shift
- **Benchmark**: the TTABC evaluation platform, four shift categories, and the repository structure tree
- **Get Started**: installation / running experiments / adding a new method
- **Citation**: BibTeX and acknowledgements

## Local Preview

Option 1: open `index.html` directly in a browser.

Option 2: start a static server (recommended, avoids local-resource restrictions in some browsers):

```bash
cd project_page
python -m http.server 8000
# Visit http://localhost:8000
```

## Deploy to GitHub Pages

This repository ships with an automated deployment workflow (`.github/workflows/deploy_page.yml`) that publishes the page whenever changes under `project_page/**` are pushed to the `main` branch.

A one-time setup is required in the repository settings:

1. Open the repository → **Settings** → **Pages**
2. Set **Source** to **GitHub Actions**
3. Save, then push `project_page/` and `.github/`

Published URL: `https://<username>.github.io/TTABC/`

> GitHub Pages is free for public repositories; private repositories require a paid plan.
> All in-page resources use relative paths, so the site works directly under the `/TTABC/` sub-path without changes.
> For a custom domain: add a `CNAME` file (containing your domain) under `project_page/`, and configure a CNAME record in your DNS.

## Notes

- All page data (result tables, memory/latency, correlation coefficients, etc.) is taken from the paper; only representative methods are shown for clarity.
- Image assets are copied from the code repository `TTABC/img/`.
- Paper and code links follow the placeholders in the repository README; replace them with the final arXiv / GitHub URLs before release.
- The `.nojekyll` file disables Jekyll processing — please do not delete it.
