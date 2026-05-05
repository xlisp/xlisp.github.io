# xlisp.github.io

Personal site for [Steve Chan](https://github.com/xlisp) — Clojure / Emacs / Python Lisp hacker, focused on ML, Deep Learning, RL and LLM.

## Layout

```
.
├── build.py              # markdown → html generator
├── requirements.txt      # markdown + pygments
├── docs/                 # write your posts here as .md
│   └── hello-world.md
├── templates/            # html templates
│   ├── index.html
│   └── post.html
├── assets/style.css      # source stylesheet (copied to root on build)
├── posts/                # generated html (do not edit)
├── index.html            # generated landing page
└── style.css             # generated (copy of assets/style.css)
```

## Build

```bash
pip install -r requirements.txt
python build.py
```

This regenerates `index.html` and every file under `posts/`.

## Write a new post

1. Drop a markdown file into `docs/`, e.g. `docs/rl-notes.md`.
2. Optional front matter:

   ```markdown
   ---
   title: Notes on PPO
   date: 2026-05-10
   slug: ppo-notes
   summary: A short read on policy gradients.
   ---

   # Notes on PPO
   ...
   ```

3. Run `python build.py`.
4. Commit and push — GitHub Pages serves the root of the repo.

## Deploy on GitHub Pages

1. Create the repo `xlisp/xlisp.github.io` on GitHub.
2. From this directory:

   ```bash
   git init
   git add .
   git commit -m "init xlisp.github.io"
   git branch -M main
   git remote add origin git@github.com:xlisp/xlisp.github.io.git
   git push -u origin main
   ```

3. In repo Settings → Pages, choose `Deploy from a branch` → `main` / `/ (root)`.

The site will be live at <https://xlisp.github.io>.
