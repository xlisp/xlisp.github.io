---
title: Hello, world
date: 2026-05-06
slug: hello-world
summary: First post — what this blog is for.
---

# Hello, world

Welcome to my notebook. This site is generated from plain markdown files
that live in `docs/`. Each time I run `python build.py` the script walks
the folder, renders every `.md` into `posts/<slug>.html`, and rebuilds the
home page.

## Why bother

I read a lot of posts, papers and code. I want a quiet place to write
about three things:

- **Reinforcement learning** — algorithms, tricks, failure modes
- **Large language models** — training, alignment, agents, tooling
- **Lisp & Emacs** — small languages and hackable editors

## A taste of code

```python
def reward(state, action):
    """Toy reward — replace with something real."""
    return -abs(state.target - action)
```

```clojure
(defn fib [n]
  (loop [a 0 b 1 i 0]
    (if (= i n) a
        (recur b (+ a b) (inc i)))))
```

> The best way to predict the future is to invent it. — Alan Kay

More to come — see you next post.
