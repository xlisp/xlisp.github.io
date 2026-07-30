# 公众号发表记录抓取（标题 / 链接 / 阅读数）

抓公众号后台「发表记录」页的所有文章，结果**自动下载成 `mp_articles.json`**，不用手动复制。

两个脚本，**先用 API 模式**：

| 脚本 | 原理 | 翻页 |
| --- | --- | --- |
| `fetch_mp_articles.js` | **API 模式（推荐）** 直接请求接口，不点按钮 | 页面不刷新，一次跑完 |
| `fetch_mp_articles_click.js` | 点击模式（兜底） 点「下一页」抓 DOM | 每页会刷新，需手动重跑续抓 |

> 这个后台点「下一页」是**整页刷新**，内存里的变量全丢。所以两个脚本都不靠内存攒数据：
> API 模式压根不翻页；点击模式每抓完一页就写进 `localStorage`，刷新后接着抓。

抓取字段：`time`（发表时间）、`title`（标题）、`url`（永久链接）、`read`（阅读数）、
`like` / `share` / `comment`，API 模式还多给 `digest`（摘要）、`cover`（封面图）。

> ⚠️ API 模式的 `read` / `like` / `share` / `comment` **实测都返回 0**：`appmsgex`
> 里不带阅读数，这些数字是页面另外调统计接口拿的。**需要真实阅读数就用点击模式**
> （直接读页面上渲染出来的数字）。链接和标题两种模式都准。

抓完的结果留了一份在 `mp_articles.json`，用来生成首页的微信按钮，见
[「三、更新到网站首页」](#三更新到网站首页)。

---

## 一、API 模式（推荐）

### 1. 打开页面

登录 <https://mp.weixin.qq.com> → 左侧 **内容与互动 → 发表记录**。
地址栏应该是 `.../cgi-bin/appmsgpublish?...&token=xxx`（脚本会检查，不对会直接提示）。
停在第几页都行，API 模式从头抓。

### 2. 运行

用 **Sources → Snippets** 跑，别直接往 Console 粘（见下面「粘贴报错」）：

1. `F12` → **Sources** 面板 → 左栏 **Snippets**（没看到就点 `»` 展开）→ **+ New snippet**
2. 打开 `fetch_mp_articles.js`，全选复制，粘进去
3. `Cmd + Enter`（Win：`Ctrl + Enter`）运行

### 3. 过程与结果

```
total articles reported by the server: 45
fields available per article: aid, appmsgid, title, link, read_num, like_num, ...
begin=0: +10, total 10
begin=10: +10, total 20
...
reached total, done
done: 45 articles
```

每次请求间隔 **5 秒**，45 篇 5 页大概 25 秒。跑完**自动弹出 `mp_articles.json` 下载**。

| 命令 | 作用 |
| --- | --- |
| `__saveJSON()` | 再下一次 JSON |
| `__saveCSV()` | 下载 CSV（带 BOM，Excel 不乱码） |
| `__markdown()` | 返回 `- [标题](链接) - 阅读数` 的 Markdown 列表 |
| `window.__articles` | 内存里的结果数组 |

第一次跑会打印 `fields available per article: ...`，即接口真实返回的字段名。
如果 `read` 全是 0，把这行贴出来，按真实字段名改 `rowsFrom()` 里的映射即可。

---

## 二、点击模式（API 模式失败时用）

因为翻页会刷新页面，流程是「跑一次 = 抓一页」，需要你在每次刷新后再按一下 `Cmd+Enter`：

1. 停在**第 1 页**
2. Snippets 里新建 snippet，粘 `fetch_mp_articles_click.js`，`Cmd + Enter`
3. 控制台打印 `page 1/5: +10, stored 10 articles`，然后 **5 秒后自动点「下一页」**，页面刷新
4. 刷新后回到 Snippets，**再按一次 `Cmd + Enter`**，它会自动接着第 2 页抓
5. 重复到最后一页，脚本自己停下并**下载 `mp_articles.json`**

数据全程存在 `localStorage['mp_articles_state']`，刷新、关标签页都不会丢。

| 命令 | 作用 |
| --- | --- |
| `__mpSaveJSON()` | 随时下载已抓到的 JSON |
| `__mpSaveCSV()` | 随时下载 CSV |
| `__mpData()` | 查看已存数据 |
| `__mpReset()` | 清空进度，**重新抓之前必须先执行** |

5 秒的等待时间在脚本开头 `var DELAY = 5000;`，嫌慢/嫌快自己改。

---

## 三、更新到网站首页

首页 `index.html` 每篇文章右侧有一个 **微信** 徽章，点开就是公众号版本。整条链路：

```
mp_help/mp_articles.json   抓下来的原始数据（把新下载的覆盖进来）
        │  python mp_help/match_wechat.py --write
        ▼
wechat.json                slug -> 公众号链接（build.py 读它）
        │  python build.py
        ▼
index.html                 <a class="post-wechat"> 徽章
```

### 有新文章时

```bash
cp ~/Downloads/mp_articles.json mp_help/mp_articles.json
python mp_help/match_wechat.py          # 先干跑，看它想加什么
python mp_help/match_wechat.py --write  # 确认没问题再写
python build.py                         # 重新生成 index.html
```

### 标题对齐

公众号标题和站内标题基本都改过（`ChatGPT` ↔ `Claude`、`数学家` ↔ `物理学家`、
换了个说法重发……），所以不能按标题精确匹配。`match_wechat.py` 的做法：

1. 先算**标题相似度**（`difflib`，去掉标点空格后比）
2. 再算**摘要 vs 正文的字符二元组重合度**（公众号 `digest` 对 `docs/*.md` 前 3000 字）
3. 两者任一够高就认；都不够就**不猜**，打印出候选让你手工填 `wechat.json`

**已有的映射永远不会被覆盖**，手工改过的条目安全。一篇站内文章只会绑一个公众号链接。

当前 45 篇公众号文章 → 37 篇对上了站内文章。剩下 8 篇站内没有对应文章（可计算性/欧美底层理论、
Loop & Graph Engineering、1958 年的 Lisp、LangGraph 教程、物理学家版（数学家版的改写重发）、
Fable 5 发布、微积分 PyTorch、神经网络数学 PyTorch）；另有 12 篇站内文章没发过公众号，
它们就不显示徽章。

手工加一条：直接编辑 `wechat.json`，key 是 `posts/<slug>.html` 里的 slug：

```json
{
  "mri-to-multimodal": "https://mp.weixin.qq.com/s/zP1tk1ZcwYgtWFn6nf77fw"
}
```

然后 `python build.py`。

---

## 四、实现要点

**API 模式**

- 拿当前页面 URL（含 `token`），改写 `begin` / `count`，加 `f=json&ajax=1` 再 `fetch`，
  `credentials: 'include'` 带上登录 cookie
- 返回体里 `publish_page` 是个 **JSON 字符串**，需要二次 `JSON.parse`；里面
  `publish_list[].publish_info` 又是 JSON 字符串，第三次 `JSON.parse` 才拿到 `appmsgex` 数组
- 一次群发多图文 → `appmsgex` 有多条，全部展开
- 按 `total_count` 判断结束，`MAX_PAGES = 100` 兜底防死循环
- 每页写一次 `localStorage` 存档

**点击模式**

- 文章条目 `.weui-desktop-mass-appmsg`
- 标题取 `<a class="weui-desktop-mass-appmsg__title">` 的**直接 `<span>` 子节点** ——
  同一个 `<a>` 里还塞着 `原创`、`已修改` 标签，直接 `innerText` 会把它们拼进标题
- 阅读数 `.appmsg-view .weui-desktop-mass-media__data__inner`，`2,249` 转成数字
- 靠分页条最大页码 `.weui-desktop-pagination__num` 判断是否最后一页

两个脚本源码都是**纯 ASCII**（中文写成 `\uXXXX` 转义），避免复制粘贴被编码问题弄坏。

---

## 五、常见问题

**`Uncaught SyntaxError: Invalid or unexpected token`**
复制时混进了全角空格 / 不间断空格 / 弯引号。用 Snippets 方式，**从 `.js` 文件本体复制**，
不要从渲染后的网页或终端输出里复制。Console 直接粘还会被拦截，需要先手打 `allow pasting`。

**`you are not on the publish-records page`**
当前不在 `cgi-bin/appmsgpublish` 页面。从左侧菜单点进「发表记录」再跑。

**`server returned ret=xxx` / `session expired`**
登录态过期，刷新页面重新登录后再跑。

**`no publish_page in response`**
接口结构变了。控制台会打印返回体的顶层字段名，按那个改 `fetch_mp_articles.js` 的解析；
或者直接改用点击模式。

**点击模式「翻页后忘了重跑」**
不影响数据，已抓的都在 `localStorage`。回到 snippet 再按 `Cmd + Enter` 就从当前页继续。

**想重新抓一遍**
先 `__mpReset()` 清掉旧进度，否则会和上次的结果合并（按 url 去重，不会重复，但会混入旧数据）。
