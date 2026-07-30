# 公众号发表记录抓取（标题 / 链接 / 阅读数）

`fetch_mp_articles.js` 在 Chrome 里跑，抓取公众号后台「发表记录」页的所有文章，自动点「下一页」翻到底。

抓取字段：

| 字段 | 说明 |
| --- | --- |
| `time` | 发表时间（如 `星期二 20:58` / `07月23日`） |
| `title` | 文章标题（已剔除「原创」「已修改」标签，`&nbsp;` 转普通空格） |
| `url` | `https://mp.weixin.qq.com/s/xxx` 永久链接 |
| `read` | 阅读人数 |
| `like` / `share` / `rec` / `comment` | 点赞 / 分享 / 推荐 / 留言数 |

## 一、打开目标页面

1. 登录 <https://mp.weixin.qq.com>
2. 左侧菜单：**内容与互动 → 发表记录**（页面上有「发表记录」标题和底部分页条 `1 2 3 4 5 下一页`）
3. 确认停在**第 1 页**再运行脚本（脚本从当前页往后翻，不会往前翻）

## 二、运行脚本（推荐：Snippets，不会有粘贴报错）

Chrome 控制台默认会拦截粘贴的代码，而且从终端/网页复制时容易混进不可见字符，导致
`Uncaught SyntaxError: Invalid or unexpected token`。用 Sources → Snippets 最稳：

1. `F12` 打开 DevTools → 顶部 **Sources** 面板
2. 左侧栏 **Snippets**（若没看到，点 `»` 展开）→ **+ New snippet**
3. 用编辑器打开本目录的 `fetch_mp_articles.js`，**全选复制**，粘到 snippet 里
4. `Cmd + Enter`（Windows：`Ctrl + Enter`）运行

脚本存成 snippet 后可反复使用，下次直接选中按 `Cmd + Enter`。

### 备选：直接在 Console 粘贴

1. `F12` → **Console**
2. 如果提示 *"Warning: Don't paste code into the DevTools Console…"*，先在输入框手打
   `allow pasting` 回车，解除粘贴限制
3. 再粘贴 `fetch_mp_articles.js` 全文，回车

## 三、运行过程与结果

控制台会逐页打印进度：

```
page 1: +10, total 10
page 2: +10, total 20
...
no more pages
done: 45 articles, saved to window.__articles
```

同时用 `console.table` 展示全部结果。跑完后可用：

| 命令 | 作用 |
| --- | --- |
| `window.__articles` | 结果数组（对象列表） |
| `copy(window.__articles)` | 复制 JSON 到剪贴板（`copy` 是 DevTools 内置函数，需在 Console 里手敲） |
| `__downloadCSV()` | 下载 `mp_articles.csv`（带 BOM，Excel 打开不乱码） |
| `__downloadJSON()` | 下载 `mp_articles.json` |
| `__markdown()` | 返回 `- [标题](链接) - 阅读数` 的 Markdown 列表字符串 |

翻页期间**不要切走标签页或手动点分页**，页面是 Vue 异步渲染，手动干预会打乱翻页判断。
45 篇 5 页大约 5 秒跑完。

## 四、实现要点

- **文章条目**：`.weui-desktop-mass-appmsg`，逐条抓取，一次群发多图文也能全部拿到
- **标题**：取 `<a class="weui-desktop-mass-appmsg__title">` 的**直接 `<span>` 子节点**。
  同一个 `<a>` 里还塞着 `原创`、`已修改` 标签，直接用 `innerText` 会把它们拼进标题
- **阅读数**：`.appmsg-view .weui-desktop-mass-media__data__inner`，`2,249` 这类带千分位
  的文本已转成数字
- **翻页**：按文本匹配分页条里的 `下一页` 链接，点击后**轮询
  `.weui-desktop-pagination__num_current` 的页码变化**来确认新页渲染完成（比死等
  `setTimeout` 稳），最多等 10 秒，超时就停
- **去重**：按 `url` 去重，翻页异常时不会重复累积
- **保险丝**：`MAX_PAGES = 100`，防止死循环
- 源码全部是 ASCII 字符（中文写成 `\uXXXX` 转义），避免复制粘贴时被编码问题弄坏

## 五、常见问题

**`Uncaught SyntaxError: Invalid or unexpected token`**
复制时混入了全角空格 / 不间断空格 / 智能引号。用上面的 Snippets 方式，从
`fetch_mp_articles.js` 文件本身复制，不要从渲染后的网页或终端输出里复制。

**只抓到当前页就停了**
分页条 DOM 变了，或「下一页」按钮的文案不同。改 `NEXT_PAGE` 常量，或看
`nextBtn()` 的选择器 `.weui-desktop-pagination a` 是否还匹配。

**`paging timed out, stopping`**
网络慢，页码 10 秒没变。调大 `while (curPage() === before && waited < 10000)`
里的 10000，或把翻页后的 `await sleep(600)` 调大。

**标题里带了「原创」「已修改」**
说明 `:scope > span` 没匹配到，微信改了 DOM 结构。检查 `grabPage()` 里的 `titleEl`。
