# AutoSub

通用视频/音频本地 VAD 语音转录与 Profile 驱动单/双语字幕翻译工具。

引擎代码与业务领域完全解耦：针对特定场景（如 Twitch Dota 2 直播录像、技术讲座、播客、动漫等），只需编写一份 `profiles/<name>.yaml`（指定 Prompt 模板、术语表、目标语言与上下文变量）即可复用全套转录与翻译管线。

## 核心特性

1. **本地纯 ONNX Silero VAD 静音裁剪（零 PyTorch 依赖）**
   - 流式推理峰值内存 `< 80 MB`，可在 1C1G 低配机器上运行；将长视频中的无人声片段裁剪后打包上传（通常可缩减 45%~60% 音频时长），并精确映射回原视频时间轴。
   - 若未安装 `onnxruntime`，自动降级为零依赖 `ffmpeg silencedetect`。
2. **Gemini 词级时间戳转写（`gemini-3.5-transcribe`）**
   - 通过 Gemini Files API 上传分段音频并调用 `/v1beta/interactions`（`verbatim` + `word` 词级时间戳），自动在 `finally` 中清理远端临时文件，每段完成后实时写入 `.srt`。
3. **Profile + 多术语表驱动字幕翻译**
   - 内置 `default`（通用场景）与 `twitch_dota2`（Twitch 俄/英 Dota 2 直播，附带 126 全英雄与装备/机制术语表 `glossaries/dota2_ru_en_zh.yaml`），也支持传入任意外部 `.yaml` Profile 和 `--glossary` 附加术语表。
   - 支持运行时 `--var key=value` 注入上下文、单/双语切换（`--bilingual` / `--mono`）、均匀分批、紧凑 `{keep, drop}` JSON 防漏号/防串句护栏、以及每批完成即落盘。
4. **三级模型自动降级链**
   - 默认链：`gemini-4-hydrogen (standard) -> gemini-3.8-flash (standard) -> gemini-3.5-flash-lite (standard)`。
   - 遇到 `503` 高负载时当前模型原地尝试 3 次后再向下切换；遇到 `429` 当日 RPD 额度耗尽时当轮直接熔断跳过。

## 安装依赖

需要系统已安装 `ffmpeg` 和 `curl`，Python `>= 3.10`：

```bash
pip install -r requirements.txt

# 可选（强烈推荐）：纯 ONNX Silero VAD，无需安装数 GB 的 PyTorch
pip install onnxruntime numpy
pip install --no-deps silero-vad
```

配置 Gemini API Key（任选其一）：

```bash
export GEMINI_API_KEY="your-key"
# 或复制 .env.example 为 .env 并填入 GEMINI_API_KEY
```

## 快速开始

### 1. 查看内置 Profile

```bash
python3 main.py profiles
```

### 2. 通用视频一键转录 + 双语翻译（`default` Profile）

```bash
python3 main.py run demo.mp4 --var topic="系统架构分享" --var speaker="主讲人"
```

仅提取原语字幕或仅翻译已有 `.srt`：

```bash
python3 main.py transcript demo.mp4
python3 main.py translate demo.srt --target-lang "Simplified Chinese" --mono
```

### 3. Twitch Dota 2 直播录像（`twitch_dota2` Profile）

```bash
# 分步执行
python3 main.py transcript P02_死灵龙.mp4 --profile twitch_dota2
python3 main.py translate P02_死灵龙.srt --profile twitch_dota2 --var hero="Visage / 死灵龙" --var streamer="4oapm"

# 或一键执行
python3 main.py run P02_死灵龙.mp4 --profile twitch_dota2 --var hero="Visage / 死灵龙" --var streamer="4oapm"
```

### 4. 自定义任意领域的 Profile 与术语表

复制 `profiles/default.yaml` 为任意路径（例如 `my_podcast.yaml`），修改：
- `target_language` / `source_language_hint` / `bilingual` / `bilingual_order`
- `translate.glossaries`: 指向一份或多份术语表 YAML（任意一级分类名，格式为 `分类名 -> {目标词: [原语别名1, 别名2]}`）
- `translate.default_vars`: 定义模板变量默认值
- `translate.prompt_template`: 使用 `{变量名}` 与 `{glossary_section}`（JSON 示例里的花括号写 `{{` / `}}`）

然后直接指定外部路径运行：

```bash
python3 main.py translate episode01.srt --profile /path/to/my_podcast.yaml --var host="Lex"
```

## 离线测试

```bash
python3 -m unittest discover -s tests -v
```
