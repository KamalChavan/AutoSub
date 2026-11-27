# AutoSub

通用视频/音频本地 VAD 多厂商语音转录与 Profile 驱动单/双语字幕翻译工具。

引擎代码与业务领域完全解耦：针对特定场景（如 Twitch Dota 2 直播录像、技术讲座、播客、动漫等），只需编写一份 `profiles/<name>.yaml`（指定 Prompt 模板、术语表、目标语言与上下文变量）即可复用全套转录与翻译管线；同时转录与翻译可自由混搭多家模型厂商。

## 核心特性

1. **本地纯 ONNX Silero VAD 静音裁剪（零 PyTorch / 零 NumPy）**
   - 直接通过 Python `ctypes` 调用单份 `onnxruntime.dll` C API，峰值内存 `< 80 MB`，可在低配机上高速运行（约 200 倍实时速度），将长视频中的无人声片段裁剪后打包上传（通常缩减 35%~55% 音频时长，直接降低按时长计费的 ASR 成本），并毫秒级精确映射回原时间轴。
2. **多厂商高精度原生词级时间戳转写（STT）**
   - **ElevenLabs Scribe v3 / v2（默认，`elevenlabs/scribe_v3`）**：`POST /v1/speech-to-text` 原生词级时间戳，支持中/英/俄混语次语言（`--secondary-lang`）、数字格式（`--numbers-format`）、去口癖（`--no-verbatim`）、说话人分离自动断句（`--diarize`）以及按需专有名词热词（`--keyterm`）。
   - **Google Gemini（`gemini/gemini-3.5-transcribe`）**：纯 Python HTTPS 上传至 Gemini Files API 并调用 `/v1beta/interactions`（`verbatim` + 词级时间戳），自动清理远端临时文件；既可作为独立转录模型，也适合作为双 ASR 交叉仲裁参考轨。
   - **OpenAI / OpenAI 兼容 Whisper 端点**：支持返回 `verbose_json` 词/段时间戳的兼容接口（如 `whisper-1`、Groq Whisper、自建 Faster-Whisper）；自动拒绝无时间戳的 `gpt-transcribe` 以避免字幕漂移。
   - 全程零重型第三方 SDK（仅依赖 `PyYAML` + 标准库 `urllib.request`），每段完成后实时增量落盘 `.srt`，并默认在同目录保存原时间轴词级时间戳 `.words.json`（可用 `--no-words-json` 关闭）。
3. **Profile + 多术语表驱动字幕翻译（LLM）**
   - 内置 `default`（通用场景）与 `twitch_dota2`（Twitch 俄/英 Dota 2 直播，附带 126 全英雄与装备/机制术语表 `glossaries/dota2_ru_en_zh.yaml`），也支持传入任意外部 `.yaml` Profile 和 `--glossary` 附加术语表。
   - 支持 `--var key=value` 注入上下文、单/双语切换（`--bilingual` / `--mono`）、按最大静音间隙智能分批、防串句护栏与轻量 JSON 容错，每批完成即落盘。
4. **多厂商跨模型自动降级链**
   - **Google Gemini（默认翻译）**：`gemini-4-hydrogen -> gemini-3.8-flash -> gemini-3.5-flash-lite`（全向后兼容旧版 `api.*` 配置）。
   - **OpenAI（`openai/gpt-6.2-sol`、`openai/gpt-6-luna` 等）**：默认走 `/v1/responses`（将固定系统提示置为 `input` 首条 `developer` 消息以命中 GPT-6.2 Prompt Caching，推理深度默认 `low`），也支持切换 `/v1/chat/completions`。
   - **OpenAI 兼容网关（`openai_compatible`）**：支持 DeepSeek、OpenRouter、Groq、DashScope、Ollama、vLLM 等任意自定义厂商别名。
   - 支持跨厂商混排降级（如 `openai/gpt-6.2-sol,gemini/gemini-4-hydrogen`）：遇到 `503` 高负载时当前模型原地尝试 3 次再向下切换；遇到 `429` / `402` 额度耗尽时当轮直接熔断跳过。
5. **高质量字幕增强管线（词级语义重断句 + 双 ASR 交叉仲裁 + Pass 0 + 自修复 + 排轴与折行）**
   - **词级词索引语义重断句（`.words.json`，默认有文件即自动启用）**：按长静音停顿与说话人切换划分声学语音段，仅将超长/多句/带停顿候选段以 `<id>:<word>` 形式送给 LLM 输出紧凑切分词 ID（`{"splits": ...}`），配合 1 轮越界自修复与规则兜底，**零文本篡改、零时间轴漂移**地消除 ASR 原始长难句与错位碎句（支持 `autosub resegment` 独立运行，或 `--no-resegment` 关闭）。
   - **双 ASR 交叉仲裁**：可在 `config/defaults.yaml` 中直接开关 `transcript.ref_enabled: true`（一次 VAD 切片，并行调用主轨 `elevenlabs/scribe_v3` 与参考轨 `gemini-3.5-transcribe`），经拼写微差降噪后将有效分歧作为 `alt_asr` 喂给翻译模型仲裁纠偏。
   - **Pass 0 全片上下文分析（`--auto-context`）**：分批翻译前通读全片提取概要、人名/外号、出场实体与高频 ASR 错词，注入固定前缀供所有批次缓存复用。
   - **漏译 ID 1 轮精确自修复（默认开启）**：检测到模型偶发漏译或无效跨间隔合并的 ID 时，复用相同系统提示（命中 Prompt Caching）仅向模型回传缺失条与邻近上下文做 1 轮精准补译，消除静默回退原文。
   - **语义碎句合并 + 确定性 CPS 排轴 + 视觉平衡折行**：前端吸收 1~2 词孤儿尾词并拦截单字异常拉伸；翻译阶段支持短停顿碎句语义合并（时间戳严格锚定原词边界）；落盘前自动执行最短驻留、CPS 读速延展与微间隙桥接（零时间轴重叠），并对超长单行译文（默认 `> 24` 等效中文字符）按标点与词边界自动视觉平衡折行。
6. **零依赖本地 SQLite 内容哈希缓存 & 可选调试日志**
   - 使用标准库 `sqlite3` 按内容哈希缓存各段 ASR、词级重断句、Pass 0 与各批翻译结果：网络中断续跑、单轨升级双 ASR、或调整折行/单双语/排轴参数重跑时秒级命中本地缓存，避免向 ElevenLabs / OpenAI / Gemini 重复扣费（支持 `autosub cache stats/clear` 管理与 `--no-cache` 临时绕过）。
   - 支持通过 `--log-llm` 或 `logging.llm_requests: true` 记录轮转 JSONL 请求/响应日志（不含 API Key），方便排查具体坏句。

---

## 安装与配置

### 1. Windows 安装包 / 绿色版（推荐，零外部依赖）

`AutoSub-Setup-<version>-win64.exe`（约 17 MB）与 `AutoSub-<version>-win64-portable.zip`（约 21 MB）已内置全部运行依赖（独立 Python、`onnxruntime.dll`、`silero_vad.onnx`、2.8 MB 纯音频静态 `ffmpeg.exe`），**无需安装 Python、PyTorch、FFmpeg 或 curl**。

- **安装包版**：双击 `AutoSub-Setup-*-win64.exe` 一路下一步（默认用户级安装至 `%LOCALAPPDATA%\Programs\AutoSub`，免管理员权限，自动把 `autosub.exe` 加入用户 `PATH`）。
- **绿色便携版**：解压 `.zip` 到任意目录，在该目录（或将其加入 `PATH`）运行 `autosub.exe`。

安装完成后打开任意 PowerShell / 命令提示符，运行自检确认环境就绪：

```powershell
autosub doctor
autosub providers
```

### 2. 配置各厂商 API Key 与代理（按需配置）

默认语音转录使用 **ElevenLabs Scribe v3**（`ELEVENLABS_API_KEY`），默认字幕翻译使用 **Google Gemini**（`GEMINI_API_KEY`，100% 向后兼容旧版 `api.*` 配置）；如需改用 **Gemini 转录** 或 **OpenAI `gpt-6.2-sol` 翻译**，按需补充对应 API Key 即可。

| 厂商 (`provider`) | 协议 (`protocol`) | 能力 | 默认环境变量 | 推荐模型示例 |
| :--- | :--- | :--- | :--- | :--- |
| `elevenlabs`（默认转录） | `elevenlabs` | STT | `ELEVENLABS_API_KEY` | `elevenlabs/scribe_v3`、`elevenlabs/scribe_v2` |
| `gemini`（默认翻译） | `gemini` | STT + LLM | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `gemini-3.5-transcribe` / `gemini-4-hydrogen` |
| `openai` | `openai` | LLM + Whisper STT | `OPENAI_API_KEY` | `openai/gpt-6.2-sol`、`openai/gpt-6-luna` |
| `openai_compatible` | `openai_compatible` | LLM + Whisper STT | `OPENAI_COMPAT_API_KEY` | DeepSeek / OpenRouter / Groq / Ollama 等 |

按优先级依次读取，**任选一种方式即可**：

1. **方式 ①：直接填写 `config/defaults.yaml`（推荐，Windows 最方便）**
   - 在开始菜单点击 **AutoSub → Open AutoSub Config & Profiles Folder**（或进入安装/解压目录），打开 `config\defaults.yaml`，在 `providers:` 下按需填写对应 `api_key`（旧版 `api:` 块仍自动兼容）：
     ```yaml
     network:
       proxy: ""
       timeout_seconds: 180

     providers:
       elevenlabs:
         api_key: "你的_ElevenLabs_API_Key"
       gemini:
         api_key: "你的_Gemini_API_Key"
       openai:
         api_key: "你的_OpenAI_API_Key"
     ```
   - 后续覆盖升级安装会自动保留你已修改的 `config\defaults.yaml`。

2. **方式 ②：设置环境变量（一劳永逸）**
   ```powershell
   [Environment]::SetEnvironmentVariable("ELEVENLABS_API_KEY", "你的_ElevenLabs_API_Key", "User")
   [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "你的_Gemini_API_Key", "User")
   [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "你的_OpenAI_API_Key", "User")
   ```
   设置后重新打开终端生效。

3. **方式 ③：使用 `.env` 文件**
   - 在安装目录、当前工作目录或 `%APPDATA%\AutoSub\` 下创建 `.env` 文件，参考 `.env.example`。
   - **ElevenLabs 快捷提示**：如果本机已安装 Node.js，也可运行 `npx -y @elevenlabs/cli@latest onboard init` 通过浏览器快速获取 ElevenLabs API Key 或配置其 MCP，再将 Key 写入 `.env` 的 `ELEVENLABS_API_KEY`。

配置完成后运行 `autosub doctor` 或 `autosub providers` 确认各厂商状态。

---

## 快速开始

> 以下示例以 Windows 安装版命令 `autosub` 为准；如果你通过 **Linux / macOS 源码运行**，把 `autosub` 替换为 `python3 main.py` 即可。

### 1. 查看已配置厂商与内置 Profile

```bash
autosub providers
autosub profiles
```

### 2. 通用视频/音频一键转录 + 双语翻译（默认 Scribe v3 转录 + Gemini 翻译）

```bash
autosub run demo.mp4 --var topic="系统架构分享" --var speaker="主讲人"
```

也可以拆成两步分别执行（例如只提取原语字幕，或对已有 `.srt` 单语翻译）：

```bash
autosub transcript demo.mp4
autosub translate demo.srt --target-lang "Simplified Chinese" --mono
```

如需改用 Gemini 转录，只需通过 `--model` / `--transcript-model` 覆盖：

```bash
autosub transcript demo.mp4 --model gemini/gemini-3.5-transcribe
```

### 3. 高阶多厂商混搭：Scribe v3（默认）+ OpenAI `gpt-6.2-sol` + 双 ASR 交叉仲裁

```bash
# 一键混搭：默认 Scribe v3 转录 -> gpt-6.2-sol 翻译（429/503 自动降级至 gemini-4-hydrogen）
autosub run "P02_死灵龙.mp4" \
  --profile twitch_dota2 \
  --translate-model openai/gpt-6.2-sol,gemini/gemini-4-hydrogen \
  --reasoning-effort medium \
  --auto-context

# 单独使用 Scribe v3 转录（支持次语言、说话人分离、去口癖与按需热词）
autosub transcript "P02_死灵龙.mp4" \
  --profile twitch_dota2 \
  --lang rus --secondary-lang eng --no-verbatim --numbers-format digits

# 单独使用 OpenAI gpt-6.2-sol 翻译已有 SRT（开启 Pass 0 全片上下文分析 + 碎句语义合并 + CPS 排轴优化）
autosub translate "P02_死灵龙.srt" \
  --profile twitch_dota2 \
  --model openai/gpt-6.2-sol \
  --reasoning-effort medium \
  --auto-context

# 高阶：双 ASR 交叉仲裁（对已有两份 SRT 翻译）
autosub translate "elevenlabs.srt" \
  --ref-srt "gemini_35_transcribe.srt" \
  --profile twitch_dota2 \
  --model openai/gpt-6.2-sol \
  --reasoning-effort medium \
  --auto-context

# 高阶：一条命令临时开启双 ASR 并行转录（默认 Scribe v3 主轨 + Gemini 参考轨）+ 仲裁翻译
autosub run "P02_死灵龙.mp4" \
  --profile twitch_dota2 \
  --ref-transcript-model gemini-3.5-transcribe \
  --translate-model openai/gpt-6.2-sol \
  --reasoning-effort medium \
  --auto-context
# 独立运行词级语义重断句（对已生成的 .words.json 输出零漂移原语 .srt）
autosub resegment "P02_死灵龙.words.json" \
  --model openai/gpt-6.2-sol \
  --reasoning-effort high
```

如果希望**默认常开**双 ASR 与 Pass 0 全片上下文，或微调重断句/折行/缓存/调试日志，可直接修改 `config/defaults.yaml`：

```yaml
transcript:
  provider: "elevenlabs"
  model: "scribe_v3"
  save_words_json: true  # 保存同目录 .words.json 词级时间戳（默认开启）
  ref_enabled: true
  ref_model: "gemini-3.5-transcribe"

translate:
  resegment: true        # 当存在 .words.json 时自动词级语义重断句（可 --no-resegment 关闭）
  resegment_max_cue_chars: 28
  auto_context: true
  merge_cues: true
  polish_timing: true
  repair_missing: true
  max_line_chars: 24     # 超 24 等效中文字符自动视觉平衡折行；设为 0 关闭

cache:
  enabled: true          # 本地 SQLite 内容哈希缓存（默认开启，TTL 30 天）
  ttl_days: 30

logging:
  llm_requests: false    # 设为 true 或加 --log-llm 开启轮转 JSONL 调试日志
```

配置好后只需运行 `autosub run "vod.mp4" -p twitch_dota2` 即可全自动跑完（需要单次临时关闭双 ASR、重断句或缓存时可追加 `--no-ref-transcript` / `--no-resegment` / `--no-cache`）。

查看或清理本地 SQLite 缓存：

```bash
autosub cache stats
autosub cache clear                        # 清理全部缓存
autosub cache clear --namespace resegment  # 按命名空间清理（支持 all / asr / resegment / translate / pass0）
```

> 提示：
> - ElevenLabs `keyterms`（`--keyterm`）会增加 `+25%` 计费，默认关闭，仅在人名/专业词识别困难时按需传入。
> - 翻译阶段默认开启**词级语义重断句**（有 `.words.json` 时自动生效，`--no-resegment` 关闭）、**相邻碎句语义合并**（`--no-merge-cues` 关闭）、**漏译 ID 1 轮精确补译**（`--no-repair-missing` 关闭）、**CPS 排轴优化**（`--no-polish-timing` 关闭）与**单行超 24 字视觉平衡折行**（`--max-line-chars 0` 关闭）。

### 4. Twitch Dota 2 直播录像（`twitch_dota2` Profile）

```bash
# 一键执行（自动应用俄/英混语参数，输出原语 .srt、.words.json 和双语 _bilingual.srt）
autosub run "P02_死灵龙.mp4" --profile twitch_dota2 --var hero="死灵龙" --var streamer="4oapm"

# 或分步执行（translate 会自动发现同目录 .words.json 做词级重断句）
autosub transcript "P02_死灵龙.mp4" --profile twitch_dota2
autosub translate "P02_死灵龙.srt" --profile twitch_dota2 --var hero="死灵龙" --var streamer="4oapm"
```

> `--var hero` 和 `--var streamer` 均为选填，不传会由模型从对话推断；但传入本局英雄和主播名（或开启 `--auto-context`）可明显提升 ASR 同音纠偏与技能语境翻译准确度。

### 5. 自定义任意领域的 Profile 与术语表

复制安装目录下的 `profiles/default.yaml` 到任意位置（例如 `my_podcast.yaml`），按需修改：
- `target_language` / `source_language_hint` / `bilingual` / `bilingual_order`
- `transcript.language_code` / `transcript.secondary_language_codes` / `transcript.save_words_json` / `transcript.ref_enabled` / `transcript.ref_model`
- `translate.resegment` / `translate.resegment_max_cue_chars` / `translate.auto_context` / `translate.merge_cues` / `translate.polish_timing` / `translate.repair_missing` / `translate.max_line_chars`
- `translate.glossaries`：指向一份或多份术语表 YAML（格式：`分类名 -> {目标词: [原语别名1, 别名2]}`）
- `translate.default_vars`：自定义上下文变量默认值
- `translate.prompt_template`：使用 `{变量名}` 与 `{glossary_section}`（JSON 示例里的花括号写 `{{` / `}}`）

然后直接指定外部路径运行：

```bash
autosub translate episode01.srt --profile D:\profiles\my_podcast.yaml --var host="Lex"
```

---

## 开发者指南：源码运行与重新打包

### Linux / macOS 源码安装

需要系统已安装 `ffmpeg`，Python `>= 3.10`：

```bash
pip install -r requirements.txt

# 可选（强烈推荐）：纯 ONNX Silero VAD（通过 ctypes 直调 C 库，无需 NumPy 或 PyTorch）
pip install --no-deps onnxruntime silero-vad
```

离线单元测试：

```bash
python3 -m unittest discover -s tests -v
```

### 构建 Windows 安装包

- **GitHub Actions 云端自动构建**：推送到 GitHub 后手动触发 `Build Windows Installer`（或推送 `v*` 标签自动发布 Release），自动编译 2.8 MB 纯音频 `ffmpeg.exe` 与 17 MB 安装包。
- **本地 Windows / WSL 一键构建**（需 Python 3.10+ 与 Inno Setup 6）：
  ```powershell
  # 可选（WSL/Linux 下先交叉编译 2.8 MB 精简音频版 ffmpeg.exe，避免打入 106 MB 完整版 FFmpeg）：
  #   bash packaging/windows/build_minimal_ffmpeg.sh

  # 构建绿色 ZIP 与 .exe 安装包（若已生成 build/cache/ffmpeg-minimal.exe 会自动优先使用）
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
  ```
