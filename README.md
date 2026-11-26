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

## 安装方式

### 方式 A：Windows 一键安装包（推荐，免装 Python / Silero / FFmpeg）

Windows 安装包（`AutoSub-Setup-<version>-win64.exe`，约 **17 MB**）与绿色便携版（`AutoSub-<version>-win64-portable.zip`，约 **21 MB**）已内置全部运行依赖：
- 独立 Python 运行时 + `PyYAML` + `certifi`（纯 Python HTTPS，无需 `curl`）
- 单份 `onnxruntime.dll` C 库（通过纯 Python `ctypes` 直接调用 ONNX Runtime C API，彻底移除 `numpy` / `OpenBLAS`、重复 `.pyd` 和 PyTorch）
- Silero VAD ONNX 模型（`models/silero_vad.onnx`，2.1 MB）
- 音频精简静态版 `ffmpeg.exe`（`bin/ffmpeg.exe`，2.8 MB）
- 可直接编辑的 `config/defaults.yaml`、`profiles/*.yaml` 与 `glossaries/*.yaml`

安装后默认自动加入用户 `PATH`（无需管理员权限），打开任意 PowerShell / 终端即可直接运行：

```powershell
# 1. 环境自检（确认 FFmpeg、Silero ONNX 与内置 Profile 均就绪）
autosub doctor

# 2. 填入 Gemini API Key（任选其一）：
#    - 在安装目录（或 %APPDATA%\AutoSub\）新建 .env，写入 GEMINI_API_KEY=你的Key
#    - 或在系统环境变量设置 GEMINI_API_KEY

# 3. 直接转录与翻译
autosub run P02_死灵龙.mp4 --profile twitch_dota2 --var hero="Visage / 死灵龙" --var streamer="4oapm"
```

#### 如何构建 Windows 安装包

- **云端自动打包（推荐，默认使用 2.8 MB 音频精简版 FFmpeg + 零 NumPy）**：将 `AutoSub` 推送到 GitHub 后，在 Actions 页面手动触发 `Build Windows Installer`（或推送 `v0.1.0` 标签自动发布 GitHub Release），CI 会先自动交叉编译仅含音频编解码的极简 `ffmpeg.exe`（体积从 106 MB 降至 ~2.8 MB），再生成 `AutoSub-Setup-<version>-win64.exe`（约 17 MB）与绿色版 `.zip`（约 21 MB）。
- **在 Windows / WSL 本地一键打包**：安装 Python 3.10+ 与 [Inno Setup 6](https://jrsoftware.org/isinfo.php)（`winget install JRSoftware.InnoSetup`），在仓库根目录执行：

```powershell
# 可选（WSL/Linux 下先交叉编译 ~4–6 MB 精简音频版 ffmpeg.exe，避免打入 106 MB 完整版 FFmpeg）：
#   sudo apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 nasm make pkg-config curl tar xz-utils
#   bash packaging/windows/build_minimal_ffmpeg.sh

# 一键打包（若存在 build/cache/ffmpeg-minimal.exe 会自动优先使用；也可加 -NoBundleFfmpeg 复用本机 FFmpeg）
powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
```

脚本会自动创建隔离环境、运行全部单元测试（任何测试失败即中止打包）、打包 `onnxruntime` / `silero_vad.onnx` / `ffmpeg.exe`、运行 `autosub.exe doctor --strict-bundle` 验收，并在 `dist/` 目录生成 `.exe` 安装包与绿色 `.zip`。

### 方式 B：Linux / macOS / 源码方式安装

需要系统已安装 `ffmpeg`，Python `>= 3.10`（纯 Python HTTPS 请求，不再依赖外部 `curl`）：

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
