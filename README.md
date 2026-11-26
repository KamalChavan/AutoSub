# AutoSub

通用视频/音频本地 VAD 语音转录与 Profile 驱动单/双语字幕翻译工具。

引擎代码与业务领域完全解耦：针对特定场景（如 Twitch Dota 2 直播录像、技术讲座、播客、动漫等），只需编写一份 `profiles/<name>.yaml`（指定 Prompt 模板、术语表、目标语言与上下文变量）即可复用全套转录与翻译管线。

## 核心特性

1. **本地纯 ONNX Silero VAD 静音裁剪（零 PyTorch / 零 NumPy）**
   - 直接通过 Python `ctypes` 调用单份 `onnxruntime.dll` C API，峰值内存 `< 80 MB`，可在低配机上高速运行（约 200 倍实时速度），将长视频中的无人声片段裁剪后打包上传（通常缩减 35%~55% 音频时长），并毫秒级精确映射回原时间轴。
2. **Gemini 词级时间戳转写（`gemini-3.5-transcribe`）**
   - 通过纯 Python HTTPS（无需外部 `curl`）上传分段音频至 Gemini Files API，调用 `/v1beta/interactions`（`verbatim` + 词级时间戳），自动清理远端临时文件，每段完成后实时写入 `.srt`。
3. **Profile + 多术语表驱动字幕翻译**
   - 内置 `default`（通用场景）与 `twitch_dota2`（Twitch 俄/英 Dota 2 直播，附带 126 全英雄与装备/机制术语表 `glossaries/dota2_ru_en_zh.yaml`），也支持传入任意外部 `.yaml` Profile 和 `--glossary` 附加术语表。
   - 支持 `--var key=value` 注入上下文、单/双语切换（`--bilingual` / `--mono`）、均匀分批、防漏号/防串句护栏，每批完成即落盘。
4. **三级模型自动降级链**
   - 默认链：`gemini-4-hydrogen (standard) -> gemini-3.8-flash (standard) -> gemini-3.5-flash-lite (standard)`。
   - 遇到 `503` 高负载时当前模型原地尝试 3 次再向下切换；遇到 `429` 当日 RPD 耗尽时当轮直接熔断跳过。

---

## 安装与配置

### 1. Windows 安装包 / 绿色版（推荐，零外部依赖）

`AutoSub-Setup-<version>-win64.exe`（约 17 MB）与 `AutoSub-<version>-win64-portable.zip`（约 21 MB）已内置全部运行依赖（独立 Python、`onnxruntime.dll`、`silero_vad.onnx`、2.8 MB 纯音频静态 `ffmpeg.exe`），**无需安装 Python、PyTorch、FFmpeg 或 curl**。

- **安装包版**：双击 `AutoSub-Setup-*-win64.exe` 一路下一步（默认用户级安装至 `%LOCALAPPDATA%\Programs\AutoSub`，免管理员权限，自动把 `autosub.exe` 加入用户 `PATH`）。
- **绿色便携版**：解压 `.zip` 到任意目录，在该目录（或将其加入 `PATH`）运行 `autosub.exe`。

安装完成后打开任意 PowerShell / 命令提示符，运行自检确认环境就绪：

```powershell
autosub doctor
```

### 2. 配置 Gemini API Key（必填）与代理（可选）

按优先级依次读取，**任选一种方式即可**（推荐方式 ①，Windows 最方便，无需处理 `.env` 文件名）：

1. **方式 ①：直接填写 `config/defaults.yaml`（推荐）**
   - 在开始菜单点击 **AutoSub → Open AutoSub Config & Profiles Folder**（或进入安装目录 / 解压目录），用记事本打开 `config\defaults.yaml`，填入：
     ```yaml
     api:
       api_key: "你的_Gemini_API_Key"
       # 如需代理访问 Google API，可顺便填写（留空则自动继承系统代理）
       proxy: ""
     ```
   - 后续覆盖升级安装会自动保留你已修改的 `config\defaults.yaml`。

2. **方式 ②：设置 Windows 用户环境变量（一劳永逸）**
   ```powershell
   [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "你的_Gemini_API_Key", "User")
   ```
   设置后重新打开终端生效。

3. **方式 ③：使用 `.env` 文件**
   - 在安装目录、当前工作目录或 `%APPDATA%\AutoSub\` 下创建 `.env` 文件（注意文件名以点开头，PowerShell 可直接执行 `Set-Content .env "GEMINI_API_KEY=你的Key"`），内容参考 `.env.example`。

配置完成后再运行一次 `autosub doctor`，看到 `Gemini API Key: 已配置` 即表示就绪。

---

## 快速开始

> 以下示例以 Windows 安装版命令 `autosub` 为准；如果你通过 **Linux / macOS 源码运行**，把 `autosub` 替换为 `python3 main.py` 即可。

### 1. 查看内置 Profile

```bash
autosub profiles
```

### 2. 通用视频/音频一键转录 + 双语翻译（`default` Profile）

```bash
autosub run demo.mp4 --var topic="系统架构分享" --var speaker="主讲人"
```

也可以拆成两步分别执行（例如只提取原语字幕，或对已有 `.srt` 单语翻译）：

```bash
autosub transcript demo.mp4
autosub translate demo.srt --target-lang "Simplified Chinese" --mono
```

### 3. Twitch Dota 2 直播录像（`twitch_dota2` Profile）

```bash
# 一键执行（自动输出原语 .srt 和双语 _bilingual.srt）
autosub run "P02_死灵龙.mp4" --profile twitch_dota2 --var hero="死灵龙" --var streamer="4oapm"

# 或分步执行
autosub transcript "P02_死灵龙.mp4" --profile twitch_dota2
autosub translate "P02_死灵龙.srt" --profile twitch_dota2 --var hero="死灵龙" --var streamer="4oapm"
```

> `--var hero` 和 `--var streamer` 均为选填，不传会由模型从对话推断；但传入本局英雄和主播名可明显提升 ASR 同音纠偏与技能语境翻译准确度。

### 4. 自定义任意领域的 Profile 与术语表

复制安装目录下的 `profiles/default.yaml` 到任意位置（例如 `my_podcast.yaml`），按需修改：
- `target_language` / `source_language_hint` / `bilingual` / `bilingual_order`
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
