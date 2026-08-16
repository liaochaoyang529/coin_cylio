# Coin Challenge Multimodal Questioner

本仓库实现并评测一个多模态 Questioner。每个 episode 中，Questioner 只能看到一张候选图片和隐藏目标的文字描述；它可以向 Oracle 询问隐藏目标的视觉属性，或者判断当前候选是否与目标匹配。

当前默认实现是 `YourQuestioner`：Questioner 与 Oracle 均可使用豆包（火山方舟）图片理解模型，Questioner 额外维护带来源和极性的结构化证据，以减少无依据的正向判断和过度保守的负向判断。

详细设计、工程改进和限制见英文技术报告 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)，英文 PDF 版本见 [TECHNICAL_REPORT.pdf](TECHNICAL_REPORT.pdf)。

## 评分规则

| 操作 | 奖励 |
| --- | ---: |
| 正确结论 | `+10` |
| 错误结论 | `-10`，并结束当前 episode |
| 向 Oracle 提问 | `-1` |

```text
episode_reward = 10 * 正确结论数 - 10 * 错误结论数 - Oracle 提问数
```

正确结论会切换到下一个候选；只有所有候选都判断正确，episode 才算完整成功。环境默认在 60 步或 600 秒后截断。

## 快速开始

所有命令都应从仓库根目录运行，因为 episode 文件中的图片路径是仓库相对路径。

### 1. 创建环境

```bash
bash scripts/install.sh
```

该脚本创建 `coin_env/` 并安装评测依赖。

### 2. 准备图片

使用 Hugging Face CLI：

```bash
mkdir -p images
hf download --repo-type dataset e-zorzi/images_coin_challenge \
  --local-dir images --force-download
```

如果仓库中仍是 Git LFS 指针，或 Hugging Face 下载较慢，可以使用带校验的镜像下载器：

```bash
./coin_env/bin/python scripts/download_images_mirror.py
```

`env.py` 会主动拒绝 Git LFS 指针文件，避免把文本指针误当作 PNG。

### 3. 配置豆包

```bash
cp .env.example .env
```

至少填写：

```dotenv
ARK_API_KEY=your_ark_api_key
ARK_MODEL_ID=your_image_understanding_endpoint_id
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

`.env` 已被 Git 忽略，不要提交真实密钥。Questioner 默认读取 `QUESTIONER_MODEL_ID`，未设置时回退到 `ARK_MODEL_ID`；Oracle 默认使用 `ARK_MODEL_ID`。

### 4. 运行小切片

先运行付费较低的小切片，确认凭据、图片和端点正常：

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 3 \
  --split train \
  --description-type category \
  --oracle-provider doubao
```

范围是左闭右开，`0 3` 表示 episode 0、1、2。

### 5. 运行完整测试集

只运行 `category`：

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type category \
  --oracle-provider doubao
```

运行全部六种描述类型：

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type all \
  --oracle-provider doubao
```

支持 `category`、`color`、`context`、`color_context_feature`、`color_feature`、`color_context`；`all` 会依次执行全部六类。

完整 CLI 参数以实际代码为准：

```bash
./coin_env/bin/python eval_model.py --help
```

## 后台运行

完整六类型评测会产生较多 API 调用。需要关闭终端时，应在启动时使用 `nohup`，不要在运行中仅依赖 `disown`，因为进程输出可能仍绑定终端。

```bash
mkdir -p logs

nohup env QUESTIONER_PROVIDER=doubao \
  ./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type all \
  --oracle-provider doubao \
  > logs/test_all.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f logs/test_all.log
```

当前评测器只在一种 description type 全部完成后写结果，不支持逐 episode 自动断点续跑。中断当前类型会丢失该类型尚未写盘的进度。

## 输出与指标

结果写入：

```text
results/<QuestionerClass>_<description>_<split>_<start>_<end>.gzip.json
```

例如：

```text
results/YourQuestioner_category_test_0_16.gzip.json
```

gzip JSON 包含 episode ID、目标图、描述、动作、Oracle 回答、reasoning、结论/提问计数、奖励、完整成功状态和耗时。

控制台指标：

- `Decision accuracy`：正确结论数 / 总结论数
- `Oracle questions`：提问总数和 episode 平均数
- `Environment reward`：总奖励和 episode 平均奖励
- `Questioner time`：Questioner 平均模型耗时
- `Full-match rate`：所有候选均判断正确的 episode 比例

## 模型配置

### 豆包 / 火山方舟

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ARK_API_KEY` | 无 | 火山方舟 API Key，必填 |
| `ARK_MODEL_ID` | 无 | 支持图片理解的端点 ID，必填 |
| `ARK_BASE_URL` | 北京地域 API 地址 | OpenAI 兼容 API 地址 |
| `ARK_TIMEOUT_SECONDS` | `90` | 单次请求超时 |
| `ARK_MAX_OUTPUT_LENGTH` | `800` | 输出字符预算，约对应 200 token |
| `ARK_THINKING` | `disabled` | 方舟 thinking 参数 |
| `QUESTIONER_PROVIDER` | `doubao` | Questioner 后端 |
| `QUESTIONER_MODEL_ID` | `ARK_MODEL_ID` | Questioner 模型/端点 |
| `ORACLE_PROVIDER` | `doubao` | 未传 CLI 参数时的 Oracle 后端 |

Questioner 和 Oracle 使用同一豆包端点时，可以显式设置：

```bash
QUESTIONER_PROVIDER=doubao \
QUESTIONER_MODEL_ID=doubao-seed-2-1-turbo-260628 \
ARK_MODEL_ID=doubao-seed-2-1-turbo-260628 \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider doubao
```

模型名应替换为实际方舟端点 ID。

### Gemini

设置 `GEMINI_API_KEY`，并运行：

```bash
QUESTIONER_PROVIDER=gemini \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider gemini
```

`google-genai` 是可选依赖；默认豆包流程不要求安装它。

### 本地 vLLM

```bash
bash scripts/launch_qwen35_vllm.sh
```

另一个终端中运行：

```bash
QUESTIONER_PROVIDER=local \
QUESTIONER_MODEL_ID=local-qwen35-4b \
ORACLE_MODEL_ID=local-qwen35-4b \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider local
```

`ClientBasedLLM` 当前连接 `http://localhost:8000/v1`，模型名必须与 vLLM 的 `--served-model-name` 一致。

## 当前 Questioner 策略

`YourQuestioner` 的主要流程：

1. 豆包读取当前候选图，并输出紧凑 JSON 动作。
2. 类别明显不匹配或存在可靠冲突时直接判 `0`。
3. 同类别且信息不足时，询问候选特异的原子 Yes/No 问题。
4. Oracle 回答解析为 `yes`、`no` 或 `uncertain`，并绑定属性和候选来源。
5. 当前候选出现直接 `no` 时，控制器强制判 `0`。
6. 类别描述下，两个不同属性均为直接 `yes` 且无冲突时，控制器强制判 `1`。
7. 无效 JSON、重复问题或缺乏证据的结论最多修复一次，再执行受证据门控的回退。

默认记忆限制：

| 环境变量 | 默认值 |
| --- | ---: |
| `QUESTIONER_MAX_QUESTIONS_PER_CANDIDATE` | `2` |
| `QUESTIONER_MAX_EVIDENCE_ITEMS` | `12` |
| `QUESTIONER_MAX_MEMORY_CHARS` | `6000` |
| `QUESTIONER_API_ATTEMPTS` | `2` |
| `QUESTIONER_API_RETRY_DELAY` | `1` 秒 |

候选切换时清空候选状态，但目标证据在同一 episode 内保留。

## 接口约束

自定义 Questioner 必须继承 `QuestionerInterface` 并实现 `ask_or_conclude(observation)`。

```python
observation = {
    "image": np.ndarray,  # RGB, H x W x 3
    "answer": str | None,
}

action = {
    "question": str | None,
    "conclusion": 0 | 1 | None,
    "reasoning": str,
}
```

`question` 和 `conclusion` 必须恰好一个非 `None`。问题长度最多 300 字符，只能询问隐藏目标。评测器还会读取 `questions`、`answers`、`n_questions` 和 `time_required`。

自定义 Oracle 需要实现：

```python
def ask(*, prompt, images) -> str:
    ...
```

当前客户端每次多模态请求只支持一张图片。

## 数据划分

`split_manifest.json` 记录了 seed 2026 的 80/10/10 划分：

| 文件 | 数量 |
| --- | ---: |
| `episodes_train_split.jsonl` | 134 |
| `episodes_val.jsonl` | 17 |
| `episodes_test.jsonl` | 16 |

注意：`eval_model.py --split train` 当前读取 `episodes_train.jsonl`，不是 `episodes_train_split.jsonl`。严格实验不要把两者混用或覆盖原始数据。

## 验证代码改动

```bash
./coin_env/bin/python -m unittest discover -s tests

./coin_env/bin/python -m py_compile \
  env.py Questioner.py Oracle.py utils.py eval_model.py

./coin_env/bin/python eval_model.py --help
```

当前测试覆盖 Questioner 记忆、候选切换、证据门控、重复问题、输出修复、上下文预算和临时错误重试。

## 常见问题

### 图片无法打开

```bash
./coin_env/bin/python scripts/download_images_mirror.py
```

### 豆包返回 `429 SetLimitExceeded`

检查端点限流、并发额度、账户余额和模型配额。当前代码只重试连接错误和超时，不会无限重试额度或参数错误。

### 中途退出后结果缺失

结果只在当前 description type 完成后写入。建议使用小范围切片、按 description type 分开运行，或启动时使用 `nohup`。当前版本没有自动合并分片结果。

### 为什么详细描述模式奖励下降

固定两问预算在弱描述上较有价值，但在信息丰富的描述上可能产生不必要的 Oracle 成本。技术报告说明了该设计权衡和自适应提问方向。

## 安全与仓库卫生

- 不要提交 `.env`、API Key、结果、日志、虚拟环境或下载缓存。
- 不要提交下载后的真实图片替换，除非任务明确要求更新资产。
- `results/` 是本地生成目录，不是比赛源码。
- 提交前运行 `git diff --check` 和离线测试。
