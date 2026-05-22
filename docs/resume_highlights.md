# GustoBot 简历亮点（STAR 原则）

> 面向智能体（Agent）开发岗，6 条核心亮点。项目已从固定 DAG 工作流演进为完整 Multi-Agent + Supervisor 编排系统，并经过性能与可靠性深度优化。

---

## 1. Supervisor + 4 Worker 多 Agent 协作架构（含决策质量保障）

**S (情境):** 菜谱领域知识异构性强——图谱查询、统计聚合、深度推理、历史典故检索四类需求差异显著，单一 Agent 难以在所有子领域都达到高准确率。

**T (任务):** 设计多 Agent 协作系统，让每个 Agent 专注自己的领域，由 Supervisor 统一调度。同时解决 LLM 决策不可靠的问题——Supervisor 可能选错 Agent 导致空结果。

**A (行动):**
- 基于 LangGraph 构建 **Supervisor Agent**，通过 `bind_tools([handoff_to_kg/rag/sql/kb_agent])` 将 4 个 Worker Agent 作为"工具"暴露，LLM 自主决策调用哪几个
- 构建 4 个**领域专精 Worker Agent**：`kg_agent`（predefined_cypher + cypher_query，Neo4j 19669 道菜）、`rag_agent`（LightRAG 深度推理）、`sql_agent`（Text2SQL 统计）、`kb_agent`（Milvus + pgvector 知识库检索），每个 Worker 是独立 mini ReAct 子图
- **并行 Worker 执行**：通过 `asyncio.gather` 同时运行多个 Worker，总耗时 = max(单个耗时)，非累加
- **结构化决策记录**：设计 `SupervisorDecisionRecord` Pydantic 模型记录每轮决策（选了什么 Agent、理由、置信度），全程可回溯
- **代码层自动扩增**：单选 + 非高置信度时，通过 `_AUTO_FANOUT_AGENTS` 映射表自动补充互补 Agent（kg↔rag、sql→kg、kb→rag），防止单选失误
- **Prompt 输出前自检**：在 Router / Supervisor / ReAct Agent 三个 prompt 中嵌入 5 条自检规则，利用 LLM instruction-following 能力在决策前做思维审查
- **Agent 执行链路透传**：将 Supervisor 的决策日志（agents_used、rounds、confidence）注入 API 响应 `metadata.agent_trace`，前端可展示实际执行的 Agent

**R (结果):** 5-Agent 协作系统（1 Supervisor + 4 Workers），决策质量具备三层保障（Prompt 自检 → 代码层扩增 → LLM 兜底）。新增知识领域只需加一个 Worker + 一行 handoff 定义。决策记录全程可观测。

---

## 2. 从固定 DAG 到 ReAct Agent → Multi-Agent 的架构演进

**S (情境):** 原有 KG 查询采用固定 DAG 流程（Guardrails → Planner → Tool Selection → Summarize → Final Answer），Planner 一次性拆解任务后无法根据中间结果动态调整，缺少 Agent 的核心特征——"观察 → 反思 → 再行动"。

**T (任务):** 在不动摇原有架构的前提下，设计可切换的三种运行模式，支持灵活演进。

**A (行动):**
- 实现 **ReAct Agent 子图**：核心循环 `agent_node → tools_node → format_node → agent_node`，LLM 通过 `bind_tools` 自主决策调用哪个工具
- 将 4 个工具包装为 **LangChain Tool**（`@tool` 装饰器），每个工具是独立 async 函数
- 在 ReAct 基础上抽象出 **Multi-Agent Supervisor 模式**（`multi_agent/supervisor.py`），通过 `USE_MULTI_AGENT` 配置项切换
- 保留固定 DAG 作为 fallback，三种模式通过环境变量一键切换，零风险上线
- **子图编译缓存**：ReAct / Supervisor / KB 三个子图首次编译后缓存到模块级变量，消除每次请求的重复编译开销

**R (结果):** 一套代码支持三种运行模式（固定 DAG / 单 Agent ReAct / 多 Agent 协作），通过环境变量切换。当前运行 Multi-Agent 模式，Supervisor 自动将任务分发给领域专家。

---

## 3. 端到端性能优化（22s → 8s，-60%）

**S (情境):** Multi-Agent 模式下端到端响应延迟 22s+，严重影响用户体验。主要瓶颈：LightRAG hybrid 模式内部 3-4 次 LLM 调用（~11s）、Supervisor 过多轮次（10 轮上限）、Worker 过多迭代（3 轮）。

**T (任务):** 在不动架构的前提下，通过消除冗余 LLM 调用、减少轮次、预热连接等方式降延迟。

**A (行动):**
- **LightRAG 检索模式优化**：hybrid → local，消除内部 3-4 次 LLM 调用，检索耗时从 11s 降至 <1s
- **Agent 轮次精简**：Supervisor 10→2 轮、Worker 3→2 轮，通过环境变量 `.env` + `docker-compose.yml` 统一管控
- **启动预热机制**：在 FastAPI `startup_event` 中预初始化 LightRAG 单例，消除首次请求冷启动延迟
- **LightRAG 单例化**：模块级 `get_lightrag_api()` 单例，避免每次工具调用重新加载索引（原每次 `LightRAGAPI()` 新建实例 + `initialize_storages()`）
- **Neo4j 连接池复用**：`get_neo4j_graph()` 单例化，`Neo4jGraph` 内部自带连接池，消除每次建连开销（~500ms）
- **SQL Engine 连接池**：`_get_engine()` 缓存 SQLAlchemy Engine，不再每次 create/dispose
- **KB 搜索并行化**：PostgreSQL pgvector 和 Milvus 从串行改 `asyncio.gather` 并行查询，延迟从累加变 max
- **真流式输出**：用 `graph.astream(stream_mode="messages")` 替换原来的伪逐词延迟输出，首个 token 在 1-2s 内到达
- **节点进度事件**：在 SSE 流中发送节点转换进度（"正在分析问题..." → "正在查询菜谱知识图谱..."），降低用户等待焦虑

**R (结果):** 端到端延迟从 22s 降至 8s（-60%），首个 token 到达时间降至 1-2s。建立了系统性的性能优化方法论：定位瓶颈 → 消除冗余 LLM 调用 → 减少轮次 → 并行化 → 预热缓存。

---

## 4. 混合检索引擎 + 多级降级容错

**S (情境):** 菜谱知识存在于 Neo4j 图谱（19669 道菜）、LightRAG 索引、Milvus 向量库、PostgreSQL pgvector、MySQL 统计库等 5 个异构数据源中，且基础设施可能出现临时不可用（如 Milvus 启动延迟）。

**T (任务):** 设计多级检索 + 降级策略，确保任何单点故障不影响用户获得可用回答。

**A (行动):**
- **多数据源路由**：Supervisor 根据问题类型将任务分发给对应数据源的 Worker（图谱→kg_agent、推理→rag_agent、统计→sql_agent、向量→kb_agent）
- **三级降级链路**：Graph Agent（Neo4j / LightRAG）→ KB Agent（Milvus / pgvector）→ LLM 自身知识 + 免责声明
- **空结果检测 + 自动兜底**：`_looks_like_empty_result()` 检测 25+ 种空结果措辞模式，命中后自动触发 `_llm_standalone_answer()`
- **基础设施容错**：Milvus 不可用时降级路径自动跳过 KB 查询，直接走 LLM 兜底，不抛异常
- **超时保护**：`asyncio.wait_for(graph.ainvoke(), timeout=300)` 包裹，超时自动返回中文提示
- **Predefined Cypher 纯正则参数提取**：砍掉 LLM JSON 解析，改为正则直接提取菜名/食材/口味参数，消除 qwen3-max 不按格式输出的问题
- **`_extract_text_from_cypher_result` 格式兼容**：同时处理 dict/list 两种 records 格式，防止因数据格式不一致崩溃

**R (结果):** 系统在任何单点故障下均可返回可用回答——最坏情况由 LLM 自身知识兜底 + 免责声明。空结果检测自动触发，用户不感知基础设施故障。参数提取改为正则后零 LLM 调用、零失败。

---

## 5. GraphRAG + Text2Cypher + Text2SQL 多模态知识查询

**S (情境):** 菜谱知识具有高度异质性——实体关系适合图谱查询（如"宫保鸡丁用什么食材"），统计数据适合 SQL 聚合（如"川菜有多少道菜"），长文本故事适合语义推理（如"宫保鸡丁的历史典故"），单一查询范式无法覆盖全部场景。

**T (任务):** 构建统一的多工具查询入口，让 Agent 自动选择最优查询范式。

**A (行动):**
- 集成 **LightRAG GraphRAG**，支持 5 种检索模式，基于预构建的社区摘要图进行实体关系推理，相比 Microsoft GraphRAG 减少 99% token 消耗
- 实现 **Text2Cypher 三阶段管道**：Few-shot 检索 → LLM 生成 Cypher → 语法验证 → Neo4j 执行
- 构建 **Text2SQL 独立子图**：Schema 获取 → SQL 生成 → 验证 → 执行，带自动重试
- 预置 **20+ 条 Predefined Cypher 模板**（按菜品属性/食材关系/统计/营养 7 类别），**LLM 直接指定 query_name 跳过意图匹配**，比 TF-IDF 匹配更准更快
- TF-IDF 匹配器升级为 `char_wb` + ngram_range=(2,4)，修复原 word 级别分词对中文失效的问题

**R (结果):** 覆盖图谱推理、统计聚合、语义推理三大查询范式。Predefined Cypher 模板为高频查询提供毫秒级响应。Multi-Agent 下 Supervisor 根据问题类型自动分配给对应专家 Agent。

---

## 6. 工程化实践：流式输出、配置管理、可观测性

**S (情境):** Agent 系统面临长尾延迟、配置混乱、决策不可观测等工程挑战，需要生产级的基础设施支撑。

**T (任务):** 建立流式输出、统一配置管理、决策可观测性三大工程能力。

**A (行动):**
- **真 SSE 流式输出**：`graph.astream(stream_mode="messages")` 实现 token 级流式，过滤内部 ToolMessage 和子图消息，配合节点进度事件提升 UX
- **环境变量 + 特性开关统一管控**：`.env` → `docker-compose.yml` → `settings.py` 三层配置体系，`USE_MULTI_AGENT`、`LIGHTRAG_RETRIEVAL_MODE`、`MAX_ITERATIONS` 等 10+ 配置项可热切换
- **决策可观测性**：Supervisor 每轮决策记录 `SupervisorDecisionRecord`（agents、confidence、reasoning），通过 `agent_trace` 透传到前端
- **路由命名语义化**：`graphrag-query` → `recipe-query`、`text2sql-query` → `stats-query`、`kb-query` → `culture-query`，消除名称歧义
- **Docker Compose 9 服务编排**：Neo4j + MySQL + Redis + Milvus + etcd + MinIO + PostgreSQL + kb_ingest + backend，一键启动

**R (结果):** 完整的工程化体系——从配置管理到流式输出到可观测性，具备生产级 Agent 系统的基础能力。

---

## 7. 多层级会话管理与状态持久化

**S (情境):** Agent 多轮对话场景中，用户刷新页面或服务重启会导致上下文丢失，需从第一轮重新开始；同时高频相似问题反复触发完整 Agent 推理链路，造成不必要的 LLM 调用开销。

**T (任务):** 实现"图状态 Redis 持久化 + 对话历史 MySQL 永久存储 + Embedding 语义缓存"三层存储架构，保障多轮对话连续性并降低重复推理成本。

**A (行动):**
- **自研 RedisCheckpointer**：实现 LangGraph `BaseCheckpointSaver` 接口，以 `session_id` 为主键将图状态（messages、router、decision_history、steps）序列化存入 Redis，支持多轮对话中断恢复与服务重启无状态丢失；Redis 不可用时自动降级 `MemorySaver`，确保单点故障不影响核心链路
- **MySQL 永久持久化**：`chat_sessions`（会话元信息）+ `chat_messages`（逐条消息）+ `chat_history_snapshots`（轮次级快照）三表存储，支持历史对话回溯与前端渲染；生产环境 MySQL，开发环境 SQLite，SQLAlchemy ORM 屏蔽差异
- **Redis Embedding 语义缓存**：对用户问题计算 Embedding 向量，与 Redis 中已缓存问题的向量做余弦相似度匹配，超阈值直接返回缓存回答，单次命中省去 3-5 次 LLM 调用；配合 LRU 淘汰 + TTL 过期控制内存水位
- **DB 隔离与 Key 前缀隔离**：Checkpoint（DB 1, `checkpoint:*`）、语义缓存（DB 0, `semantic:*`）、会话历史（DB 0, `history:*`）三层 key 空间隔离，互不干扰

**R (结果):** 三层存储架构覆盖热状态（Redis Checkpoint，秒级恢复）、温缓存（语义缓存，省 LLM 调用）、冷记录（MySQL，永久归档）。Redis 不可用时自动降级，不影响核心问答链路。

---

## 8. 多层 LLM 缓存策略

**S (情境):** Agent 链路中每次请求都需 5-8 次 LLM 调用，其中 System Prompt（Router / Supervisor / Worker 指令合计 5000+ token）每次请求完全重复计费；高频相似问题（如"红烧肉怎么做"被不同用户反复问）重复触发完整推理链路，造成大量冗余 token 消耗。

**T (任务):** 设计覆盖"请求级 + 前缀级"的两层缓存，从不同粒度消除重复 LLM 推理开销。

**A (行动):**
- **Redis Embedding 语义缓存（请求级）**：以 `text-embedding-v4` 向量化用户问题 → Redis SCAN 遍历 `semantic:vec:*` 取所有缓存向量 → Python 逐条计算余弦相似度 → 最大值 ≥ 阈值（0.92）则命中，直接返回 `semantic:resp:*` 中已缓存的完整回答，跳过整条 Agent 链路（单次命中省去 5-8 次 LLM 调用）；未命中则走完整链路后将最终回答写入缓存。向量 / 回答 / 元数据按 `semantic:{vec|resp|meta}:{MD5}` 三 key 拆分存储，按访问模式分离冷热数据——遍历比对只读向量、命中后只读回答、淘汰只读元数据
- **LLM 自动前缀缓存（前缀级）**：编写 `prepend_system_prompt()` 工具函数，将 Router / Supervisor / Worker 的固定 System Prompt 统一置于 messages[0] 位置；利用 OpenAI 兼容 API 自动前缀缓存机制，服务端检测到相同前缀后复用 KV Cache，后续请求该前缀部分仅计 1/10 价格，日均节省约 90% 的 System Prompt token 开销；同时预埋 Anthropic `cache_control` 扩展点，切换供应商无需改调用方代码
- **工程保障**：语义缓存 TTL 12 小时 + LRU 淘汰（上限 1000 条）控制内存水位；Embedding API 异常时静默降级不阻塞主链路；Redis DB 隔离（DB 0 语义缓存 + 会话历史，DB 1 Checkpointer）避免 key 冲突

**R (结果):** 两层互补——语义缓存命中时直接免调 LLM（粗粒度，覆盖重复问题），前缀缓存未命中时仍省固定前缀的重复计算（细粒度，覆盖每次请求）。缓存对用户透明，命中后响应延迟从 8s 降至 50ms 以内。

---

## 技术栈速览

| 层次 | 技术 |
|------|------|
| **Agent 编排** | LangGraph StateGraph, Supervisor Pattern, asyncio.gather 并行, RedisCheckpointer |
| **Multi-Agent 架构** | 1 Supervisor + 3 Domain Workers (kg / knowledge / sql), Auto-Fanout |
| **知识图谱** | Neo4j (19669 菜谱节点), Cypher, LightRAG (GraphRAG), Text2Cypher |
| **向量检索** | Milvus, Reranker (Cohere / Jina / Voyage / BGE 可插拔), 双阈值过滤 |
| **结构化查询** | MySQL, Text2SQL, SQLAlchemy Connection Pool |
| **后端框架** | FastAPI, SSE Streaming, Redis Semantic Cache, Pydantic |
| **会话管理** | Redis Checkpointer（图状态持久化）, MySQL 三表持久化, DB 隔离 |
| **LLM 缓存优化** | 向量语义缓存（Embedding + 余弦相似度匹配）, 自动前缀缓存（Prompt Caching） |
| **工程化** | Docker Compose 9 服务编排, 三层配置体系 (.env→compose→settings), 特性开关 |
| **性能优化** | LLM 调用精简, 连接池复用, 单例模式, 子图编译缓存, 并行化, 缓存命中 |
| **容错降级** | 三级降级链路 (Graph→KB→LLM), Redis→Memory 自动降级, Embedding API 静默降级, 超时保护 |
