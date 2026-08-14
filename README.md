# foresite
predictive kv-cache management for ai agents; adaptively optimizes context eviction for caches based on cache pessure, using a mix of agent execution signals and recency to predict reuse of cached info

___

**The problem**

Agent workloads (comprising multi-step tool use and multi-agent orchestration) grow KV caches fast, and re-send large amounts of overlapping context on every step. Existing eviction policies, including ones used by vLLM and SGLang decide what to evict based purely on recency (LRU) or frequency (LFU).
- Least Recently Used `(LRU)`: KV cache evicts items that have not been accessed for the longest period of time.
- Least Frequently Used `(LFU)`: KV cache evicts items that have the lowest usage count.

Neither uses any signal about what an agent is actually doing.

LRU performs near-optimally when the cache is large relative to the working set: there's enough room that eviction decisions barely matter. 
Its accuracy degrades as cache pressure increases: at small cache sizes (roughly 1–5% relative to distinct items in play), LRU is forced into frequent, consequential eviction, and recency alone becomes a poor predictor of what's needed next.

**The core idea**

A system that predicts, for each item touched during an LLM agent's execution trace, whether that item will be reused later and uses that prediction to make smarter KV-cache eviction decisions than plain LRU.

Given the finding that LRU works near-optimally with large cache sizes, the model was built into a hybdird system dynamically switching between the agentic behaviour-based predictor and LRU.

**The solution**

A hybrid eviction policy that measures live cache pressure (smaller cache size relative to number of items represents higher pressure) and adaptively switches between the predictor and standard LRU.

To predict context reuse, _foresite_ computes agents' behavioural signals including:

_Content/size signals_
- `content_length_chars`: raw character length of the content
- `token_count`: real token count (via tokenizer)

__Temporal signals_
- `steps_since_last_seen`: how many events since this exact content last appeared (-1 if first occurrence)
- `seconds_since_last_seen`: real wall-clock time since last touch
- `measured_latency_seconds`: how long the model call that produced this content took

_Structural/graph signals
- `dag_completion_fraction`: how much of the overall multi-agent task was complete at this point
- `agent_depth`: hops from the planner (0 = root, 1 = sub-agent)
- `fan_out`: how many other agents depend on this one's output
- `is_dependency_of_sink`: whether this feeds into the task's final synthesized answer

_Event type_
- `event_type_model_call`
- `event_type_tool_call`
- `event_type_final_answer`

_Positional signal_
- `row_index`: position within the trace

_Runtime signal (hybrid policy only, not the base predictor)_

A live cache-pressure measure (cache size relative to recently observed distinct items), used ONLY to decide whether to consult the predictor or LRU.

**What's actually here**
- A real multi-agent orchestrator (`orchestrator.py`): a planner decomposes a task into a dependency DAG (Directed Acyclic Graph), independent sub-tasks run as concurrent agents (verified via millisecond-level spawn timestamps), and a synthesizer combines results.
Real local inference (`trace_agent.py`): runs on Ollama locally; every step is logged with real engine-reported timing.
Real tools: web_search grounded in FRAMES (_a published multi-hop QA dataset_) and real Wikipedia content; a sandboxed Python executor; a constrained SQLite lookup tool.
A leakage-checked feature extraction pipeline (`feature_extraction.py`): every backward-looking feature is independently re-verified against a structurally future-blind recomputation before being trusted.
A trained predictor (  train_predictor.py  ): logistic regression baseline + XGBoost; trace-level train/test splitting; GroupKFold cross-validation, calibration checking.
A rigorous evaluation harness (`kv-cache-simulator.py`): benchmarks the predictor against LRU, LFU, and Belady's optimal (the theoretical unachievable-in-practice ceiling for the maximum best result) on real, held-out trace data.
