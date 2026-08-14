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

___

**The core idea**

A system that predicts, for each item touched during an LLM agent's execution trace, whether that item will be reused later and uses that prediction to make smarter KV-cache eviction decisions than plain LRU.

Given the finding that LRU works near-optimally with large cache sizes, the model was built into a hybdird system dynamically switching between the agentic behaviour-based predictor and LRU.

___

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
___

**What's actually here**

****1. Multi-agent orchestrator** (`orchestrator.py`):**
**- instead of one AI agent doing a task, this splits a task into smaller sub-tasks and hands them to separate mini-agents
- a "planner" computes which sub-tasks depend on each other using a dependency map DAG (Directed Acyclic Graph)
- sub-tasks that don't depend on each other run at the same time via `asynchio`, confirmed by checking real timestamps showing multiple agents starting within milliseconds of each other.
- a "synthesizer" combines each agents' results into one answer

**Local inference (`trace_agent.py`):** 
- AI model runs locally on the developer's laptop via Ollama
- every step is logged with real processing time.

**2. Tools:**

web_search grounded in:
- FRAMES (a published multi-hop QA dataset)
- real Wikipedia content
- a sandboxed Python executor
- a constrained SQLite lookup tool.

**3.Leakage-checked feature extraction pipeline (`feature_extraction.py`):**
- turns raw agent logs into a clean table a model can learn from.
- every piece of information fed to the model is re-verified to ensure it came from before the moment being predicted against a structurally future-blind recomputation before being trusted.

**4. Trained predictor (`train_predictor.py`):** 
- logistic regression baseline + XGBoost; trace-level train/test splitting; GroupKFold cross-validation, calibration checking.
A rigorous evaluation harness (`kv-cache-simulator.py`): benchmarks the predictor against LRU, LFU, and Belady's optimal (the theoretical unachievable-in-practice ceiling for the maximum best result) on real, held-out trace data.

- logistic regression: considers each signal, learns a weight for how much that signal pushes the prediction toward "will be reused" or "won't.", adds all weights and compresses the result into a probability between 0 and 1.
- XGBoost: a powerful model built from small decision trees (a chain of Y/N questions like "was this a model_call? if yes, is fan_out > 2?..."); builds one tree, mines incorrect decisions within that tree; builds a second tree focused on fixing those mistakes;
result = dozens of trees stacked together with each patching the previous ones' errors.
- GroupKFold cross-validation: splits real agent traces into groups; trains and tests model each time holding a different group as the test set. If the model performs consistently well across all these different holdouts; hence tests model's reliability

**Rigorous evaluation harness ('kv-cache-simulator.py'):** 
compares the trained predictor against 3 strategies to decide what to keep in memory:
- **`LRU:`** keep most recently used information
- **`LFU`:** keep most frequently used information
- **Belady's Algorithm:** A theoretical, perfect strategy that can see the complete future (of context reuse); impossible in real life, but useful as a ceiling to measure how close to ideal the predictor is.
___

**The actual result**
