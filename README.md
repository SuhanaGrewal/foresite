# foresite
predictive kv-cache management for ai agents; optimizes context eviction for tight caches using agent execution signals to predict reuse of cached info

___

**The Problem**

Agent workloads (comprising multi-step tool use and multi-agent orchestration) grow KV caches fast, and re-send large amounts of overlapping context on every step. Existing eviction policies, including ones used by vLLM and SGLang decide what to evict based purely on recency (LRU) or frequency (LFU).
- Least Recently Used (LRU): KV cache evicts items that have not been accessed for the longest period of time.
- Least Frequently Used (LFU): KV cache evicts items that have the lowest usage count.

Neither uses any signal about what an agent is actually doing.

While LRU works with great accuraccy for large caches (where context doesn't have to be evicted) it fails when caches are smaller (whatever consitues a small cache size in our model). Higher cache pressure requires a different policy.

**The Core Idea**

A system predicts, for each item touched during an LLM agent's execution trace, whether that item will be reused later and uses that prediction to make smarter KV-cache eviction decisions than plain LRU.

It was however found the LRU works almost optimally with large cache sizes and so we came up with a 

Foresite uses a hybrid  behavioral signals,including an agent's position in a multi-agent dependency graph, its temporal access pattern, and the cost of recomputing its content can produce better eviction decisions than recency alone.

*
