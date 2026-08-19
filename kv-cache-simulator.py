'''
Step 1: LRU vs LFU cache eviction simulator.
This simulates a fake sequence of "agent events" (each one representing the agent touching some piece of cached context), a small
fixed-size cache, and eviction policies to compare.

The eviction policies are:
- LRU (Least Recently Used): Evict KV blocks that haven't been recently used.
- LFU (Least Frequently Used): Evict KV blocks that have been least referred to (usually reference number = 0).
- Badely Algorithm: This is later incorporated as a baseline for the maximum/best possible hit rate.
- Predictor (run_predictor, Step 6): evicts using the trained cache-reuse model's predicted
  probability instead of a recency/frequency heuristic -- see run_predictor's docstring.
'''

import random
from collections import OrderedDict, defaultdict
CACHE_SIZE = 5

def generate_fake_events(num_events=200, num_items=15, seed=42):
    """
    Make up a sequence of item IDs, like an agent repeatedly touching
    the same pieces of context. 
    
    Some items are:
    - Hot: reused a lot; simulating an agent looping (e.g.: agent completing a goal will keep returning
    to the cache and appending it with new iterations via kv blocks).
    - Cold: used once; simulating a finished task (e.g.: agent returns the capital of a country after being asked once). 
    """
    random.seed(seed)
    items = [f"item_{i}" for i in range(num_items)]
    # give some items much higher probability of being picked (hot items)
    weights = [random.choice([1, 1, 1, 5, 10]) for _ in items]
    events = random.choices(items, weights=weights, k=num_events)
    return events


def run_lru(events, cache_size=CACHE_SIZE, record_decisions=False):
    """
    record_decisions=True additionally returns a list of per-eviction-step
    dicts ({"step": i, "evicted": item_id}), one entry per real eviction
    (cache full + miss) -- used by demo.py to compare LRU's choices against
    another policy's at the same trace positions. Default behavior/return
    type (a single float) is unchanged for every existing caller.
    """
    cache = OrderedDict()  # keeps insertion/access order
    hits = 0
    decisions = []
    for i, item in enumerate(events):
        if item in cache:
            hits += 1
            cache.move_to_end(item) # mark as most-recently-used
        else:
            if len(cache) >= cache_size:
                victim, _ = cache.popitem(last=False) # evict least-recently-used
                decisions.append({"step": i, "evicted": victim})
            cache[item] = True
    hit_rate = hits / len(events)
    return (hit_rate, decisions) if record_decisions else hit_rate


def run_lfu(events, cache_size=CACHE_SIZE):
    cache = set()
    freq = defaultdict(int)
    hits = 0
    for item in events:
        freq[item] += 1
        if item in cache:
            hits += 1
        else:
            if len(cache) >= cache_size: # evict the item in the cache with the lowest frequency
                victim = min(cache, key=lambda x: freq[x])
                cache.remove(victim)
            cache.add(item)
    return hits / len(events)

''' 
Result:
 - LRU hit rate: 65% 
 (65% times the information was available in the cache; 35% times had to be recomputed)
 - LFU hit rate: 73.% 
 (73.5% time the information was available in the cache; 26.5% times had to be recomputed)
 
For this simulation, the frequency of using cached information proved a better of context being reused, than recency.

This disparity sharpens when cache size reduced. Thus, a better eviction policy is important for smaller cache sizes.
'''

''' Belady Algorithm: Decides what to evict by looking ahead into the future.
- Evicts item that will be requested furthest away / never in the future.
- Possible here because we have the full events list; a live system could not do this.
- "Cheating" optimal policy: It sets the ceiling for the highest hit rate possible.
'''

def run_belady(events, cache_size=CACHE_SIZE):
    cache = set()
    hits = 0
    for i, item in enumerate(events): # i is the current item on the list
        if item in cache:
            hits += 1 
        else:
            if len(cache) >= cache_size:
                # for each item currently in the cache (i), find how far in
                # the future (from position i+1 onward) it's next used
                future = events[i + 1:] 
                next_use_distance = {}
                for cached_item in cache:
                    if cached_item in future:
                        next_use_distance[cached_item] = future.index(cached_item)
                    else:
                        next_use_distance[cached_item] = float("inf")  # never used again
                victim = max(cache, key=lambda x: next_use_distance[x])
                cache.remove(victim)
            cache.add(item)
    return hits / len(events)


''' Predictor Algorithm (Step 6): Decides what to evict using the trained
cache-reuse model's predicted probability, instead of a recency/frequency
heuristic or Belady's future-lookahead cheat.
- On each eviction decision, every item currently in the cache is scored via
  model.predict_proba on its precomputed feature vector (looked up by
  item_id in feature_lookup, produced by feature_extraction.py from real
  trace data -- NOT computed live here, since the model's features are
  historical/trace-derived, not something this simulator's fake events
  carry).
- Evicts whichever cached item has the LOWEST predicted reuse probability --
  the item the model is most confident won't be touched again soon.

LIMITATION, fixed by run_predictor_dynamic below: feature_lookup is a single
static vector per item_id, so however it was built, every touch of that item
scores identically for the whole run -- an item's score never reflects how
recently or how often it's actually been touched DURING this simulation.
Kept here as a general "score from a precomputed static lookup" primitive;
the benchmark (run_eviction_benchmark.py) uses the dynamic version instead.
'''

def run_predictor(events, cache_size, model, feature_lookup):
    cache = set()
    hits = 0
    for item in events:
        if item in cache:
            hits += 1
        else:
            if len(cache) >= cache_size:
                candidates = list(cache)
                try:
                    X = [feature_lookup[c] for c in candidates]
                except KeyError as e:
                    raise ValueError(
                        f"No precomputed features found for cached item {e}; "
                        "feature_lookup must cover every item_id that can appear in events."
                    ) from e
                scores = model.predict_proba(X)[:, 1]  # predicted probability of reuse
                victim = candidates[int(scores.argmin())]  # evict lowest predicted reuse probability
                cache.remove(victim)
            cache.add(item)
    return hits / len(events)


''' Predictor Algorithm, Dynamic (Step 6, fixed): re-scores an item's
predicted reuse probability EVERY time it's touched, not just once -- the
same "update the item's state on every touch" pattern LRU (recency) and LFU
(frequency) already use, instead of run_predictor's frozen static lookup.

- `touches` is a list of per-touch dicts: {"item_id": ..., "static_features":
  {feature_name: value, ...}}. static_features holds this specific touch's
  own real values for content_length_chars, token_count,
  measured_latency_seconds, dag_completion_fraction, agent_depth, fan_out,
  is_dependency_of_sink, event_type_* -- properties of WHAT this touch is
  (which message/tool-call/final-answer, from its own original trace),
  which don't have a coherent "live" redefinition when multiple unrelated
  traces are concatenated (e.g. dag_completion_fraction is "completed
  subtasks / total subtasks in THIS trace's plan" -- concatenating two
  different tasks' DAG-completion fractions into one number means nothing).
  static_features may also carry placeholder values for row_index/
  steps_since_last_seen/seconds_since_last_seen from this touch's ORIGINAL
  per-trace occurrence; those three are always overwritten below with
  values recomputed live against the combined sequence, never used as-is.
- row_index, steps_since_last_seen, and seconds_since_last_seen ARE
  recomputed live, from the combined sequence's own traversal state as
  touches replay in order -- these are the features that genuinely change
  meaning once multiple traces share one cache timeline.
- seconds_since_last_seen uses a simulated 1-second-per-touch clock (equal
  to steps_since_last_seen), since this simulator -- like LRU/LFU/Belady's
  above -- operates on abstract discrete touch-steps, not real wall-clock
  time. Concatenated traces' real timestamps come from unrelated runs at
  unrelated real times; diffing them across a trace boundary would produce
  an arbitrary number, not a meaningful "how long ago" for this shared cache.
- computed in two passes for efficiency: a first pass builds every touch's
  live feature row (pure bookkeeping, no model calls -- each touch's live
  features only ever depend on touches strictly before it, so this stays
  causal/backward-looking, same discipline feature_extraction.py's leakage
  check enforces), then ONE batched model.predict_proba call scores every
  touch at once, then a second pass replays the cache simulation using
  those precomputed scores. Equivalent to scoring one touch at a time, but
  far fewer individual model calls.
'''

def run_predictor_dynamic(touches, cache_size, model, feature_names, record_decisions=False):
    """
    record_decisions=True additionally returns a list of per-eviction-step
    dicts ({"step": i, "evicted": item_id, "predicted_score": float}) --
    same additive pattern as run_lru's record_decisions, default
    behavior/return type unchanged for every existing caller.
    """
    last_seen_position = {}
    live_rows = []
    for i, touch in enumerate(touches):
        item_id = touch["item_id"]
        steps_since_last_seen = i - last_seen_position[item_id] if item_id in last_seen_position else -1
        seconds_since_last_seen = float(steps_since_last_seen) if steps_since_last_seen != -1 else -1.0

        row = dict(touch["static_features"])
        row["row_index"] = i
        row["steps_since_last_seen"] = steps_since_last_seen
        row["seconds_since_last_seen"] = seconds_since_last_seen
        live_rows.append(row)

        last_seen_position[item_id] = i

    X = [[row[name] for name in feature_names] for row in live_rows]
    scores = model.predict_proba(X)[:, 1]  # one predicted-reuse-probability score per touch, in order

    cache = set()
    hits = 0
    decisions = []
    current_score = {}  # item_id -> most recent predicted score, updated every touch
    for i, touch in enumerate(touches):
        item_id = touch["item_id"]
        current_score[item_id] = scores[i]
        if item_id in cache:
            hits += 1
        else:
            if len(cache) >= cache_size:
                victim = min(cache, key=lambda x: current_score[x])  # evict lowest CURRENT predicted reuse probability
                cache.remove(victim)
                decisions.append({"step": i, "evicted": victim, "predicted_score": float(current_score[victim])})
            cache.add(item_id)
    hit_rate = hits / len(touches)
    return (hit_rate, decisions) if record_decisions else hit_rate

WORKING_SET_FRACTION = 0.2
PRESSURE_THRESHOLD = 0.3


def run_hybrid(
    touches,
    cache_size,
    model,
    feature_names,
    working_set_fraction=WORKING_SET_FRACTION,
    pressure_threshold=PRESSURE_THRESHOLD,
    record_decisions=False,
):
    """
    record_decisions=True additionally returns a list of per-eviction-step
    dicts ({"step": i, "evicted": item_id, "predicted_score": float,
    "mode": "predictor" or "lru_fallback", "pressure": float}) -- same
    additive pattern as run_lru/run_predictor_dynamic's record_decisions,
    default behavior/return type unchanged for every existing caller.
    predicted_score is current_score[victim] regardless of which mode
    fired, since it's tracked every touch either way -- useful context
    even on an lru_fallback eviction (shows what the predictor thought of
    the item LRU-recency ended up choosing anyway).
    """
    events = [t["item_id"] for t in touches]

    # same live-feature-building + one batched predict_proba pass as
    # run_predictor_dynamic, so the predictor half of this hybrid scores
    # identically to the standalone dynamic predictor -- not a separate,
    # possibly-inconsistent reimplementation
    last_seen_position = {}
    live_rows = []
    for i, touch in enumerate(touches):
        item_id = touch["item_id"]
        steps_since_last_seen = i - last_seen_position[item_id] if item_id in last_seen_position else -1
        seconds_since_last_seen = float(steps_since_last_seen) if steps_since_last_seen != -1 else -1.0

        row = dict(touch["static_features"])
        row["row_index"] = i
        row["steps_since_last_seen"] = steps_since_last_seen
        row["seconds_since_last_seen"] = seconds_since_last_seen
        live_rows.append(row)

        last_seen_position[item_id] = i

    X = [[row[name] for name in feature_names] for row in live_rows]
    scores = model.predict_proba(X)[:, 1]

    cache = set()
    hits = 0
    decisions = []
    current_score = {}
    last_touched_position = {}  # item_id -> most recent index touched, for the LRU fallback
    for i, item_id in enumerate(events):
        current_score[item_id] = scores[i]

        if item_id in cache:
            hits += 1
        else:
            if len(cache) >= cache_size:
                # working set: distinct items touched in a window sized as a
                # FRACTION OF THE TRACE'S OWN ELAPSED POSITION, independent of
                # cache_size -- see docstring for why this avoids the
                # saturation and scale-dependence failure modes of the two
                # earlier designs
                ws_window_size = max(1, int(working_set_fraction * i))
                working_set = events[max(0, i - ws_window_size) : i]
                working_set_size = len(set(working_set))
                pressure = cache_size / max(working_set_size, 1)

                if pressure <= pressure_threshold:  # LOW ratio: cache is small relative to the working set
                    mode = "predictor"
                    victim = min(cache, key=lambda x: current_score[x])  # predictor: lowest predicted reuse probability
                else:
                    mode = "lru_fallback"
                    victim = min(cache, key=lambda x: last_touched_position[x])  # LRU fallback: oldest last touch
                cache.remove(victim)
                decisions.append(
                    {
                        "step": i,
                        "evicted": victim,
                        "predicted_score": float(current_score[victim]),
                        "mode": mode,
                        "pressure": pressure,
                    }
                )
            cache.add(item_id)

        last_touched_position[item_id] = i

    hit_rate = hits / len(events)
    return (hit_rate, decisions) if record_decisions else hit_rate


if __name__ == "__main__":
    events = generate_fake_events()
 
    lru_hit_rate = run_lru(events)
    lfu_hit_rate = run_lfu(events)
    belady_hit_rate = run_belady(events)
 
    print(f"Total events simulated: {len(events)}")
    print(f"Cache size: {CACHE_SIZE} slots")
    print(f"LRU hit rate: {lru_hit_rate:.1%}")
    print(f"LFU hit rate: {lfu_hit_rate:.1%}")
    print(f"Belady's (optimal) hit rate: {belady_hit_rate:.1%}")
