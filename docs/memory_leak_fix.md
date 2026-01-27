# Memory Leak Investigation and Fixes

## Overview

This document describes the memory management improvements made to the RQ worker in docling-serve to address memory growth issues during long-running document conversion workloads.

## Problem Statement

RQ workers processing document conversions were experiencing continuous memory growth, eventually leading to OOM (Out of Memory) kills in Kubernetes environments. Memory would grow unbounded as jobs were processed, with no mechanism to release cached models and pipeline state.

## Root Cause Analysis

The docling library caches document converters and pipelines for performance optimization. However, in a long-running worker process:

1. **Converter Cache**: `DocumentConverter` instances are cached by configuration hash
2. **Pipeline Cache**: Processing pipelines with loaded ML models are retained
3. **No Automatic Cleanup**: These caches persist for the lifetime of the worker process

Each unique conversion configuration creates new cached entries, and even with identical configurations, internal state accumulates over time.

## Fixes Implemented

### 1. Periodic Cache Clearing (commit 6b01ac8)

Added a configurable job counter that triggers cache clearing after N jobs.

**File**: `docling_serve/rq_worker_instrumented.py`

```python
class InstrumentedWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._jobs_processed = 0
        self._jobs_between_cache_clear = int(
            os.environ.get("DOCLING_SERVE_JOBS_BETWEEN_CACHE_CLEAR", "3")
        )
```

**Cache Clear Logic**:
```python
def _clear_memory_caches(self, job_id: str) -> None:
    """Clear docling's internal caches to free memory."""
    import gc
    from docling.document_converter import DocumentConverter

    mem_before = self._get_memory_mb()

    # Clear converter cache
    converters_cleared = 0
    if hasattr(DocumentConverter, "_instances"):
        converters_cleared = len(DocumentConverter._instances)
        DocumentConverter._instances.clear()

    # Clear pipeline caches
    pipelines_cleared = 0
    if hasattr(DocumentConverter, "_pipeline_cache"):
        pipelines_cleared = len(DocumentConverter._pipeline_cache)
        DocumentConverter._pipeline_cache.clear()

    # Force garbage collection
    gc.collect()

    mem_after = self._get_memory_mb()
    logger.info(
        f"[CACHE CLEAR] jobs_since_last_clear={self._jobs_between_cache_clear} | "
        f"converters={converters_cleared} | pipelines={pipelines_cleared} | "
        f"memory={mem_after}MB (freed {mem_before - mem_after}MB)"
    )
```

### 2. Memory Monitoring (commit 49b7cb7)

Added per-job memory tracking to observe memory behavior over time.

```python
def _get_memory_mb(self) -> int:
    """Get current process RSS memory in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss // (1024 * 1024)
    except Exception:
        return 0
```

Memory is logged after each job:
```
[MEMORY] job_id=xxx | rss=1115MB | jobs_total=1
```

### 3. Logging Configuration (commits b268ef4, 1a6b304)

Fixed logging to ensure memory metrics are visible:

```python
# Configure logging BEFORE any imports
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True  # Override any existing configuration
)
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_JOBS_BETWEEN_CACHE_CLEAR` | `3` | Number of jobs between cache clears |

## Results

### Cache Clearing Effectiveness

The cache clear successfully frees memory from docling's internal caches:

| Clear Event | Memory Before | Memory After | Freed |
|-------------|---------------|--------------|-------|
| Job 3       | 1109 MB       | 859 MB       | 250 MB |
| Job 6       | 1306 MB       | 1066 MB      | 240 MB |
| Job 9       | 1485 MB       | 1236 MB      | 249 MB |
| Job 12      | 1690 MB       | 1448 MB      | 242 MB |
| Job 15      | 1844 MB       | 1589 MB      | 255 MB |
| Job 18      | 1984 MB       | 1741 MB      | 243 MB |

**Consistent ~240-250 MB freed per cache clear cycle.**

### Memory Growth Pattern

Memory trajectory over 20 jobs on a single worker:

```
Job 1:  1160 MB (initial load)
Job 3:   896 MB (after clear)
Job 6:  1101 MB (after clear)
Job 9:  1301 MB (after clear)
Job 12: 1448 MB (after clear)
Job 15: 1589 MB (after clear)
Job 18: 1741 MB (after clear)
Job 20: 2157 MB (before next clear)
```

### Remaining Issues

While the cache clearing prevents catastrophic memory growth, there is still a **slow baseline increase** of ~100-150 MB per clear cycle. This suggests additional memory accumulation in:

1. **Docling internal state** not covered by the cache clear
2. **Python object fragmentation** over time
3. **Third-party library caches** (PyTorch, EasyOCR, etc.)

### Recommendations for Further Improvement

1. **Reduce `jobs_between_cache_clear`** to 1 or 2 for tighter memory control (trade-off: slower due to model reloading)

2. **Worker recycling**: Configure Kubernetes to restart workers after N jobs:
   ```yaml
   # In deployment, use worker max-jobs flag if available
   # Or set memory limits and let K8s restart on OOM
   resources:
     limits:
       memory: "3Gi"
   ```

3. **Investigate docling upstream**: The baseline growth may be addressable in the docling library itself

4. **Consider process-per-job**: For extreme memory sensitivity, run each job in a subprocess that terminates after completion

## Log Format Reference

### Worker Start
```
[WORKER START] name=docling-worker | queues= | config={'jobs_between_cache_clear': 3, 'scratch_dir': '/data/scratch'}
```

### Job Execution
```
[JOB EXEC] job_id=xxx | task_id=xxx | queue=convert | trace_id=xxx
[JOB START] job_id=xxx | task_id=xxx | queue=convert | trace_id=xxx
[TASK START] task_id=xxx | type=convert | documents=1 | total_size=0 B | pipeline=standard | ocr=easyocr | formats=md,json
```

### Job Success
```
[DOC OK] doc_id | status=success | size=442.9 KB | pages=10
[TASK SUCCESS] task_id=xxx | time=54.55s | documents=1/1 succeeded | size=442.9 KB | pages=10 | throughput=0.2 pages/sec
[JOB SUCCESS] job_id=xxx | task_id=xxx | duration=54.57s
[MEMORY] job_id=xxx | rss=1115MB | jobs_total=1
```

### Cache Clear
```
[CACHE CLEAR] jobs_since_last_clear=3 | converters=1 | pipelines=1 | memory=859MB (freed 250MB)
```

## Memory Profiling for Debugging

To diagnose the remaining memory growth, a comprehensive memory profiling system is available.

### Enabling Memory Profiling

Set environment variables to enable detailed memory tracking:

```bash
# Enable memory profiling (captures detailed allocation info)
export DOCLING_SERVE_MEMORY_PROFILING=true

# Optional: Configure profiling depth
export DOCLING_SERVE_MEMORY_PROFILING_TOP_N=10      # Top allocations to track
export DOCLING_SERVE_MEMORY_PROFILING_TRACE_MALLOC=true  # Python tracemalloc
export DOCLING_SERVE_MEMORY_PROFILING_OBJGRAPH=true      # Object counting
```

### What Gets Tracked

When memory profiling is enabled:

1. **tracemalloc**: Tracks Python memory allocations with full stack traces
   - Shows exactly which lines of code are allocating memory
   - Computes diffs between snapshots to identify growing allocations

2. **objgraph**: Counts Python objects by type
   - Identifies which object types are accumulating
   - Can trace reference chains to find why objects aren't being freed

3. **OTEL Metrics**: Exports memory stats via OpenTelemetry
   - `docling_worker_memory_rss_bytes`: Process RSS memory
   - `docling_worker_memory_heap_bytes`: Python heap size
   - `docling_worker_memory_tracemalloc_bytes`: Tracked allocations
   - `docling_worker_object_count`: Object counts by type
   - `docling_worker_object_growth_total`: Cumulative object growth

### Log Output with Profiling Enabled

```
[MEMORY] job_id=xxx | rss=1115MB | jobs_total=1
[MEMORY DETAIL] heap=892.3MB
[MEMORY DETAIL] tracemalloc_tracked=456.7MB
[MEMORY DETAIL] Top allocations:
  234.5MB (12345 objects) at /path/to/docling/converter.py:123
  [GROWTH] +45.2MB (+1234 objects) at /path/to/easyocr/reader.py:456
[MEMORY DETAIL] Object growth since last snapshot:
  dict: +5432 (now 123456)
  numpy.ndarray: +234 (now 5678)
[MEMORY DETAIL] gc_counts=(123, 45, 6)
```

### Programmatic Access

The worker exposes methods for debugging:

```python
# Get comprehensive diagnostic report
diagnostics = worker.get_memory_diagnostics()
# Returns: {current_memory, gc, top_allocations, object_counts, memory_trend}

# Find potentially leaking objects with backref chains
leaks = worker.find_memory_leaks()
# Returns: [{type, count, growth, backref_chain}, ...]
```

### Installing Optional Dependencies

For full profiling capabilities:

```bash
pip install objgraph    # For object counting and backref analysis
pip install memray      # For external profiling (optional)
```

## Version History

| Version | Commit | Description |
|---------|--------|-------------|
| 1.1.7   | -      | Added comprehensive memory profiling with OTEL integration |
| 1.1.6   | 6b01ac8 | Fixed cache clearing to properly release models |
| 1.1.5   | 49b7cb7 | Added RQ worker memory management |
| 1.1.4   | b268ef4 | Fixed logging configuration |
| 1.1.3   | 1a6b304 | Enabled INFO logging in rq_worker |
