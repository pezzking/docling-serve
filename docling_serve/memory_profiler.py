"""Memory profiling utilities for tracking and diagnosing memory leaks.

This module provides tools to track memory usage, identify growing objects,
and export memory metrics via OpenTelemetry. It can be enabled via environment
variables for debugging memory issues in production.

Environment Variables:
    DOCLING_SERVE_MEMORY_PROFILING: Enable memory profiling (default: false)
    DOCLING_SERVE_MEMORY_PROFILING_TOP_N: Number of top allocations to track (default: 10)
    DOCLING_SERVE_MEMORY_PROFILING_TRACE_MALLOC: Enable tracemalloc (default: true)
    DOCLING_SERVE_MEMORY_PROFILING_OBJGRAPH: Enable objgraph object tracking (default: true)
"""

import gc
import linecache
import logging
import os
import sys
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A snapshot of memory state at a point in time."""

    timestamp: float
    rss_mb: float
    heap_mb: float
    tracemalloc_mb: float
    top_allocations: list[dict[str, Any]] = field(default_factory=list)
    object_counts: dict[str, int] = field(default_factory=dict)
    object_growth: dict[str, int] = field(default_factory=dict)
    gc_counts: tuple[int, int, int] = (0, 0, 0)
    job_id: str | None = None
    jobs_total: int = 0


class MemoryProfiler:
    """Memory profiler with tracemalloc and objgraph integration.

    Tracks memory allocations, object counts, and growth between snapshots.
    Integrates with OpenTelemetry for metrics export.
    """

    def __init__(
        self,
        enabled: bool = False,
        top_n: int = 10,
        use_tracemalloc: bool = True,
        use_objgraph: bool = True,
    ):
        """Initialize the memory profiler.

        Args:
            enabled: Whether profiling is enabled
            top_n: Number of top allocations/objects to track
            use_tracemalloc: Enable Python tracemalloc for allocation tracking
            use_objgraph: Enable objgraph for object counting
        """
        self.enabled = enabled
        self.top_n = top_n
        self.use_tracemalloc = use_tracemalloc
        self.use_objgraph = use_objgraph

        self._tracemalloc_started = False
        self._previous_snapshot: tracemalloc.Snapshot | None = None
        self._previous_object_counts: dict[str, int] = {}
        self._baseline_object_counts: dict[str, int] = {}
        self._snapshots: list[MemorySnapshot] = []

        # OTEL meter (set up later if available)
        self._meter = None
        self._memory_rss_gauge = None
        self._memory_heap_gauge = None
        self._memory_tracemalloc_gauge = None
        self._object_count_gauge = None
        self._object_growth_counter = None

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize profiling tools."""
        if self.use_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)  # 25 frames of traceback
            self._tracemalloc_started = True
            logger.info("[MEMORY PROFILER] tracemalloc started with 25 frame depth")

        if self.use_objgraph:
            try:
                import objgraph  # type: ignore[import-untyped]  # noqa: F401

                logger.info("[MEMORY PROFILER] objgraph available for object tracking")
            except ImportError:
                logger.warning(
                    "[MEMORY PROFILER] objgraph not installed, "
                    "object tracking disabled. Install with: pip install objgraph"
                )
                self.use_objgraph = False

    def setup_otel_metrics(self, meter) -> None:
        """Set up OpenTelemetry metrics for memory tracking.

        Args:
            meter: OpenTelemetry Meter instance
        """
        self._meter = meter

        # Memory gauges
        self._memory_rss_gauge = meter.create_gauge(
            name="docling_worker_memory_rss_bytes",
            description="Worker RSS memory usage in bytes",
            unit="By",
        )

        self._memory_heap_gauge = meter.create_gauge(
            name="docling_worker_memory_heap_bytes",
            description="Worker heap memory usage in bytes",
            unit="By",
        )

        self._memory_tracemalloc_gauge = meter.create_gauge(
            name="docling_worker_memory_tracemalloc_bytes",
            description="Memory tracked by tracemalloc in bytes",
            unit="By",
        )

        # Object tracking
        self._object_count_gauge = meter.create_gauge(
            name="docling_worker_object_count",
            description="Count of Python objects by type",
            unit="1",
        )

        self._object_growth_counter = meter.create_counter(
            name="docling_worker_object_growth_total",
            description="Cumulative growth of Python objects by type",
            unit="1",
        )

        logger.info("[MEMORY PROFILER] OTEL metrics configured")

    def get_rss_mb(self) -> float:
        """Get current process RSS memory in MB."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except FileNotFoundError:
            pass

        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return usage / (1024 * 1024)
            return usage / 1024
        except Exception:
            return 0.0

    def get_heap_mb(self) -> float:
        """Get approximate heap size by summing object sizes."""
        try:
            # This is expensive, only use when profiling is enabled
            if not self.enabled:
                return 0.0

            total = 0
            for obj in gc.get_objects():
                try:
                    total += sys.getsizeof(obj)
                except (TypeError, ReferenceError):
                    pass
            return total / (1024 * 1024)
        except Exception:
            return 0.0

    def get_tracemalloc_stats(self) -> tuple[float, list[dict[str, Any]]]:
        """Get tracemalloc memory stats and top allocations.

        Returns:
            Tuple of (total_mb, top_allocations)
        """
        if not self.use_tracemalloc or not tracemalloc.is_tracing():
            return 0.0, []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        total_mb = sum(stat.size for stat in top_stats) / (1024 * 1024)

        top_allocations = []
        for stat in top_stats[: self.top_n]:
            frame = stat.traceback[0] if stat.traceback else None
            if frame:
                # Get the actual source line
                line = linecache.getline(frame.filename, frame.lineno).strip()
                top_allocations.append(
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                        "source": line[:100] if line else "",
                    }
                )

        # Compute diff from previous snapshot
        if self._previous_snapshot:
            diff_stats = snapshot.compare_to(self._previous_snapshot, "lineno")
            growing = [s for s in diff_stats if s.size_diff > 0]
            growing.sort(key=lambda s: s.size_diff, reverse=True)

            for diff_stat in growing[: self.top_n]:
                frame = diff_stat.traceback[0] if diff_stat.traceback else None
                if frame:
                    top_allocations.append(
                        {
                            "file": frame.filename,
                            "line": frame.lineno,
                            "size_diff_mb": diff_stat.size_diff / (1024 * 1024),
                            "count_diff": diff_stat.count_diff,
                            "type": "growth",
                        }
                    )

        self._previous_snapshot = snapshot
        return total_mb, top_allocations

    def get_object_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        """Get object counts and growth by type.

        Returns:
            Tuple of (current_counts, growth_since_last)
        """
        if not self.use_objgraph:
            return {}, {}

        try:
            import objgraph  # type: ignore[import-untyped]

            # Get most common types
            type_counts = objgraph.typestats()

            # Filter to interesting types (skip small counts)
            filtered = {k: v for k, v in type_counts.items() if v > 100}

            # Sort by count
            sorted_counts = dict(
                sorted(filtered.items(), key=lambda x: x[1], reverse=True)[: self.top_n]
            )

            # Calculate growth
            growth = {}
            for type_name, count in sorted_counts.items():
                prev_count = self._previous_object_counts.get(type_name, 0)
                if count > prev_count:
                    growth[type_name] = count - prev_count

            # Also track growth from baseline
            baseline_growth = {}
            for type_name, count in sorted_counts.items():
                baseline = self._baseline_object_counts.get(type_name, count)
                if count > baseline:
                    baseline_growth[type_name] = count - baseline

            self._previous_object_counts = sorted_counts.copy()

            # Set baseline if not set
            if not self._baseline_object_counts:
                self._baseline_object_counts = sorted_counts.copy()

            return sorted_counts, growth

        except Exception as e:
            logger.debug(f"Error getting object counts: {e}")
            return {}, {}

    def take_snapshot(
        self,
        job_id: str | None = None,
        jobs_total: int = 0,
    ) -> MemorySnapshot:
        """Take a complete memory snapshot.

        Args:
            job_id: Optional job ID for context
            jobs_total: Total jobs processed

        Returns:
            MemorySnapshot with current memory state
        """
        import time

        rss_mb = self.get_rss_mb()

        # Only compute expensive metrics if profiling is enabled
        if self.enabled:
            heap_mb = self.get_heap_mb()
            tracemalloc_mb, top_allocations = self.get_tracemalloc_stats()
            object_counts, object_growth = self.get_object_counts()
        else:
            heap_mb = 0.0
            tracemalloc_mb = 0.0
            top_allocations = []
            object_counts = {}
            object_growth = {}

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            heap_mb=heap_mb,
            tracemalloc_mb=tracemalloc_mb,
            top_allocations=top_allocations,
            object_counts=object_counts,
            object_growth=object_growth,
            gc_counts=gc.get_count(),
            job_id=job_id,
            jobs_total=jobs_total,
        )

        self._snapshots.append(snapshot)

        # Keep last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        # Update OTEL metrics
        self._update_otel_metrics(snapshot)

        return snapshot

    def _update_otel_metrics(self, snapshot: MemorySnapshot) -> None:
        """Update OpenTelemetry metrics from snapshot."""
        if not self._meter:
            return

        attributes = {"worker": os.environ.get("HOSTNAME", "unknown")}

        if self._memory_rss_gauge:
            self._memory_rss_gauge.set(
                int(snapshot.rss_mb * 1024 * 1024), attributes=attributes
            )

        if self._memory_heap_gauge and snapshot.heap_mb > 0:
            self._memory_heap_gauge.set(
                int(snapshot.heap_mb * 1024 * 1024), attributes=attributes
            )

        if self._memory_tracemalloc_gauge and snapshot.tracemalloc_mb > 0:
            self._memory_tracemalloc_gauge.set(
                int(snapshot.tracemalloc_mb * 1024 * 1024), attributes=attributes
            )

        if self._object_count_gauge:
            for type_name, count in snapshot.object_counts.items():
                self._object_count_gauge.set(
                    count, attributes={**attributes, "type": type_name}
                )

        if self._object_growth_counter:
            for type_name, growth in snapshot.object_growth.items():
                if growth > 0:
                    self._object_growth_counter.add(
                        growth, attributes={**attributes, "type": type_name}
                    )

    def log_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Log a memory snapshot with detailed breakdown."""
        logger.info(
            f"[MEMORY] job_id={snapshot.job_id} | rss={snapshot.rss_mb:.0f}MB | "
            f"jobs_total={snapshot.jobs_total}"
        )

        if not self.enabled:
            return

        # Log detailed info when profiling is enabled
        if snapshot.heap_mb > 0:
            logger.info(f"[MEMORY DETAIL] heap={snapshot.heap_mb:.1f}MB")

        if snapshot.tracemalloc_mb > 0:
            logger.info(
                f"[MEMORY DETAIL] tracemalloc_tracked={snapshot.tracemalloc_mb:.1f}MB"
            )

        # Log top allocations
        if snapshot.top_allocations:
            logger.info("[MEMORY DETAIL] Top allocations:")
            for alloc in snapshot.top_allocations[:5]:
                if "size_diff_mb" in alloc:
                    logger.info(
                        f"  [GROWTH] +{alloc['size_diff_mb']:.2f}MB "
                        f"({alloc['count_diff']:+d} objects) "
                        f"at {alloc['file']}:{alloc['line']}"
                    )
                else:
                    logger.info(
                        f"  {alloc['size_mb']:.2f}MB ({alloc['count']} objects) "
                        f"at {alloc['file']}:{alloc['line']}"
                    )

        # Log object growth
        if snapshot.object_growth:
            logger.info("[MEMORY DETAIL] Object growth since last snapshot:")
            for type_name, growth in list(snapshot.object_growth.items())[:5]:
                current = snapshot.object_counts.get(type_name, 0)
                logger.info(f"  {type_name}: +{growth} (now {current})")

        # Log GC state
        logger.info(f"[MEMORY DETAIL] gc_counts={snapshot.gc_counts}")

    def get_diagnostic_report(self) -> dict[str, Any]:
        """Generate a diagnostic report for debugging.

        Returns:
            Dictionary with memory diagnostic information
        """
        current = self.take_snapshot()

        object_growth_from_baseline: dict[str, int] = {}

        # Calculate total growth from baseline
        for type_name, count in current.object_counts.items():
            baseline = self._baseline_object_counts.get(type_name, count)
            if count > baseline:
                object_growth_from_baseline[type_name] = count - baseline

        report: dict[str, Any] = {
            "current_memory": {
                "rss_mb": current.rss_mb,
                "heap_mb": current.heap_mb,
                "tracemalloc_mb": current.tracemalloc_mb,
            },
            "gc": {
                "counts": current.gc_counts,
                "thresholds": gc.get_threshold(),
            },
            "top_allocations": current.top_allocations,
            "object_counts": current.object_counts,
            "object_growth_from_baseline": object_growth_from_baseline,
            "snapshots_count": len(self._snapshots),
        }

        # Add memory trend if we have multiple snapshots
        if len(self._snapshots) >= 2:
            first = self._snapshots[0]
            report["memory_trend"] = {
                "rss_growth_mb": current.rss_mb - first.rss_mb,
                "jobs_processed": current.jobs_total - first.jobs_total,
                "mb_per_job": (
                    (current.rss_mb - first.rss_mb)
                    / max(1, current.jobs_total - first.jobs_total)
                ),
            }

        return report

    def find_leaking_objects(self) -> list[dict[str, Any]]:
        """Find objects that might be leaking (growing references).

        Returns:
            List of potentially leaking objects with backref chains
        """
        if not self.use_objgraph:
            return []

        try:
            import objgraph  # type: ignore[import-untyped]

            results = []

            # Find objects with growing counts
            growth = objgraph.growth(limit=self.top_n)
            if growth:
                for type_name, count, delta in growth:
                    # Get backrefs for a sample object
                    try:
                        objs = objgraph.by_type(type_name)
                        if objs:
                            sample = objs[0]
                            chain = objgraph.find_backref_chain(
                                sample, objgraph.is_proper_module, max_depth=10
                            )
                            results.append(
                                {
                                    "type": type_name,
                                    "count": count,
                                    "growth": delta,
                                    "backref_chain": [str(type(o)) for o in chain[:5]],
                                }
                            )
                    except Exception:
                        results.append(
                            {
                                "type": type_name,
                                "count": count,
                                "growth": delta,
                                "backref_chain": [],
                            }
                        )

            return results

        except Exception as e:
            logger.debug(f"Error finding leaking objects: {e}")
            return []

    def reset_baseline(self) -> None:
        """Reset the baseline for object counting."""
        self._baseline_object_counts = {}
        self._previous_object_counts = {}
        self._previous_snapshot = None
        self._snapshots = []
        logger.info("[MEMORY PROFILER] Baseline reset")

    def stop(self) -> None:
        """Stop profiling and clean up."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            logger.info("[MEMORY PROFILER] tracemalloc stopped")


# Global profiler instance
_profiler: MemoryProfiler | None = None


def get_memory_profiler() -> MemoryProfiler:
    """Get or create the global memory profiler instance."""
    global _profiler

    if _profiler is None:
        enabled = os.environ.get("DOCLING_SERVE_MEMORY_PROFILING", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        top_n = int(os.environ.get("DOCLING_SERVE_MEMORY_PROFILING_TOP_N", "10"))
        use_tracemalloc = os.environ.get(
            "DOCLING_SERVE_MEMORY_PROFILING_TRACE_MALLOC", "true"
        ).lower() in ("true", "1", "yes")
        use_objgraph = os.environ.get(
            "DOCLING_SERVE_MEMORY_PROFILING_OBJGRAPH", "true"
        ).lower() in ("true", "1", "yes")

        _profiler = MemoryProfiler(
            enabled=enabled,
            top_n=top_n,
            use_tracemalloc=use_tracemalloc,
            use_objgraph=use_objgraph,
        )

        if enabled:
            logger.info(
                f"[MEMORY PROFILER] Initialized: tracemalloc={use_tracemalloc}, "
                f"objgraph={use_objgraph}, top_n={top_n}"
            )

    return _profiler
