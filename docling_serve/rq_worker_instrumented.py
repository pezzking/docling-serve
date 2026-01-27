"""Instrumented RQ worker with OpenTelemetry tracing support."""

import gc
import logging
import os
import time
from pathlib import Path

from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from docling_jobkit.convert.manager import (
    DoclingConverterManagerConfig,
)
from docling_jobkit.orchestrators.rq.orchestrator import RQOrchestratorConfig
from docling_jobkit.orchestrators.rq.worker import CustomRQWorker

from docling_serve.memory_profiler import MemoryProfiler, get_memory_profiler
from docling_serve.rq_instrumentation import extract_trace_context
from docling_serve.worker_logging import (
    format_duration,
    log_job_finished,
    log_worker_startup,
)

logger = logging.getLogger(__name__)

# Default value; can be overridden via jobs_between_cache_clear parameter
DEFAULT_JOBS_BETWEEN_CACHE_CLEAR = 10


def _get_memory_mb() -> float:
    """Get current process memory usage in MB (RSS)."""
    try:
        # Works on Linux (reads from /proc)
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass

    # Fallback for macOS/other - use resource module
    try:
        import resource

        # ru_maxrss is in bytes on macOS, KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage / (1024 * 1024)  # bytes to MB
        return usage / 1024  # KB to MB
    except Exception:
        return 0.0


class InstrumentedRQWorker(CustomRQWorker):
    """RQ Worker with OpenTelemetry tracing instrumentation."""

    def __init__(
        self,
        *args,
        orchestrator_config: RQOrchestratorConfig,
        cm_config: DoclingConverterManagerConfig,
        scratch_dir: Path,
        jobs_between_cache_clear: int = DEFAULT_JOBS_BETWEEN_CACHE_CLEAR,
        memory_profiling: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            orchestrator_config=orchestrator_config,
            cm_config=cm_config,
            scratch_dir=scratch_dir,
            **kwargs,
        )
        self.tracer = trace.get_tracer(__name__)
        self._jobs_between_cache_clear = jobs_between_cache_clear
        self._jobs_since_cache_clear = 0
        self._total_jobs_processed = 0
        self._worker_start_time = time.time()

        # Initialize memory profiler
        self._memory_profiler = get_memory_profiler()
        if memory_profiling and not self._memory_profiler.enabled:
            # Override with explicit flag if passed
            self._memory_profiler = MemoryProfiler(
                enabled=True,
                top_n=int(os.environ.get("DOCLING_SERVE_MEMORY_PROFILING_TOP_N", "10")),
                use_tracemalloc=True,
                use_objgraph=True,
            )

        # Set up OTEL metrics for memory tracking
        try:
            meter = metrics.get_meter(__name__)
            self._memory_profiler.setup_otel_metrics(meter)
        except Exception as e:
            logger.debug(f"Could not set up OTEL metrics for memory profiler: {e}")

        # Log worker startup
        queue_names = [q.name for q in kwargs.get("queues", [])]
        log_worker_startup(
            worker_name=kwargs.get("name", "docling-worker"),
            queue_names=queue_names,
            config={
                "jobs_between_cache_clear": self._jobs_between_cache_clear,
                "scratch_dir": str(scratch_dir),
                "memory_profiling": self._memory_profiler.enabled,
            },
            log=logger,
        )

    def perform_job(self, job, queue):
        """
        Perform job with distributed tracing support.

        This extracts the trace context from the job metadata and creates
        a span that continues the trace from the API request.
        """
        job_start_time = time.time()

        # Extract parent trace context from job metadata
        parent_context = extract_trace_context(job)

        # Create span name from job function
        func_name = job.func_name if hasattr(job, "func_name") else "unknown"
        span_name = f"rq.job.{func_name}"

        # Extract task_id from job kwargs if available
        task_id = "unknown"
        if hasattr(job, "kwargs") and job.kwargs:
            task_data = job.kwargs.get("task_data", {})
            if isinstance(task_data, dict):
                task_id = task_data.get("task_id", "unknown")

        # Start span with parent context
        with self.tracer.start_as_current_span(
            span_name,
            context=parent_context,
            kind=SpanKind.CONSUMER,
        ) as span:
            try:
                # Add job attributes to span
                span.set_attribute("rq.job.id", job.id)
                span.set_attribute("rq.job.func_name", func_name)
                span.set_attribute("rq.queue.name", queue.name)

                if hasattr(job, "description") and job.description:
                    span.set_attribute("rq.job.description", job.description)

                if hasattr(job, "timeout") and job.timeout:
                    span.set_attribute("rq.job.timeout", job.timeout)

                # Add job kwargs info
                if hasattr(job, "kwargs") and job.kwargs:
                    # Add conversion manager before executing
                    job.kwargs["conversion_manager"] = self.conversion_manager
                    job.kwargs["orchestrator_config"] = self.orchestrator_config
                    job.kwargs["scratch_dir"] = self.scratch_dir

                    # Log task info if available
                    task_type = job.kwargs.get("task_type")
                    if task_type:
                        span.set_attribute("docling.task.type", str(task_type))

                    sources = job.kwargs.get("sources", [])
                    if sources:
                        span.set_attribute("docling.task.num_sources", len(sources))

                trace_id = f"{span.get_span_context().trace_id:032x}"
                logger.info(
                    f"[JOB EXEC] job_id={job.id} | task_id={task_id} | "
                    f"queue={queue.name} | trace_id={trace_id}"
                )

                # Execute the actual job
                result = super(CustomRQWorker, self).perform_job(job, queue)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                # Log job completion
                job_duration = time.time() - job_start_time
                log_job_finished(
                    job_id=job.id,
                    task_id=task_id,
                    success=True,
                    duration=job_duration,
                    log=logger,
                )

                # Update counters and periodic memory cleanup
                self._total_jobs_processed += 1
                self._jobs_since_cache_clear += 1

                # Take memory snapshot and log
                snapshot = self._memory_profiler.take_snapshot(
                    job_id=job.id,
                    jobs_total=self._total_jobs_processed,
                )
                self._memory_profiler.log_snapshot(snapshot)

                if self._jobs_since_cache_clear >= self._jobs_between_cache_clear:
                    self._clear_caches()

                return result

            except Exception as e:
                # Log job failure
                job_duration = time.time() - job_start_time
                log_job_finished(
                    job_id=job.id,
                    task_id=task_id,
                    success=False,
                    duration=job_duration,
                    log=logger,
                )
                logger.error(f"[JOB ERROR] job_id={job.id} | error={e}")

                # Record exception and mark span as failed
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Also clear caches on error to free partially loaded resources
                self._jobs_since_cache_clear += 1
                if self._jobs_since_cache_clear >= self._jobs_between_cache_clear:
                    self._clear_caches()

                raise

    def _clear_caches(self) -> None:
        """Clear pipeline caches to free memory.

        This method properly clears:
        1. The DoclingConverterManager's LRU cache of DocumentConverters
        2. Each cached converter's initialized_pipelines dict
        3. Model references within pipelines (build_pipe, enrichment_pipe)
        4. PyTorch GPU/MPS memory caches
        5. Python garbage collection
        """
        jobs_cleared = self._jobs_since_cache_clear
        self._jobs_since_cache_clear = 0
        mem_before = _get_memory_mb()

        pipelines_cleared = 0
        converters_cleared = 0

        # Step 1: Clear initialized pipelines from all cached converters
        # The _get_converter_from_hash is an LRU cache containing DocumentConverters
        if hasattr(self.conversion_manager, "_get_converter_from_hash"):
            cache_func = self.conversion_manager._get_converter_from_hash
            # Access the cache info to see what's cached
            if hasattr(cache_func, "cache_info"):
                cache_info = cache_func.cache_info()
                converters_cleared = cache_info.currsize

            # Get cached converters before clearing the LRU cache
            # We need to clear their pipelines first to release model references
            if hasattr(cache_func, "cache_clear"):
                # The cache stores converters by options hash
                # Access internal cache dict if available (Python 3.9+)
                try:
                    # Try to access cached items to clear their pipelines
                    if hasattr(cache_func, "__wrapped__"):
                        # For each converter in the options_map, clear its pipelines
                        options_map = getattr(
                            self.conversion_manager, "_options_map", {}
                        )
                        for options_hash in list(options_map.keys()):
                            try:
                                converter = cache_func(options_hash)
                                pipelines_cleared += self._clear_converter_pipelines(
                                    converter
                                )
                            except (KeyError, TypeError):
                                pass
                except Exception as e:
                    logger.debug(f"Could not access cached converters: {e}")

        # Step 2: Clear the LRU cache itself (removes converter references)
        if hasattr(self.conversion_manager, "clear_cache"):
            self.conversion_manager.clear_cache()

        # Step 3: Clear the options map to prevent stale references
        if hasattr(self.conversion_manager, "_options_map"):
            self.conversion_manager._options_map.clear()

        # Step 4: Release PyTorch memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Also try MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except ImportError:
            pass

        # Step 5: Force garbage collection (run multiple times for ref cycles)
        gc.collect(generation=0)
        gc.collect(generation=1)
        gc.collect(generation=2)

        mem_after = _get_memory_mb()

        # Log cache clear with detailed info
        logger.info(
            f"[CACHE CLEAR] jobs_since_last_clear={jobs_cleared} | "
            f"converters={converters_cleared} | pipelines={pipelines_cleared} | "
            f"memory={mem_after:.0f}MB (freed {mem_before - mem_after:.0f}MB)"
        )

    def _clear_converter_pipelines(self, converter) -> int:
        """Clear all initialized pipelines from a DocumentConverter.

        Returns the number of pipelines cleared.
        """
        cleared = 0

        # Clear the initialized_pipelines dict (note: no underscore prefix)
        if hasattr(converter, "initialized_pipelines"):
            pipelines = converter.initialized_pipelines
            for key, pipeline in list(pipelines.items()):
                # Clear model references in the pipeline
                self._clear_pipeline_models(pipeline)
                cleared += 1
            pipelines.clear()

        return cleared

    def _clear_pipeline_models(self, pipeline) -> None:
        """Clear model references from a pipeline to help garbage collection."""
        # Clear build_pipe models (OCR, layout, etc.)
        if hasattr(pipeline, "build_pipe") and pipeline.build_pipe:
            for i, model in enumerate(pipeline.build_pipe):
                # Try to delete heavy attributes from models
                self._clear_model_internals(model)
            pipeline.build_pipe.clear()

        # Clear enrichment_pipe models (code/formula, picture classifier, etc.)
        if hasattr(pipeline, "enrichment_pipe") and pipeline.enrichment_pipe:
            for model in pipeline.enrichment_pipe:
                self._clear_model_internals(model)
            pipeline.enrichment_pipe.clear()

    def _clear_model_internals(self, model) -> None:
        """Attempt to clear internal model weights and caches."""
        # Common attributes that hold heavy data
        heavy_attrs = [
            "model",
            "predictor",
            "layout_predictor",
            "table_predictor",
            "reader",  # EasyOCR reader
            "detector",
            "recognizer",
            "net",
            "network",
        ]

        for attr in heavy_attrs:
            if hasattr(model, attr):
                try:
                    # Delete the attribute to release the reference
                    delattr(model, attr)
                except (AttributeError, TypeError):
                    # Some attributes may be read-only or properties
                    try:
                        setattr(model, attr, None)
                    except (AttributeError, TypeError):
                        pass

    def worker_status(self) -> dict:
        """Return current worker status for logging/monitoring."""
        uptime = time.time() - self._worker_start_time
        return {
            "total_jobs_processed": self._total_jobs_processed,
            "jobs_since_cache_clear": self._jobs_since_cache_clear,
            "uptime": format_duration(uptime),
            "uptime_seconds": round(uptime, 2),
            "memory_rss_mb": _get_memory_mb(),
        }

    def get_memory_diagnostics(self) -> dict:
        """Get detailed memory diagnostics for debugging.

        Returns a comprehensive report including:
        - Current memory usage
        - Top allocations (if tracemalloc enabled)
        - Object counts and growth (if objgraph enabled)
        - Memory trends over time
        """
        return self._memory_profiler.get_diagnostic_report()

    def find_memory_leaks(self) -> list:
        """Find potentially leaking objects with backref chains.

        Useful for debugging memory growth by identifying objects
        that are accumulating references.
        """
        return self._memory_profiler.find_leaking_objects()
