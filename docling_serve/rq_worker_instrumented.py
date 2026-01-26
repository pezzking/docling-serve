"""Instrumented RQ worker with OpenTelemetry tracing support."""

import gc
import logging
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from docling_jobkit.convert.manager import (
    DoclingConverterManagerConfig,
)
from docling_jobkit.orchestrators.rq.orchestrator import RQOrchestratorConfig
from docling_jobkit.orchestrators.rq.worker import CustomRQWorker

from docling_serve.rq_instrumentation import extract_trace_context
from docling_serve.worker_logging import (
    format_duration,
    log_cache_clear,
    log_job_finished,
    log_worker_startup,
)

logger = logging.getLogger(__name__)

# Clear pipeline cache every N jobs to prevent memory accumulation
JOBS_BETWEEN_CACHE_CLEAR = 10


class InstrumentedRQWorker(CustomRQWorker):
    """RQ Worker with OpenTelemetry tracing instrumentation."""

    def __init__(
        self,
        *args,
        orchestrator_config: RQOrchestratorConfig,
        cm_config: DoclingConverterManagerConfig,
        scratch_dir: Path,
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
        self._jobs_since_cache_clear = 0
        self._total_jobs_processed = 0
        self._worker_start_time = time.time()

        # Log worker startup
        queue_names = [q.name for q in kwargs.get("queues", [])]
        log_worker_startup(
            worker_name=kwargs.get("name", "docling-worker"),
            queue_names=queue_names,
            config={
                "jobs_between_cache_clear": JOBS_BETWEEN_CACHE_CLEAR,
                "scratch_dir": str(scratch_dir),
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
                if self._jobs_since_cache_clear >= JOBS_BETWEEN_CACHE_CLEAR:
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
                if self._jobs_since_cache_clear >= JOBS_BETWEEN_CACHE_CLEAR:
                    self._clear_caches()

                raise

    def _clear_caches(self) -> None:
        """Clear pipeline caches to free memory."""
        jobs_cleared = self._jobs_since_cache_clear
        self._jobs_since_cache_clear = 0

        # Clear the docling document converter pipeline cache
        if hasattr(self.conversion_manager, "converter"):
            converter = self.conversion_manager.converter
            if hasattr(converter, "clear_pipeline_cache"):
                converter.clear_pipeline_cache()

        # Force garbage collection
        gc.collect()

        # Log cache clear with structured logging
        log_cache_clear(jobs_since_clear=jobs_cleared, log=logger)

    def worker_status(self) -> dict:
        """Return current worker status for logging/monitoring."""
        uptime = time.time() - self._worker_start_time
        return {
            "total_jobs_processed": self._total_jobs_processed,
            "jobs_since_cache_clear": self._jobs_since_cache_clear,
            "uptime": format_duration(uptime),
            "uptime_seconds": round(uptime, 2),
        }
