"""Human-readable logging utilities for docling workers.

This module provides structured, human-readable logging for document processing,
including metrics like file size, processing time, pipeline route, and document details.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling_jobkit.datamodel.http_inputs import FileSource, HttpSource
from docling_jobkit.datamodel.task import Task

logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format duration into human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


@dataclass
class DocumentMetrics:
    """Metrics for a single document conversion."""

    filename: str
    source_type: str  # "file", "url", "stream"
    file_size: int | None = None
    num_pages: int | None = None
    status: str | None = None
    processing_time: float | None = None
    pipeline_used: str | None = None
    ocr_engine: str | None = None
    output_formats: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured logging."""
        d: dict[str, Any] = {
            "filename": self.filename,
            "source_type": self.source_type,
        }
        if self.file_size is not None:
            d["file_size"] = self.file_size
            d["file_size_human"] = format_size(self.file_size)
        if self.num_pages is not None:
            d["num_pages"] = self.num_pages
        if self.status is not None:
            d["status"] = self.status
        if self.processing_time is not None:
            d["processing_time_sec"] = round(self.processing_time, 3)
            d["processing_time_human"] = format_duration(self.processing_time)
        if self.pipeline_used:
            d["pipeline"] = self.pipeline_used
        if self.ocr_engine:
            d["ocr_engine"] = self.ocr_engine
        if self.output_formats:
            d["output_formats"] = self.output_formats
        if self.errors:
            d["errors"] = self.errors
        return d


@dataclass
class TaskMetrics:
    """Aggregated metrics for a task with multiple documents."""

    task_id: str
    task_type: str
    num_documents: int = 0
    total_size: int = 0
    total_pages: int = 0
    succeeded: int = 0
    failed: int = 0
    start_time: float | None = None
    end_time: float | None = None
    documents: list[DocumentMetrics] = field(default_factory=list)
    pipeline: str | None = None
    ocr_engine: str | None = None
    output_formats: list[str] = field(default_factory=list)

    @property
    def total_time(self) -> float | None:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def pages_per_second(self) -> float | None:
        total = self.total_time
        if total and total > 0 and self.total_pages > 0:
            return self.total_pages / total
        return None

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured logging."""
        d = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "num_documents": self.num_documents,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }
        if self.total_size > 0:
            d["total_size"] = self.total_size
            d["total_size_human"] = format_size(self.total_size)
        if self.total_pages > 0:
            d["total_pages"] = self.total_pages
        if self.total_time is not None:
            d["total_time_sec"] = round(self.total_time, 3)
            d["total_time_human"] = format_duration(self.total_time)
        if self.pages_per_second is not None:
            d["pages_per_second"] = round(self.pages_per_second, 2)
        if self.pipeline:
            d["pipeline"] = self.pipeline
        if self.ocr_engine:
            d["ocr_engine"] = self.ocr_engine
        if self.output_formats:
            d["output_formats"] = self.output_formats
        return d


class TaskLogger:
    """Context manager for logging task execution with metrics."""

    def __init__(self, task: Task, log: logging.Logger | None = None):
        self.task = task
        self.log = log or logger
        self.metrics = TaskMetrics(
            task_id=task.task_id,
            task_type=task.task_type.value,
            num_documents=len(task.sources),
        )
        self._extract_task_options()

    def _extract_task_options(self) -> None:
        """Extract pipeline/OCR options from task."""
        opts = self.task.convert_options
        if opts:
            # Get pipeline type
            if hasattr(opts, "pipeline") and opts.pipeline:
                self.metrics.pipeline = str(opts.pipeline.value)

            # Get OCR engine
            if hasattr(opts, "ocr_engine") and opts.ocr_engine:
                ocr = opts.ocr_engine
                self.metrics.ocr_engine = (
                    str(ocr.value) if hasattr(ocr, "value") else str(ocr)
                )

            # Get output formats
            if hasattr(opts, "to_formats") and opts.to_formats:
                self.metrics.output_formats = [f.value for f in opts.to_formats]

    def start(self) -> "TaskLogger":
        """Mark task start time and log initial info."""
        self.metrics.start_time = time.time()
        self._log_task_started()
        return self

    def _log_task_started(self) -> None:
        """Log task start with source information."""
        sources_info = []
        for source in self.task.sources:
            info = self._describe_source(source)
            sources_info.append(info)
            self.metrics.total_size += info.get("size", 0)

        pipeline_info = (
            f" | pipeline={self.metrics.pipeline}" if self.metrics.pipeline else ""
        )
        ocr_info = (
            f" | ocr={self.metrics.ocr_engine}" if self.metrics.ocr_engine else ""
        )
        formats_info = (
            f" | formats={','.join(self.metrics.output_formats)}"
            if self.metrics.output_formats
            else ""
        )

        self.log.info(
            f"[TASK START] task_id={self.metrics.task_id} | "
            f"type={self.metrics.task_type} | "
            f"documents={self.metrics.num_documents} | "
            f"total_size={format_size(self.metrics.total_size)}"
            f"{pipeline_info}{ocr_info}{formats_info}"
        )

        # Log individual documents at debug level
        for i, info in enumerate(sources_info, 1):
            self.log.debug(
                f"  [{i}/{len(sources_info)}] {info['type']}: {info['name']} "
                f"({format_size(info.get('size', 0))})"
            )

    def _describe_source(
        self, source: Union[DocumentStream, FileSource, HttpSource, Any]
    ) -> dict[str, Any]:
        """Extract source information for logging."""
        if isinstance(source, DocumentStream):
            size = len(source.stream.read()) if hasattr(source.stream, "read") else 0
            if hasattr(source.stream, "seek"):
                source.stream.seek(0)  # Reset stream position
            return {
                "type": "stream",
                "name": source.name or "unnamed",
                "size": size,
            }
        elif isinstance(source, FileSource):
            # Estimate size from base64 data if available
            size = 0
            if hasattr(source, "base64") and source.base64:
                size = len(source.base64) * 3 // 4
            return {
                "type": "file",
                "name": source.filename,
                "size": size,
            }
        elif isinstance(source, HttpSource):
            return {
                "type": "url",
                "name": str(source.url),
                "size": 0,  # Size unknown for URLs
            }
        else:
            return {"type": "unknown", "name": str(source), "size": 0}

    def _extract_document_metrics(self, result: ConversionResult) -> DocumentMetrics:
        """Extract metrics from a conversion result."""
        doc = result.document
        input_doc = result.input

        # Extract filename
        filename = self._get_filename(input_doc)

        # Determine source type
        source_type = "unknown"
        if hasattr(input_doc, "file"):
            source_type = "file" if input_doc.file else "stream"

        # Get file size
        file_size = None
        if hasattr(input_doc, "filesize") and input_doc.filesize:
            file_size = input_doc.filesize

        # Get page count
        num_pages = self._get_page_count(doc, result)

        # Get status
        status_str = self._get_status_str(result)

        # Get processing time and errors
        processing_time = self._get_processing_time(result)
        errors = self._collect_errors(result)

        return DocumentMetrics(
            filename=filename,
            source_type=source_type,
            file_size=file_size,
            num_pages=num_pages,
            status=status_str,
            processing_time=processing_time,
            pipeline_used=self.metrics.pipeline,
            ocr_engine=self.metrics.ocr_engine,
            output_formats=self.metrics.output_formats,
            errors=errors,
        )

    def _get_filename(self, input_doc: Any) -> str:
        """Extract filename from input document."""
        if hasattr(input_doc, "file") and input_doc.file:
            return Path(str(input_doc.file)).name
        if hasattr(input_doc, "document_hash"):
            return f"doc_{input_doc.document_hash[:8]}"
        return "unknown"

    def _get_page_count(self, doc: Any, result: ConversionResult) -> int | None:
        """Get page count from document or result."""
        if hasattr(doc, "pages") and doc.pages:
            return len(doc.pages)
        if hasattr(result, "pages") and result.pages:
            return len(result.pages)
        return None

    def _get_status_str(self, result: ConversionResult) -> str:
        """Get status string from result."""
        if hasattr(result, "status"):
            if hasattr(result.status, "value"):
                return result.status.value
            return str(result.status)
        return "unknown"

    def _get_processing_time(self, result: ConversionResult) -> float | None:
        """Sum processing time from timings."""
        if not hasattr(result, "timings") or not result.timings:
            return None
        total = 0.0
        for timing_item in result.timings.values():
            if hasattr(timing_item, "time") and timing_item.time:
                total += timing_item.time
        return total if total > 0 else None

    def _collect_errors(self, result: ConversionResult) -> list[str]:
        """Collect error messages from result."""
        errors = []
        if hasattr(result, "errors") and result.errors:
            for err in result.errors:
                if hasattr(err, "error_message"):
                    errors.append(err.error_message)
                else:
                    errors.append(str(err))
        return errors

    def log_document_result(self, result: ConversionResult) -> None:
        """Log individual document conversion result."""
        doc_metrics = self._extract_document_metrics(result)
        self.metrics.documents.append(doc_metrics)

        # Update aggregates
        if doc_metrics.file_size:
            self.metrics.total_size += doc_metrics.file_size
        if doc_metrics.num_pages:
            self.metrics.total_pages += doc_metrics.num_pages

        if result.status == ConversionStatus.SUCCESS:
            self.metrics.succeeded += 1
        else:
            self.metrics.failed += 1

        # Log the document result
        self._log_document_line(doc_metrics, result.status)

    def _log_document_line(
        self, metrics: DocumentMetrics, status: ConversionStatus
    ) -> None:
        """Output log line for a document result."""
        size_info = (
            f" | size={format_size(metrics.file_size)}" if metrics.file_size else ""
        )
        pages_info = f" | pages={metrics.num_pages}" if metrics.num_pages else ""
        time_info = (
            f" | time={format_duration(metrics.processing_time)}"
            if metrics.processing_time
            else ""
        )
        error_info = f" | errors={len(metrics.errors)}" if metrics.errors else ""

        status_emoji = "✓" if status == ConversionStatus.SUCCESS else "✗"
        log_level = (
            logging.INFO if status == ConversionStatus.SUCCESS else logging.WARNING
        )

        self.log.log(
            log_level,
            f"[DOC {status_emoji}] {metrics.filename} | status={metrics.status}"
            f"{size_info}{pages_info}{time_info}{error_info}",
        )

        for err in metrics.errors:
            self.log.warning(f"  Error: {err}")

    def finish(self, success: bool = True) -> None:
        """Mark task end and log summary."""
        self.metrics.end_time = time.time()
        self._log_task_summary(success)

    def _log_task_summary(self, success: bool) -> None:
        """Log task completion summary."""
        total_time = self.metrics.total_time or 0
        status = "SUCCESS" if success else "FAILED"

        pages_info = (
            f" | pages={self.metrics.total_pages}" if self.metrics.total_pages else ""
        )
        pps_info = ""
        if self.metrics.pages_per_second:
            pps_info = f" | throughput={self.metrics.pages_per_second:.1f} pages/sec"

        self.log.info(
            f"[TASK {status}] task_id={self.metrics.task_id} | "
            f"time={format_duration(total_time)} | "
            f"documents={self.metrics.succeeded}/{self.metrics.num_documents} succeeded | "
            f"size={format_size(self.metrics.total_size)}"
            f"{pages_info}{pps_info}"
        )

        # Log detailed metrics at debug level
        self.log.debug(f"Task metrics: {self.metrics.to_log_dict()}")

    def log_error(self, error: Exception) -> None:
        """Log task error."""
        self.metrics.end_time = time.time()
        total_time = self.metrics.total_time or 0

        self.log.error(
            f"[TASK ERROR] task_id={self.metrics.task_id} | "
            f"time={format_duration(total_time)} | "
            f"error={type(error).__name__}: {error}"
        )


def log_worker_startup(
    worker_name: str,
    queue_names: list[str],
    config: dict[str, Any],
    log: logging.Logger | None = None,
) -> None:
    """Log worker startup information."""
    log = log or logger
    log.info(
        f"[WORKER START] name={worker_name} | "
        f"queues={','.join(queue_names)} | "
        f"config={config}"
    )


def log_worker_shutdown(
    worker_name: str,
    jobs_processed: int,
    uptime: float,
    log: logging.Logger | None = None,
) -> None:
    """Log worker shutdown information."""
    log = log or logger
    log.info(
        f"[WORKER STOP] name={worker_name} | "
        f"jobs_processed={jobs_processed} | "
        f"uptime={format_duration(uptime)}"
    )


def log_job_started(
    job_id: str,
    task_id: str,
    queue_name: str,
    trace_id: str | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Log job execution start."""
    log = log or logger
    trace_info = f" | trace_id={trace_id}" if trace_id else ""
    log.info(
        f"[JOB START] job_id={job_id} | task_id={task_id} | queue={queue_name}{trace_info}"
    )


def log_job_finished(
    job_id: str,
    task_id: str,
    success: bool,
    duration: float,
    log: logging.Logger | None = None,
) -> None:
    """Log job execution completion."""
    log = log or logger
    status = "SUCCESS" if success else "FAILED"
    log.info(
        f"[JOB {status}] job_id={job_id} | task_id={task_id} | "
        f"duration={format_duration(duration)}"
    )


def log_cache_clear(
    jobs_since_clear: int,
    memory_freed: int | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Log cache clearing event."""
    log = log or logger
    mem_info = f" | memory_freed={format_size(memory_freed)}" if memory_freed else ""
    log.info(f"[CACHE CLEAR] jobs_since_last_clear={jobs_since_clear}{mem_info}")
