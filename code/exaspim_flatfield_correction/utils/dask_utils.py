import logging
import os
from enum import Enum
from typing import Any, Mapping

import dask
from distributed import Client, LocalCluster


class WorkerMode(Enum):
    """Execution style for LocalCluster workers."""

    PROCESSES = "processes"
    THREADS = "threads"


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "create_dask_client",
    "set_dask_config",
    "WorkerMode",
]


def create_dask_client(
    results_dir: str,
    num_workers: int,
    worker_mode: WorkerMode | str = WorkerMode.PROCESSES,
    config_dict: Mapping[str, Any] | None = None,
) -> Client:
    """
    Configure and return a Dask distributed client.

    Parameters
    ----------
    results_dir : str
        Directory where pipeline outputs are written; used to stash Dask temp files.
    num_workers : int
        Number of logical workers requested from the CLI.
    worker_mode : WorkerMode or str, optional
        Execution style for Dask: process-per-worker or a multithreaded worker.
    config_dict : Mapping[str, Any] or None, optional
        Additional Dask configuration options layered over the defaults.

    Returns
    -------
    distributed.Client
        Connected client bound to the configured LocalCluster.

    Raises
    ------
    ValueError
        If an invalid worker mode is supplied or ``CO_MEMORY`` is malformed.
    """
    # When running from CodeOcean, ensure Dask temp dir lives under /results which sits on fast EBS storage.
    base_config = {
        "temporary-directory": os.path.join(results_dir, "dask-temp")
    }
    merged_overrides = dict(base_config)
    if config_dict:
        merged_overrides.update(config_dict)
    set_dask_config(merged_overrides)

    # normalize worker_mode from CLI strings
    if isinstance(worker_mode, str):
        worker_mode = WorkerMode(worker_mode)

    worker_processes, threads_per_worker = _resolve_threads_per_worker(
        worker_mode, num_workers
    )

    memory_limit = "auto"
    if worker_mode == WorkerMode.PROCESSES:
        # Dask only respects memory_limit when processes=True (distributed docs).
        memory_limit = _resolve_memory_limit(worker_processes)

    client = Client(
        LocalCluster(
            processes=worker_mode == WorkerMode.PROCESSES,
            n_workers=worker_processes,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        )
    )
    _LOGGER.info(
        "Initialized Dask client with %s mode, %d workers, memory_limit=%s",
        worker_mode.value,
        num_workers,
        memory_limit,
    )
    return client


def set_dask_config(config_dict: Mapping[str, Any] | None = None) -> None:
    """
    Configure global Dask settings for the pipeline.

    Default tuning parameters are applied first; provided overrides replace them.

    Parameters
    ----------
    config_dict : Mapping[str, Any] or None, optional
        Overrides layered on top of the pipeline defaults. 
        Note that only top-level keys are overriden, meaning 
        nested dicts are replaced, not merged.
    """
    defaults = {
        "distributed.worker.memory.target": 0.7,
        "distributed.worker.memory.spill": 0.8,
        "distributed.worker.memory.pause": 0.9,
        "distributed.worker.memory.terminate": 0.95,
        "distributed.scheduler.allowed-failures": 10,
        "logging": {
            "distributed.shuffle._scheduler_plugin": "error",
        },
    }
    cfg = dict(defaults)
    if config_dict:
        # Shallow update is sufficient for top-level keys used here
        cfg.update(config_dict)
    dask.config.set(cfg)


def _resolve_memory_limit(num_workers: int) -> int | str:
    """
    Determine per-worker memory limit driven by the ``CO_MEMORY`` environment variable.

    Parameters
    ----------
    num_workers : int
        Number of Dask worker processes.

    Returns
    -------
    int or str
        Explicit byte limit if ``CO_MEMORY`` is set, otherwise ``\"auto\"``.

    Raises
    ------
    ValueError
        If ``CO_MEMORY`` is present but not a positive integer.
    """
    raw = os.getenv("CO_MEMORY")
    if not raw:
        return "auto"
    try:
        co_memory = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"CO_MEMORY must be an integer number of bytes; got {raw!r}"
        ) from exc
    _LOGGER.info("CO_MEMORY: %s", co_memory)
    if co_memory <= 0:
        raise ValueError(
            "CO_MEMORY must be set to a positive integer value "
            "to allocate memory for Dask workers."
        )
    return int(co_memory / max(1, num_workers))


def _resolve_threads_per_worker(
    worker_mode: WorkerMode,
    num_workers: int,
) -> tuple[int, int]:
    """
    Split the CLI worker request into processes vs threads.

    Parameters
    ----------
    worker_mode : WorkerMode
        Selected worker execution mode.
    num_workers : int
        Requested worker count from the CLI.

    Returns
    -------
    tuple of int
        ``(process_count, threads_per_worker)`` suited for ``LocalCluster``.

    Raises
    ------
    ValueError
        If ``worker_mode`` holds an unsupported value.
    """
    if worker_mode == WorkerMode.PROCESSES:
        cpu_total = int(os.getenv("CO_CPUS", os.cpu_count() or 1))
        return num_workers, max(1, cpu_total // max(1, num_workers))
    if worker_mode == WorkerMode.THREADS:
        return 1, num_workers
    raise ValueError(f"Unknown WorkerMode: {worker_mode}")
