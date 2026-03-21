"""Utility wrappers and functions

Description:
    This module provides utility functions and decorators for the AutoMartini workflow.
    It includes decorators for timing and memory profiling functions, a context manager
    for changing the working directory, and helper functions for cleaning directories and
    detecting CUDA availability.

Requirements:
    - Python 3.x

Author: DY
Date: YYYY-MM-DD
"""

import logging
import os
import subprocess as sp
import shutil
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

def get_logger(name="AutoMartini"):
    """Get the configured logger instance.
    
    Since logging is now configured in AutoMartini.__init__.py, this function
    simply returns the already-configured logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name (default: "AutoMartini")
        
    Returns
    -------
    logging.Logger
        The configured logger instance
    """
    return logging.getLogger(name)


# Backward compatibility - provide logger at module level
# For new code, prefer: from AutoMartini.utils import get_logger; logger = get_logger()
logger = get_logger()


def timeit(*args, **kwargs):
    """Backwards-compatible timeit decorator"""
    # If called with no args, it's being used as @timeit
    if len(args) == 0:
        return _timeit(**kwargs)
    # If first arg is a function, it's being used as @timeit
    if len(args) == 1 and callable(args[0]):
        return _timeit()(args[0])
    # If first arg is a level, it's being used as @timeit(level=...)
    if len(args) == 1 and isinstance(args[0], int):
        return _timeit(level=args[0])
    # New style with explicit parameters
    return _timeit(*args, **kwargs)


def _timeit(level=logging.DEBUG, unit='s'):
    """Decorator to measure and log execution time of a function, with adjustable log level and time unit.
    
    Args:
        level (int): Logging level (default: logging.DEBUG)
        unit (str): Time unit to display. Options:
            - 'ms': milliseconds
            - 's': seconds (default)
            - 'm': minutes
            - 'auto': automatically choose best unit
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            # Convert to requested unit
            if unit == 'ms' or (unit == 'auto' and execution_time < 1):
                display_time = execution_time * 1000
                unit_str = 'milliseconds'
            elif unit == 'm' or (unit == 'auto' and execution_time > 60):
                display_time = execution_time / 60
                unit_str = 'minutes'
            else:  # seconds is default
                display_time = execution_time
                unit_str = 'seconds'
            logger.log(
                level,
                "Function '%s.%s' executed in %.6f %s",
                func.__module__,
                func.__name__,
                display_time,
                unit_str,
            )
            return result
        return wrapper
    return decorator


def memprofit(*args, **kwargs):
    """Backwards-compatible memory profiling decorator"""
    # If called with no args, it's being used as @memprofit
    if len(args) == 0:
        return _memprofit(**kwargs)
    
    # If first arg is a function, it's being used as @memprofit
    if len(args) == 1 and callable(args[0]):
        return _memprofit()(args[0])
        
    # If first arg is a level, it's being used as @memprofit(level=...)
    if len(args) == 1 and isinstance(args[0], int):
        return _memprofit(level=args[0])
    
    # New style with explicit parameters
    return _memprofit(*args, **kwargs)


def _memprofit(level=logging.DEBUG):
    """Decorator to profile and log the memory usage of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()  # Start memory tracking
            result = func(*args, **kwargs)  # Execute the function
            current, peak = tracemalloc.get_traced_memory()  # Get memory usage
            logger.log(
                level,
                "Memory usage after executing '%s.%s': %.2f MB, Peak: %.2f MB",
                func.__module__,
                func.__name__,
                current / 1024**2,
                peak / 1024**2,
            )
            tracemalloc.stop()  # Stop memory tracking
            return result
        return wrapper
    return decorator


@contextmanager
def cd(newdir):
    """
    Context manager to temporarily change the current working directory.

    Parameters:
        newdir (str or Path): The target directory to change into.

    Yields:
        None. After the context, reverts to the original directory.
    """
    prevdir = Path.cwd()
    os.chdir(newdir)
    logger.info("Changed working directory to: %s", newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


def clean_dir(directory=".", pattern="#*"):
    """
    Remove files matching a specific pattern from a directory.

    Parameters:
        directory (str or Path, optional): Directory to search (default: current directory).
        pattern (str, optional): Glob pattern for files to remove (default: "#*").
    """
    directory = Path(directory)
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def _kwargs_to_str(hyphen="-", **kwargs):
    return " ".join([f"{hyphen}{key} {value}" for key, value in kwargs.items()])


def gmx(command, gmx_callable="gmx_mpi", **kwargs):
    """Execute a GROMACS command."""
    clinput = kwargs.pop("clinput", None)
    cltext = kwargs.pop("cltext", True)

    if shutil.which(gmx_callable) is None and gmx_callable == "gmx_mpi":
        if shutil.which("gmx") is not None:
            gmx_callable = "gmx"
            logger.info("gmx_mpi not found; falling back to gmx")

    cmd = gmx_callable + " " + command + " " + _kwargs_to_str(**kwargs)
    try:
        sp.run(cmd.split(), input=clinput, text=cltext, check=True)
    except sp.CalledProcessError as e:
        logger.error("GROMACS command failed with exit code %s", e.returncode)
        raise


def get_ntomp():
    """Detect number of available CPU cores for OpenMP threads."""
    env_ntomp = os.environ.get("OMP_NUM_THREADS")
    if env_ntomp:
        try:
            ntomp_val = int(env_ntomp)
            logger.info("Using OMP_NUM_THREADS=%s from environment", ntomp_val)
            return ntomp_val
        except ValueError:
            logger.warning("Invalid OMP_NUM_THREADS value: %s", env_ntomp)

    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_ntasks and slurm_cpus_per_task:
        try:
            ntasks = int(slurm_ntasks)
            cpus_per_task = int(slurm_cpus_per_task)
            total_cores = ntasks * cpus_per_task
            logger.info(
                "SLURM allocation: %s tasks × %s cpus/task = %s total cores",
                ntasks,
                cpus_per_task,
                total_cores,
            )
            return total_cores
        except ValueError:
            logger.warning(
                "Invalid SLURM values - ntasks: %s, cpus_per_task: %s",
                slurm_ntasks,
                slurm_cpus_per_task,
            )

    if slurm_cpus_per_task:
        try:
            ntomp_val = int(slurm_cpus_per_task)
            logger.info("Using SLURM_CPUS_PER_TASK=%s (no ntasks found)", ntomp_val)
            return ntomp_val
        except ValueError:
            logger.warning("Invalid SLURM_CPUS_PER_TASK value: %s", slurm_cpus_per_task)

    slurm_nprocs = os.environ.get("SLURM_NPROCS")
    if slurm_nprocs:
        try:
            ntomp_val = int(slurm_nprocs)
            logger.info("Using SLURM_NPROCS=%s from SLURM allocation", ntomp_val)
            return ntomp_val
        except ValueError:
            logger.warning("Invalid SLURM_NPROCS value: %s", slurm_nprocs)

    pbs_ncpus = os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS")
    if pbs_ncpus:
        try:
            ntomp_val = int(pbs_ncpus)
            logger.info("Using PBS_NCPUS/NCPUS=%s from PBS allocation", ntomp_val)
            return ntomp_val
        except ValueError:
            logger.warning("Invalid PBS_NCPUS/NCPUS value: %s", pbs_ncpus)

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            ntomp_val = max(1, quota // period)
            logger.info("Using cgroup CPU quota: %s cores available", ntomp_val)
            return ntomp_val
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    try:
        import multiprocessing

        total_cores = multiprocessing.cpu_count()
        logger.warning("No job scheduler detected. Found %s total system cores.", total_cores)
        logger.warning("This may cause oversubscription on shared systems.")
        logger.warning("Consider setting OMP_NUM_THREADS explicitly for your allocation.")
        logger.info("Defaulting to ntomp=1 for safety on shared systems")
        return 1
    except Exception:
        logger.warning("Could not detect CPU cores, defaulting to ntomp=1")
        return 1
