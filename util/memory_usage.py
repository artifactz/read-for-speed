import os
import psutil


def get_memory_usage():
    """Get the current memory usage of the process in bytes."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # Resident Set Size (RSS)


def get_memory_usage_mb():
    """Get the current memory usage of the process in megabytes."""
    return get_memory_usage() / 1e6
