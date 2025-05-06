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


def get_top_memory_users_mb():
    """Get the top 10 processes with the highest memory usage."""
    this_pid = os.getpid()
    all_processes = [(p.info["cmdline"], p.info["memory_info"].rss / 1e6) for p in psutil.process_iter(attrs=["cmdline", "memory_info"]) if p.pid != this_pid]
    all_processes.sort(key=lambda x: x[1], reverse=True)
    return all_processes[:10]
