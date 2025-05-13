"""
Main functions.
"""

import os, logging, time
from typing import IO
from util.memory_usage import get_memory_usage_mb


def run(input_pdf_file: IO | str, output_pdf_file: IO | str):
    """
    Default main function, running everything in the main process.
    """
    t0 = time.time()
    logging.info(f"Memory usage before imports: {get_memory_usage_mb()}")
    from pdf_overlay import generate_text_overlay
    from pdf_merge import merge

    logging.info(f"Memory usage before overlay generation: {get_memory_usage_mb()}")
    metadata = generate_text_overlay(input_pdf_file)
    if hasattr(input_pdf_file, "seek"):
        input_pdf_file.seek(0)
    overlay_pdf_path = metadata["path"]

    logging.info(f"Memory usage before merge: {get_memory_usage_mb()}")
    merge(input_pdf_file, overlay_pdf_path, output_pdf_file)
    os.remove(overlay_pdf_path)
    del metadata["path"]
    logging.info(f"Memory usage after merge: {get_memory_usage_mb()}")

    return dict(metadata, total_duration=time.time() - t0)


def run_processes(input_pdf_path: str, output_pdf_path: str):
    """
    Main function that runs overlay generation and merging in separate processes to free up more memory.
    Inputs have to be paths to files.
    """
    t0 = time.time()
    import subprocess, json

    logging.info(f"Memory usage before overlay generation: {get_memory_usage_mb()}")
    p = subprocess.Popen(["python", "pdf_overlay.py", input_pdf_path], stdout=subprocess.PIPE)
    metadata = json.load(p.stdout)
    p.stdout.close()
    p.wait()
    overlay_pdf_path = metadata["path"]

    logging.info(f"Memory usage before merge: {get_memory_usage_mb()}")
    p = subprocess.Popen(["python", "pdf_merge.py", input_pdf_path, overlay_pdf_path, output_pdf_path])
    p.wait()
    os.remove(overlay_pdf_path)
    del metadata["path"]
    logging.info(f"Memory usage after merge: {get_memory_usage_mb()}")

    return dict(metadata, total_duration=time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_pdf_path = "samples/encrypted/sample34.pdf"
    output_pdf_path = "out/pdf/sample34.pdf"
    metadata = run(input_pdf_path, output_pdf_path)
    # metadata = run_processes(input_pdf_path, output_pdf_path)
    print("Metadata:", metadata)
    print(f"Overlay added successfully. Saved as {output_pdf_path}")
