"""
Google Cloud Functions proxy script.
"""

import requests
import functions_framework


CLOUD_RUN_URL = "https://pdf-processor-service-211638601865.europe-west10.run.app/overlay"


@functions_framework.http
def pdf_proxy(request):
    """Receives a PDF, forwards it to Cloud Run, and returns the response."""
    # headers = [("Access-Control-Allow-Origin", "https://artifactz.github.io")]  # Avoid CORS errors
    headers = [("Access-Control-Allow-Origin", "*")]  # XXX local file testing

    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400, headers

    # reject files larger than 15 MB
    if int(request.headers["Content-Length"]) > 15e6:
        return {"error": "File too large"}, 400, headers

    file = request.files["file"]
    files = {"file": (file.filename, file.stream, file.mimetype)}
    response = requests.post(CLOUD_RUN_URL, files=files)

    for k, v in headers:
        response.headers[k] = v

    return response.content, response.status_code, response.headers.items()
