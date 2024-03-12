"""Provide common pytest fixtures."""
import os
import subprocess
import sys
import time
import uuid

import pytest
import requests
from requests import RequestException

WS_PORT = 7891


JWT_SECRET = str(uuid.uuid4())
os.environ["JWT_SECRET"] = JWT_SECRET
test_env = os.environ.copy()


@pytest.fixture(name="hypha_server", scope="session")
def hypha_server():
    """Start server as test fixture and tear down after test."""
    with subprocess.Popen(
        [sys.executable, "-m", "hypha.server", f"--port={WS_PORT}"],
        env=test_env,
    ) as proc:
        addr = f"http://127.0.0.1:{WS_PORT}"
        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(f"{addr}/health/liveness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise RuntimeError("Failed to start websocket server")
        yield addr
        proc.kill()
        proc.terminate()
