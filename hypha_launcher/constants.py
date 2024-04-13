
S3_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk"
S3_MODELS_URL = f"{S3_BASE_URL}/model-repository/"
S3_CONDA_ENVS_URL = f"{S3_BASE_URL}/bioimage.io/environments"
TRITON_IMAGE = "docker://nvcr.io/nvidia/tritonserver:23.03-py3"
S3_IMAGE = "docker://minio/minio:RELEASE.2023-10-24T04-42-36Z.hotfix.793d9ee53"

DOCKER_REGISTRY_PREFIX = "docker://ghcr.io/bioimage-io"

IMJOY_APP_TEMPLATE = """
{app_code}

async def main():
    services = []
    server = api
    _register_service = server.register_service
    async def patched_register_service(*args, **kwargs):
        svc_info = await _register_service(*args, **kwargs)
        services.append(svc_info)
        return svc_info
    server.register_service = patched_register_service
    server.registerService = patched_register_service

    try:
        client_id = server.rpc.get_client_info()['id']
        await hypha_startup(server)
        report_service = await server.get_service("hypha-launcher")
        await report_service.report_services(client_id, services)
    except Exception as e:
        print("Failed to start hypha server:", e)
        await report_service.report_error(client_id, str(e))

import asyncio
loop = asyncio.get_event_loop()
loop.create_task(main)
loop.run_forever()
"""


REPORT_ADDRESS_SCRIPT = """
import json
import http.client
from executor.engine.utils import PortManager

http_port = PortManager.get_port()
task_id = "{task_uuid}"
post_data = {{"port": http_port, "uuid": task_id}}
post_data_json = json.dumps(post_data)
for ip in {host_ips}:
    try:
        conn = http.client.HTTPConnection(ip, {ip_record_server_port})
        conn.request("POST", "/", post_data_json, {{"Content-Type": "application/json"}})
        response = conn.getresponse()
        if response.status == 200:
            break
    except Exception as e:
        print("Failed to record IP:", e)
else:
    raise ValueError("Failed to record IP")

"""

LAUNCH_TRITON_SCRIPT = REPORT_ADDRESS_SCRIPT + f"""
import os
from hypha_launcher.utils.container import ContainerEngine

container_engine_kwargs = {{container_engine_kwargs}}
container_engine = ContainerEngine(**container_engine_kwargs)

models_dir = "{{model_repository}}"
volumes = dict([(models_dir, "/models")])

triton_cmd = 'bash -c "tritonserver --model-repository=/models --log-verbose=3 --log-info=1 --log-warning=1 --log-error=1 --model-control-mode=poll --exit-on-error=false --repository-poll-secs=10 --allow-grpc=False --http-port=' + str(http_port) + '"'
cmd = container_engine.get_command(
    triton_cmd, "{TRITON_IMAGE}",
    volumes=volumes,
)
print(cmd)
os.system(cmd)
"""
