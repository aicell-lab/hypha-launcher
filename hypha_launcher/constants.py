
S3_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/model-repository/"
TRITON_IMAGE = "docker://nvcr.io/nvidia/tritonserver:23.03-py3"
S3_IMAGE = "docker://minio/minio:RELEASE.2022-09-01T23-53-36Z.fips"

IMJOY_APP_TEMPLATE = """
{app_code}

import asyncio
loop = asyncio.get_event_loop()
loop.create_task(hypha_startup())
loop.run_forever()
"""
