import asyncio
from hypha_launcher.api import HyphaLauncher

import pytest


@pytest.mark.asyncio
async def test_launch_s3():
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_s3_server()
    await asyncio.sleep(10)
    await job['stop']()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_launch_triton():
    hypha_launcher = HyphaLauncher()
    ip_job = await hypha_launcher.launch_ip_record_server()
    job = await hypha_launcher.launch_triton_server()
    print(job['address'])
    await asyncio.sleep(15)
    await job['stop']()
    await ip_job['stop']()


@pytest.mark.asyncio
async def test_launch_command():
    hypha_launcher = HyphaLauncher(hpc_manager_kwargs={"hpc_type": "local"})
    port = hypha_launcher.get_free_port()
    job = await hypha_launcher.launch_command(f"python -m http.server {port}")
    await asyncio.sleep(5)
    await job['stop']()
    await asyncio.sleep(1)
