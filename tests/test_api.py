import asyncio
from hypha_launcher.api import HyphaLauncher
from imjoy_rpc.hypha import connect_to_server

import pytest


@pytest.mark.asyncio
async def test_launch_s3():
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_s3_server()
    await asyncio.sleep(10)
    await job['stop']()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_launch_triton(hypha_server):
    server = await connect_to_server({"server_url": hypha_server})
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_triton(server, hpc_type="local")
    await asyncio.sleep(15)
    await job['stop']()


@pytest.mark.asyncio
async def test_launch_command():
    hypha_launcher = HyphaLauncher(hpc_manager_kwargs={"hpc_type": "local"})
    port = hypha_launcher.get_free_port()
    job = await hypha_launcher.launch_command(f"python -m http.server {port}")
    await asyncio.sleep(5)
    await job['stop']()
    await asyncio.sleep(1)
