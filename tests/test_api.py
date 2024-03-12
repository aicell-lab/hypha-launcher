import asyncio
from hypha_launcher.api import HyphaLauncher
from imjoy_rpc.hypha import connect_to_server

import pytest


@pytest.mark.asyncio
async def test_launch_s3():
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_s3_server("test", "test")
    await asyncio.sleep(5)
    await job['stop']()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_launch_triton(hypha_server):
    server = await connect_to_server({"server_url": hypha_server})
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_triton_server(server)
    await asyncio.sleep(15)
    await job['stop']()
