# Hypha Launcher

A hypha service for launching jobs on HPC, Kubernetes, or local machine.

## **Features**
 - Support multiple backends (HPC, Kubernetes, Local)
 - Provide S3-like storage interface
 - Support launching triton servers for AI model serving

## Hypha Launcher API

## Basic Usage

To begin using `hypha-launcher`, you need to import the `HyphaLauncher` class from the `hypha_launcher.api` module.

```python
from hypha_launcher.api import HyphaLauncher
```

You can create an instance of `HyphaLauncher` by simply calling it:

```python
hypha_launcher = HyphaLauncher()
```

Optionally, you can pass `hpc_manager_kwargs` to customize settings for HPC job management:

```python
hypha_launcher = HyphaLauncher(hpc_manager_kwargs={"hpc_type": "local"})
```

## API Methods

Here are the primary methods available in `HyphaLauncher`:

### `launch_s3_server()`

Launches an S3 compatible server.

**Example**:
```python
async def launch_s3():
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_s3_server()
    await asyncio.sleep(10)  # Let the server run for a while
    await job['stop']()  # Stop the server
```

### `launch_ip_record_server()`

Starts an IP record server, useful for logging and tracking IP usage.

**Example**:
```python
async def setup_ip_record():
    hypha_launcher = HyphaLauncher()
    job = await hypha_launcher.launch_ip_record_server()
    return job
```

### `launch_triton_server()`

Deploys a Triton Inference Server, which is an open-source inference server.

**Example**:
```python
async def launch_triton():
    hypha_launcher = HyphaLauncher()
    ip_job = await hypha_launcher.launch_ip_record_server()
    job = await hypha_launcher.launch_triton_server()
    print(job['address'])
    await asyncio.sleep(15)
    await job['stop']()
    await ip_job['stop']()
```

### `launch_command(command: str)`

Launches a generic command as a job. This can be used to run any executable or script.

**Example**:
```python
async def run_http_server():
    hypha_launcher = HyphaLauncher(hpc_manager_kwargs={"hpc_type": "local"})
    port = hypha_launcher.get_free_port()
    job = await hypha_launcher.launch_command(f"python -m http.server {port}")
    await asyncio.sleep(5)
    await job['stop']()
```

### `get_free_port()`

Returns an available port on the host machine.

**Example**:
```python
port = hypha_launcher.get_free_port()
print(f"Available port: {port}")
```

## Advanced Usage

For advanced use cases such as deploying on Kubernetes clusters or integrating with existing HPC environments, additional configuration and setup may be required. These are typically managed through the `hpc_manager_kwargs` during the initialization of the `HyphaLauncher` instance and may involve setting up credentials, specifying cluster details, etc.
