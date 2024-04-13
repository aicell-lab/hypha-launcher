import os
import re
import aiohttp
import asyncio
from urllib.parse import urlparse
from os.path import join as pjoin
import xml.etree.ElementTree as ET

from tqdm.asyncio import tqdm_asyncio
import requests


def find_relative_path(url, index_url):
    """
    Finds the relative path of a file URL with respect to an index URL.

    :param url: The full URL to the file.
    :param index_url: The base URL representing the root directory.
    :return: The relative path from the index URL to the file's directory.
    """
    # Parse the URLs to extract path components
    url_path = urlparse(url).path
    index_url_path = urlparse(index_url).path
    # Ensure the base path ends with a slash
    if not index_url_path.endswith("/"):
        index_url_path += "/"
    # Construct the full path for comparison and remove the base URL path
    relative_full_path = url_path.replace(index_url_path, "", 1)
    # Remove the filename from the path to get the directory path
    relative_dir_path = "/".join(relative_full_path.split("/")[:-1])
    return relative_dir_path


async def download_file(url, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    local_filename = os.path.join(dest_dir, url.split("/")[-1])

    tout = aiohttp.ClientTimeout(10**10)
    async with aiohttp.ClientSession(timeout=tout) as session:
        async with session.get(url) as resp:
            total_size = int(resp.headers.get("content-length", 0))
            chunk_size = 1024
            with tqdm_asyncio(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=local_filename,
                ascii=True,
            ) as progress_bar:
                with open(local_filename, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
    return local_filename


async def download_files(urls, dest_dir, n_parallel=5, base_url=None):
    semaphore = asyncio.Semaphore(n_parallel)

    async def bounded_download(url):
        async with semaphore:
            if base_url:
                r_path = find_relative_path(url, base_url)
                dest_dir_ = pjoin(dest_dir, r_path)
            else:
                dest_dir_ = dest_dir
            await download_file(url, dest_dir_)

    await asyncio.gather(*[bounded_download(url) for url in urls])


def download_content(url: str) -> str:
    """
    Downloads the content at the specified URL and returns it as a string.

    :param url: The URL of the content to download.
    :return: The content of the URL as a string.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_s3_xml(content: str, key_pattern: str) -> list[str]:
    # Parse the XML content
    root = ET.fromstring(content)

    # Define the namespace mapping
    ns = {"ns": "http://s3.amazonaws.com/doc/2006-03-01/"}

    # Compile the regex pattern for matching keys
    pattern = re.compile(key_pattern)

    # Find all 'Contents' elements considering the namespace
    contents_elements = root.findall("ns:Contents", ns)

    # Extract the 'Key' element text if it matches the pattern
    matching_keys = []
    for elem in contents_elements:
        key = elem.find("ns:Key", ns)
        if key is None:
            continue
        key_text = key.text
        if key_text and pattern.match(key_text):
            matching_keys.append(key_text)
    return matching_keys


if __name__ == "__main__":
    import fire

    fire.Fire(download_files)
