from .utils.download import download_files, parse_s3_xml, download_content


S3_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/model-repository/"


class App():
    def __init__(self):
        pass

    async def download_models_from_s3(
            self, pattern: str, dest_dir: str,
            n_parallel: int = 5,
            s3_base_url=S3_BASE_URL):
        """Download models from S3

        Args:
            pattern (str): pattern to match model files
            dest_dir (str): destination directory
            s3_base_url (str, optional): base url of S3.
        """
        print(f"Downloading models from {s3_base_url}")
        xml_content = download_content(s3_base_url)
        items = parse_s3_xml(xml_content, pattern)
        urls = [s3_base_url + item for item in items]
        await download_files(
            urls, dest_dir, n_parallel=n_parallel,
            base_url=s3_base_url)
