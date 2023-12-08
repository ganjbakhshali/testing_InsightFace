import os
import requests
import tarfile
from tqdm import tqdm

def download_lfw():
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    pairs_url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"

    # Check if lfw.tgz already exists
    if not os.path.exists("lfw.tgz"):
        # Download the LFW dataset
        response = requests.get(lfw_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading LFW') as pbar:
            with open("lfw.tgz", "wb") as f:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

    # Extract the contents of the tarball
    with tarfile.open("lfw.tgz", "r:gz") as tar_ref:
        tar_ref.extractall("lfw_dataset")

    # Download pairs.txt
    pairs_response = requests.get(pairs_url)
    with tqdm(total=len(pairs_response.content), unit='B', unit_scale=True, desc='Downloading pairs.txt') as pbar:
        with open("pairs.txt", "wb") as f:
            for data in pairs_response.iter_content():
                pbar.update(len(data))
                f.write(data)

if __name__ == "__main__":
    download_lfw()
