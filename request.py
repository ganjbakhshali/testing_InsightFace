import requests

import requests
import os
import random
from tqdm import tqdm


url = "http://127.0.0.1:8000/recognize"
filenames = next(os.walk("io/CALFW_test/"))[2]
filenames=random.sample(filenames, 6000)
acc=0
for filename in tqdm(filenames):
    files = {"file": (filename, open(f"io/CALFW_test/{filename}", "rb"))}
    params = {"update": False, "origin_size": True, "tta": False, "show": False, "save": True}

    response = requests.post(url, files=files, data=params)

    print(response.json()['name_recognized'])
    print(response.json()['file_path'])
    if response.json()['name_recognized'] in response.json()['file_path']:
        acc+=1
        print(f"num of correct recognized:{acc}")

print(f"accuracy = {(acc/6000)*100}")