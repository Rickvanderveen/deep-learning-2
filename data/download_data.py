# This script downloads the all data from the huggingface repo.
# All data will be downloaded in the `local_dir`, which is set to the parent directory
# of this file.
# 
# Since the hugginface repo is private you need a token. This token can be set as an
# environment variable.
# On linux:
# 
#   $ export HF_TOKEN=<token-to-acces-repo>
# 

from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    file_path = Path(__file__)
    data_dir = file_path.parent

    # Downloading the data 
    snapshot_download(
            repo_id="Rickvanderveen/spai-dataset",
            repo_type="dataset",
            local_dir=data_dir.absolute(),
    )

if __name__ == "__main__":
    main()