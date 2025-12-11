import os, glob
from huggingface_hub import snapshot_download


data_dir = "../Data/LSVQ"
snapshot_download("teowu/LSVQ-videos", repo_type="dataset", local_dir=data_dir, resume_download=True)

gz_files = glob.glob("*.tar.gz")
gz_files.sort()

original_cwd = os.getcwd()
os.chdir(data_dir)

for gz_file in gz_files:
    file_name = os.path.basename(gz_file)
    os.system(f"tar -xzf {file_name}")

os.chdir(original_cwd)