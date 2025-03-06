import os, shutil
import itertools
import wandb
import zipfile


def unzip_internal_file(zip_path, extract_dir, output_filename):
    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        #Create directory structure
        # os.makedirs(os.path.join(extract_dir,params_dir), exist_ok=True)

        # List all files in the ZIP
        file_list = zip_ref.namelist()

        # Find the deepest level of directories
        max_depth = max(f.count('/') for f in file_list)

        # Get all files at the most internal level (deepest directory)
        internal_files = [f for f in file_list if f.count('/') == max_depth]

        # Extract all internal files from the deepest directory level
        for internal_file in internal_files:
            # Extract the file, removing the parent directory structure
            zip_ref.extract(internal_file, path=extract_dir)
            print(f"Extracted file: {internal_file}")

        # Rename the file to remove parent directories
        base_filename = os.path.basename(internal_file)
        extracted_file_path = os.path.join(extract_dir, internal_file)
        final_path = os.path.join(extract_dir, base_filename)

        # Move and delete zip and empty unzipped folder structure
        os.rename(extracted_file_path, os.path.join(extract_dir,output_filename))
        shutil.rmtree(os.path.join(extract_dir, internal_files[0].split("/")[0]))
        os.remove(zip_path)


def fetch_artifacts_from_sweep(sweep_path, params_dict, download_dir="outputs"):
    # Initialize the W&B API
    api = wandb.Api()

    #Fetch sweep
    sweep = api.sweep(sweep_path)

    # List to store the artifact files
    artifact_files = []

    # Iterate over each run in the sweep
    for run in sweep.runs:
        # Check if the specific parameter value matches
        if all(run.config.get(param_key) == param_value for param_key, param_value in params_dict.items()):
            hparams= '_'.join(f"{key}_{value}" for key, value in params_dict.items())
            params_str=hparams + f"_seed_{run.config.get('random_seed')}"
            print(f"Fetching artifacts of run {run.id} ..., with params {params_str}")

            # Get all the artifacts for the run
            for artifact in run.logged_artifacts():
                for file in artifact.files():
                    if 'evaluation.zip' == file.name:
                        artifact.download(root=download_dir)
                        unzip_internal_file(f"{download_dir}/{file.name.split('/')[-1]}",
                                            os.path.join(download_dir,hparams),
                                f"evaluation_{params_str}.csv")
                        artifact_files.append(file)

    return artifact_files

if __name__=="__main__":
    wandb.login()

    sweep_name = "Fixed-Pruning-Sweep20250214_160858"
    sweep_id = "60w0223z"
    sweep_path=f"koala-lab/kinship/{sweep_name}/{sweep_id}"
    vocab_sizes=[16,32,64,128]
    max_lens=[1]

    download_dir=f"outputs_{sweep_name}"

    for v,l in itertools.product(vocab_sizes, max_lens):
        params = {"vocab_size" : v, "max_len" : l}
        artifacts = fetch_artifacts_from_sweep(sweep_path, params, download_dir=download_dir)

    os.rename(download_dir, os.path.join(os.getcwd(),f"../../results/uniform/{download_dir}"))

