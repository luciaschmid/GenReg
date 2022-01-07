import subprocess
import argparse

datasets = {
    "ModelNet40": r"modelnet.cs.princeton.edu/ModelNet40.zip",
    "7Scenes": r"download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/tsdf.zip",
}

def download_dataset(dataset):
    if dataset not in list(datasets.keys()):
        raise Exception("Not a valid dataset choice!!!")

    print(f'Downloading {dataset}...')
    subprocess.run(["wget", datasets[dataset], "-P", "data"])
    print(f'Extracting {dataset}...')
    name = "tsdf" if dataset == "7Scenes" else dataset
    subprocess.run(["unzip", "-q", f"data/{name}.zip", "-d", "data"])
    subprocess.run(["rm", f"data/{name}.zip"])
    print(f'{dataset} done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Downloads ModelNet40 or 7Scenes Datasets',
        usage='python download_datasets.py',
        description='',
        add_help=True
    )

    parser.add_argument('-d', '--dataset', type=str, default="ModelNet40", help='Which dataset to download ModelNet40 or 7Scenes')
    args = parser.parse_args()
    dataset = args.dataset

    download_dataset(args.dataset)
