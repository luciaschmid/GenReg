import subprocess
import argparse

datasets = {
    "ModelNet40": r"modelnet.cs.princeton.edu/ModelNet40.zip",
    "7Scenes": r"https://drive.google.com/file/d/1XdQ3muo5anFA28ZFch06z_iTEjXxLEQi/view",
}

def download_dataset(dataset):
    if dataset not in list(datasets.keys()):
        raise Exception("Not a valid dataset choice!!!")

    if dataset == "7Scenes":
        print("Can not download dataset for you directly. Please download 7Scenes from ",
              "https://drive.google.com/file/d/1XdQ3muo5anFA28ZFch06z_iTEjXxLEQi/view and extract into data/7Scene")
    else:
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
