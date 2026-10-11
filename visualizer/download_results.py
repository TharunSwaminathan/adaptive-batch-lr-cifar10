
import json
import urllib.request
from pathlib import Path

REPO = "TharunSwaminathan/adaptive-batch-lr-cifar10"
BRANCH = "jiong"

# GitHub experiment folders and our local destinations
EXPERIMENTS = {
    "20261008T215541_390318Z": "cifar10/50epochs",
    "20261008T222534_830127Z": "cifar10/80epochs",
    "20261007T062522_033784Z": "cifar100/120epochs",
}

BASE = f"https://api.github.com/repos/{REPO}/contents"
OUTPUT = Path("public/data")


def github_contents(path):
    """Retrieve a directory listing from GitHub."""
    url = f"{BASE}/{path}?ref={BRANCH}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CNN-Visualizer"}
    )

    with urllib.request.urlopen(request) as response:
        return json.load(response)


def find_csv(path):
    """Search experiment folders for the training CSV."""
    items = github_contents(path)

    for item in items:
        if item["type"] == "file" and item["name"].endswith(".csv"):
            return item["download_url"]

        if item["type"] == "dir":
            result = find_csv(item["path"])
            if result:
                return result

    return None


def download_file(url, destination):
    """Save a file from GitHub to our local project."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CNN-Visualizer"}
    )

    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())


for run_folder, local_folder in EXPERIMENTS.items():
    for experiment in ["E1", "E2", "E3", "E4"]:
        github_path = (
            f"results/unified_experiments/"
            f"{run_folder}/runs/{experiment}"
        )

        csv_url = find_csv(github_path)

        if csv_url is None:
            print(f"NOT FOUND: {local_folder}/{experiment}")
            continue

        destination = (
            OUTPUT / local_folder / f"{experiment.lower()}.csv"
        )

        download_file(csv_url, destination)
        print(f"Downloaded: {destination}")

print("Download process completed.")
