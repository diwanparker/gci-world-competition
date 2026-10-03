"""Download official GCI World Competition dataset from Matsuo Lab Google Drive."""
import os
import requests

DATA_DIR = "./input"
FILES = {
    "train.csv": "1ov4BQJcmSidP_YO5gaszsbfkmQgcRj1F",
    "test.csv": "1p6qAIGmpPgbfYpm7Clh6cIPm3768RHL0",
    "sample_submission.csv": "1ryBfQlw55rv2CsLXjLIb9BUwQKwzZQ06",
}


def download_file_from_google_drive(file_id: str, destination: str):
    url = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()
    response = session.get(url, params={"id": file_id}, stream=True)

    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            response = session.get(url, params={"id": file_id, "confirm": v}, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(destination) / (1024 * 1024)
    print(f"Downloaded {destination} ({size_mb:.2f} MB)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, file_id in FILES.items():
        dest_path = os.path.join(DATA_DIR, name)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"File already exists: {dest_path}")
            continue
        print(f"Downloading {name}...")
        download_file_from_google_drive(file_id, dest_path)


if __name__ == "__main__":
    main()
