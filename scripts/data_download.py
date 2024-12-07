# data_download.py
# author: Hrayr Muradyan
# date: 2024-12-02

import os
import pandas as pd
import click
import kagglehub
from pathlib import Path

def download_read_combine_data(url):

    path = Path(kagglehub.dataset_download(url))
    train_data_path = path / "train.csv"
    test_data_path = path / "test.csv" 

    train_data = pd.read_csv(train_data_path, index_col = "Unnamed: 0")
    test_data = pd.read_csv(test_data_path, index_col = "Unnamed: 0")

    return pd.concat([train_data, test_data], axis=0).reset_index(drop=True)


@click.command()
@click.option('--url', type=str, help="URL of the dataset to download")
@click.option('--save_to', type=str, help="The path to save the downloaded dataset")
@click.option('--file_to', type=str, help="The file name to save the dataset into")
@click.option('--force_save', type=bool, help="Do you want to overwrite the file if it exists?", default=False)
def main(url, save_to, file_to, force_save):
    save_to = Path(save_to)
    dataset = download_read_combine_data(url)
    if not save_to.exists():
        save_to.mkdir(parents=True, exist_ok=True)

    file_to_save = save_to / file_to

    if file_to_save.exists() and not force_save:
        print(f"""
                The file "{file_to_save}" already exists... 
                The script will not overwrite the file. 
                If you want to force save the file, specify argument --force_save=True
                Terminating the script...
        """)
        return

    if save_to.is_dir():
        dataset.to_csv(file_to_save, index=False)
    else:
        raise ValueError("The argument save_to is not a directory!")
    
if __name__ == '__main__':
    main()

