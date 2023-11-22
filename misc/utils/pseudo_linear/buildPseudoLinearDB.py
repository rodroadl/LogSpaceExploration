# Heather Fryling
# Fall 2023
# Builds a pseudolinear/pseudolog database.
#
# Usage:
# python fname input_dir output_dir file_ending1 file_ending2 ...
#
# Example 1:
# python buildPseudolinearDB.py dataset processed_dataset .png .jpg
# 
# This will build pseudolinear and pseudolog versions of the dataset in the directory
# processed_dataset for all files ending with .png or .jpg.
#
# Example 2:
# python buildPseudolinearDB.py dataset processed_dataset
#
# This will build pseudolinear and pseudolog versions of the dataset in the directory
# processed_dataset for all files.
# The file_endings are optional parameters. If none are given, the program will attempt to
# process all files.
#
# Pseudolinear images are 16-bit tiffs generated by reversing sRGB, then multiplying by 65535.
# Pseudolog images are 32-bit exrs generated by taking the natural log of the pseudolinear images.
# 
# The file structure of the original database is searched recursively and recreated exactly in 
# output_directory/pseudolinear and output_directory/pseudolog.

import os
import sys
from img_utility import *
from pseudo_utility import *
from file_utility import get_filenames
from tqdm import tqdm

PSEUDOLINEAR = 'pseudolinear'
PSEUDOLOG = 'pseudolog'

def main(argv):
  if len(argv) < 3:
    print(f"usage: python {argv[0]} <original directory> <output directory> optional:<file_ending>")
    return
  original_directory = argv[1]
  output_directory = argv[2]
  i = 3
  endings = []
  while i < len(argv):
    endings.append(argv[i])
    i += 1
  endings = None if len(endings) < 1 else endings
  os.makedirs(os.path.join(output_directory, PSEUDOLINEAR), exist_ok=True)
  os.makedirs(os.path.join(output_directory, PSEUDOLOG), exist_ok=True)
  fpaths = get_filenames(original_directory, endings=endings)
  for img_path in tqdm(fpaths, desc=f'Building pseudolinear/log DB.'):
    path_split = img_path.split(os.path.sep)
    write_dir = os.path.join(*path_split[1:-1])
    fname = path_split[-1]
    linear_dir = os.path.join(output_directory, PSEUDOLINEAR, write_dir)
    log_dir = os.path.join(output_directory, PSEUDOLOG, write_dir)
    if not os.path.exists(linear_dir):
        os.makedirs(linear_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    img = load_img(img_path)
    img = srgb_to_xyz(img, max_val=255)
    img = float_to_tiff(img)
    img = bgr_to_rgb(img)
    fname, _ = fname.split('.')
    tiffname = f'{fname}.tiff'
    exrname = f'{fname}.exr'
    save_tiff(img, os.path.join(linear_dir, tiffname))
    img = tiff_to_log(img)
    save_log(img, os.path.join(log_dir, exrname))

if __name__ == "__main__":
    main(sys.argv)

    