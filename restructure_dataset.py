"""
Code restructures original rgb cafo dataset to feed to interpolation pipeline.

rgb_dir should be path to rgb cafo dataset, and num_files should equal number of frames
each folder needs to have (current default is 12).
"""

import shutil
import os


def delete_empty_folders(main_dir):
    """Delete empty loc files."""

    for folder in sorted(os.listdir(main_dir)):
        if 'DS_Store' not in folder:
            curr_dir = os.path.join(main_dir, folder)
            if len(os.listdir(curr_dir)) == 0:
                os.rmdir(curr_dir)


def delete_incomplete_folders(main_dir, num_files):
    """Delete folders with less than specified number of frames."""

    for folder in sorted(os.listdir(main_dir)):
        if 'DS_Store' not in folder:
            folder_path = os.path.join(main_dir, folder)
            image_count = len(os.listdir(folder_path))
            if image_count != num_files:
                shutil.rmtree(folder_path)


def restructure_dataset(main_dir):
    """Move all year folders up one level. """

    for loc in sorted(os.listdir(main_dir)):
        if 'DS_Store' not in loc:
            loc_path = os.path.join(main_dir, loc)
            for year in sorted(os.listdir(loc_path)):
                old_path = os.path.join(loc_path, year)
                new_path = loc_path + "_" + year
                shutil.move(old_path, new_path)

    delete_empty_folders(main_dir)


def divide_folders(main_dir, num_files):
    """Divide folders to have specified number of frames."""

    count = 0
    curr_subdir = None

    for folder in sorted(os.listdir(main_dir)):
        if 'DS_Store' not in folder:
            folder_dir = os.path.join(main_dir, folder)

            for image in sorted(os.listdir(folder_dir)):
                if count % num_files == 0:
                    subdir_name = os.path.join(main_dir, '{0:03d}'.format(count // num_files + 1))
                    os.mkdir(subdir_name)
                    curr_subdir = subdir_name
                shutil.move(os.path.join(folder_dir, image), os.path.join(curr_subdir, image))
                count += 1

    delete_empty_folders(main_dir)
    delete_incomplete_folders(main_dir, num_files)


def main():
    rgb_dir = '/path/to/rgb/dataset'

    restructure_dataset(rgb_dir)
    divide_folders(rgb_dir, num_files=12)


if __name__ == '__main__':
    main()
