import shutil
import os

N = 3  # the number of files in subfolder


def move_files(abs_dirname):
    """Move files into subdirectories."""

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    files.sort()

    i = 0
    curr_subdir = None

    for f in files:
        if not ('DS_Store') in f:
            # create new subdir if necessary
            if i % N == 0:
                subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i // N + 1))
                os.mkdir(subdir_name)
                curr_subdir = subdir_name

            # move file to current dir
            f_base = os.path.basename(f)

            shutil.move(f, os.path.join(curr_subdir, f_base))
            i += 1


def delete_incomplete_folders(main_dir):
    """Delete folders with less than specified number of frames."""

    for filename in sorted(os.listdir(main_dir)):
        if not ('DS_Store') in filename:
            for subfile in sorted(os.listdir(os.path.join(main_dir, filename))):
                if not ('DS_Store') in subfile:
                    full_path = os.path.join(main_dir, filename, subfile)
                    image_count = len(os.listdir(full_path))
                    if image_count != N:
                        shutil.rmtree(full_path)


def rename_files(main_dir):
    name_list = ["frame_00.png", "frame_01_gt.png", "frame_02.png"]
    for filename in sorted(os.listdir(main_dir)):
        if not ('DS_Store') in filename:
            for subfile in sorted(os.listdir(os.path.join(main_dir, filename))):
                if not ('DS_Store') in subfile:
                    full_path = os.path.join(main_dir, filename, subfile)
                    count = 0
                    for images in sorted(os.listdir(full_path)):
                        os.rename(os.path.join(full_path, images), os.path.join(full_path, name_list[count]))
                        count+=1


def main():
    """After moving 2018 to loc_2018"""

    src_dir = '/Users/hannahkim/Desktop/slomo/test_dataset'

    for filename in sorted(os.listdir(src_dir)):
        if not ('DS_Store') in filename:
            main_path = os.path.join(src_dir, filename)
            move_files(main_path)

    delete_incomplete_folders(src_dir)

    rename_files(src_dir)


if __name__ == '__main__':
    main()
