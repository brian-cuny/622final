import numpy as np
import os
import cv2
import image_slicer
from shutil import rmtree
import argparse
import pickle


def folder_to_numpy(folder):
    """Return 2D Numpy array of sodoku puzzle cells for use in machine learning algorithm

    Args:
        folder (:str:): The folder that contains all the Sodoku images

    Returns:
        2D Numpy Array
    """

    if not os.path.exists('temp'):
        os.mkdir('temp')
    to_ret = np.array([ele for file in os.listdir(folder) for ele in separate_file(os.path.join(folder, file))])
    rmtree('temp')
    return to_ret.astype(float)


def separate_file(file_name):
    """Private helper method that extracts each cell

    Args:
        file_name (:str:): File to slice into 81 cells and trims

    Yields:
        2D Numpy Array representing each cell
    """

    def mask_calculations(mask):
        mask_left = mask[:, 0:mask.shape[1] // 2]
        l = np.max(np.append(np.where(mask_left.all(axis=0))[0], 0))

        mask_right = mask[:, mask.shape[1] // 2:mask.shape[1]]
        r = np.min(np.append(np.where(mask_right.all(axis=0))[0], mask.shape[1] // 2)) + mask.shape[1] // 2

        mask_top = mask[0:mask.shape[0] // 2, :]
        t = np.max(np.append(np.where(mask_top.all(axis=1))[0], 0))

        mask_bottom = mask[mask.shape[0] // 2:mask.shape[0], :]
        b = np.min(np.append(np.where(mask_bottom.all(axis=1))[0], mask.shape[0] // 2)) + mask.shape[0] // 2

        return t, r, b, l

    original = cv2.imread(file_name, 0)
    top, right, bottom, left = mask_calculations(original > 30)

    if abs(left - right) < original.shape[1] * .1 or abs(top - bottom) < original.shape[0] * .1:
        cv2.imwrite(os.path.join('temp', 'original.jpg'), original)
    else:
        cv2.imwrite(os.path.join('temp', 'original.jpg'), original[top:bottom, left:right])

    tiles = image_slicer.slice(os.path.join('temp', 'original.jpg'), 81, save=False)
    image_slicer.save_tiles(tiles, directory='temp', prefix='temp')

    for j in range(1, 10):
        for i in range(1, 10):
            file = cv2.imread(os.path.join('temp', f'temp_0{j}_0{i}.png'), 0)
            top, right, bottom, left = mask_calculations(file < 235)
            best = max([left, top, file.shape[0] - bottom, file.shape[1] - right])
            file = file[best + 1:file.shape[0] - best - 1, best + 1:file.shape[1] - best - 1]

            mask = file < 200
            coordinates = np.argwhere(mask)
            if len(coordinates) == 0:
                yield cv2.resize(file, (28, 28)).flatten()
            else:
                x0, y0 = coordinates.min(axis=0)
                x1, y1 = coordinates.max(axis=0) + 1
                yield cv2.resize(file[x0:x1, y0:y1], (28, 28)).flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess folder of Sodoku puzzles into Numpy for use in associated ML algorithm. '
                    'Data Saved as Pickle')
    parser.add_argument('folder', help='Folder path that contains sodoku puzzles')
    results = parser.parse_args()

    with open('data.p', 'wb') as file:
        pickle.dump(folder_to_numpy(results.folder), file)