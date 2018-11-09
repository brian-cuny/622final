from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np
import argparse
from preprocess_data import folder_to_numpy


def train(data, answers):
    """Trains Sodoku solving ML algorithm and returns pickled model for future use

    Args:
        data: (:2D Numpy Array:): The preprocessed data containing sodoku puzzles
        answers (:1D Numpy Array:): The classifications (answers) for the corresponding data

    Returns:
        None
    """

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=5)
    )
    pipeline.fit(data, answers)

    with open('model.p', 'wb') as model_file:
        pickle.dump(pipeline, model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sodoku Solver. Saves Pickle of Model')
    parser.add_argument('folder', help='Folder of Sodoku Puzzles to Train Model')
    parser.add_argument('answers', help='csv file of corresponding solutions. 0 for empty spaces')
    results = parser.parse_args()

    train(folder_to_numpy(results.folder), np.genfromtxt(results.answers, delimiter=','))


