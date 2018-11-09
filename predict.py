import numpy as np
from pandas import DataFrame
import pickle
import argparse
from preprocess_data import folder_to_numpy


def solve_sodoku(puzzle):
    """Sodoku puzzle solver. Takes in puzzle and returns solved puzzle or Numpy of zeros

    Args:
        puzzle: (:2D Numpy Array:): 9 by 9 2D Numpy Array of Sodoku puzzle to solve. 0s in place of empty elements

    Returns:
        2D Numpy Array: The solution or zeros if puzzle cannot be solved
    """

    def possible(info):
        for ele in range(81):
            if info[ele // 9, ele % 9] != 0:
                yield np.array([info[ele // 9, ele % 9]])
            else:
                invalid = np.unique(np.concatenate([info[ele // 9, :],
                                                    info[:, ele % 9],
                                                    info[ele // 9 - ele // 9 % 3: ele // 9 - ele // 9 % 3 + 3,
                                                    ele % 9 - ele % 3: ele % 9 - ele % 3 + 3
                                                    ].flatten()
                                                    ]))
                yield np.array([i for i in range(1, 10) if i not in invalid])

    puzzle = DataFrame(data={'possible': [x for x in possible(puzzle)]})
    i = 0
    mapping = np.zeros((9, 9))
    while i != 81:
        possibles = puzzle.iat[i, 0]

        if mapping[i // 9, i % 9] == np.max(possibles):
            mapping[i // 9, i % 9] = 0
            i -= 1
            if i == -1:
                print("Puzzle cannot be solved")
                return np.zeros((9, 9))
            continue
        else:
            set_possibles = set(possibles[np.where(possibles > mapping[i // 9, i % 9])])
            invalid = set(np.concatenate([mapping[i // 9, 0:i % 9],
                                          mapping[0:i // 9, i % 9],
                                          mapping[i // 9 - i // 9 % 3:i // 9, i % 9 - i % 3:i % 9 - i % 3 + 3].flatten()
                                          ]))

            results = set_possibles - invalid
            if len(results) == 0:
                mapping[i // 9, i % 9] = 0
                i -= 1
            else:
                mapping[i // 9, i % 9] = min(results)
                i += 1

    return mapping


def predict(s_model, data):
    """Displays predictions of preprocessed Sodoku puzzles.

    Args:
        s_model: (:Pickle:): Pickle file of Sodoku Model
        data: (:2D Numpy Array:): Preprocessed 2D Numpy Array of puzzles to solve

    Returns:
         Nothing
    """

    with open(s_model, 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    print('Completed Sodoku Puzzles:')
    predictions = model.predict(data)

    for puzzle in predictions.reshape((predictions.shape[0]//81, 9, 9)):
        print(solve_sodoku(puzzle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with Sodoku Solver Model')
    parser.add_argument('model', help='Pickled Sodoku Solver Model')
    parser.add_argument('folder', help='Folder with Sodoku Puzzles to Solve')
    results = parser.parse_args()

    predict(results.model, folder_to_numpy(results.folder))