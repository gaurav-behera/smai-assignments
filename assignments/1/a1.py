# driver code for all the other python files
import argparse
from knn_spotify_tasks import run_all_knn
from regression_tasks import run_all_regression
from regularization_tasks import run_all_regularization


def main():
    parser = argparse.ArgumentParser(description="Driver for Assignment 1.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["knn", "regression", "regularization"],
        required=True,
        help="Choose the task to run: 'knn', 'regression' or 'regularization'.",
    )
    args = parser.parse_args()

    match args.task:
        case "knn":
            print("Running all the KNN tasks...")
            run_all_knn()
            print("All KNN tasks completed.")
        case "regression":
            print("Running all the regression tasks...")
            run_all_regression()
            print("All regression tasks completed.")
        case "regularization":
            print("Running all the regularization tasks...")
            run_all_regularization()
            print("All regularization tasks completed.")
        case _:
            print(
                "Invalid task specified. Please choose from 'knn', 'regression', or 'regularization'."
            )
            exit(1)


if __name__ == "__main__":
    main()
