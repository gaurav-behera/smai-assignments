# driver code for all the other python files
import argparse
from k_means_tasks import task_3_2
from gmm_tasks import task_4_2
from pca_tasks import task_5_2

def main():
    parser = argparse.ArgumentParser(description="Driver for Assignment 2.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["3.2", "4.2", "5.2"],
        required=True,
        help="Enter the task to run.",
    )
    args = parser.parse_args()

    match args.task:
        case "3.2":
            print("Running task 3.2...")
            task_3_2()
            print("Task 3.2 completed.")
        case "4.2":
            print("Running task 4.2...")
            task_4_2()
            print("Task 4.2 completed.")
        case "5.2":
            print("Running task 5.2...")
            task_5_2()
            print("Task 5.2 completed.")
        case _:
            print(
                "Invalid task specified."
            )
            exit(1)


if __name__ == "__main__":
    main()
