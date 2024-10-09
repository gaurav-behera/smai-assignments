import argparse
from mlp_classification import task_2_1, task_2_3, task_2_4, task_2_5, task_2_6, task_2_7
from mlp_regression import task_3_1, task_3_3, task_3_4, task_3_5

def main():
    parser = argparse.ArgumentParser(description="Driver for Assignment 3.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["2.1", "2.3", "2.4", "2.5", "2.6","2.7", "3.1", "3.3", "3.4", "3.5"],
        required=True,
        help="Enter the task to run.",
    )
    args = parser.parse_args()

    match args.task:
        case "2.1":
            print("Running task 2.1...")
            task_2_1()
            print("Task 2.1 completed.")
        case "2.3":
            print("Running task 2.3...")
            task_2_3()
            print("Task 2.3 completed.")
        case "2.4":
            print("Running task 2.4...")
            task_2_4()
            print("Task 2.4 completed.")
        case "2.5":
            print("Running task 2.5...")
            task_2_5()
            print("Task 2.5 completed.")
        case "2.6":
            print("Running task 2.6...")
            task_2_6()
            print("Task 2.6 completed.")
        case "2.7":
            print("Running task 2.7...")
            task_2_7()
            print("Task 2.7 completed.")
        case "3.1":
            print("Running task 3.1...")
            task_3_1()
            print("Task 3.1 completed.")
        case "3.3":
            print("Running task 3.3...")
            task_3_3()
            print("Task 3.3 completed.")
        case "3.4":
            print("Running task 3.4...")
            task_3_4()
            print("Task 3.4 completed.")
        case "3.5":
            print("Running task 3.5...")
            task_3_5()
            print("Task 3.5 completed.")
        case _:
            print("Invalid task specified.")
            exit(1)


if __name__ == "__main__":
    main()
