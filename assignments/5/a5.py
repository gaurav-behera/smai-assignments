import argparse
from kde import task_2_2, task_2_3
from hmm import task_3_2, task_3_4

def main():
    parser = argparse.ArgumentParser(description="Driver for Assignment 3.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["2.2", "2.3", "3.2", "3.4"],
        required=True,
        help="Enter the task to run.",
    )
    args = parser.parse_args()

    match args.task:
        case "2.2":
            print("Running task 2.2...")
            task_2_2()
            print("Task 2.2 completed.")
        case "2.3":
            print("Running task 2.3...")
            task_2_3()
            print("Task 2.3 completed.")
        case "3.2":
            print("Running task 3.2...")
            task_3_2()
            print("Task 3.2 completed.")
        case "3.4":
            print("Running task 3.4...")
            task_3_4()
            print("Task 3.4 completed.")
        case _:
            print("Invalid task specified.")
            exit(1)


if __name__ == "__main__":
    main()
