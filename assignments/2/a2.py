# driver code for all the other python files
import argparse
from k_means_tasks import task_3_2, task_6_1, task_6_2_b, task_7_1
from gmm_tasks import task_4_2, task_6_3, task_6_4, task_7_2
from pca_tasks import task_5_2, task_5_3, task_6_2_a

def main():
    parser = argparse.ArgumentParser(description="Driver for Assignment 2.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["3.2", "4.2", "5.2", "5.3", "6.1", "6.2", "6.3", "6.4", "7.1", "7.2"],
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
        case "5.3":
            print("Running task 5.3...")
            task_5_3()
            print("Task 5.3 completed.")
        case "6.1":
            print("Running task 6.1...")
            task_6_1()
            print("Task 6.1 completed.")
        case "6.2":
            print("Running task 6.2...")
            task_6_2_a() # scree plot and reduced dataset
            task_6_2_b() # kmeans
            print("Task 6.2 completed.")
        case "6.3":
            print("Running task 6.3...")
            task_6_3()
            print("Task 6.3 completed.")
        case "6.4":
            print("Running task 6.4...")
            task_6_4()
            print("Task 6.4 completed.")
        case "7.1":
            print("Running task 7.1...")
            task_7_1()
            print("Task 7.1 completed.")
        case "7.2":
            print("Running task 7.2...")
            task_7_2()
            print("Task 7.2 completed.")
        case _:
            print(
                "Invalid task specified."
            )
            exit(1)


if __name__ == "__main__":
    main()
