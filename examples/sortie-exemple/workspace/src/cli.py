import argparse
from src.task_manager import TaskManager


def main():
    parser = argparse.ArgumentParser(description="Task Manager CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("title", help="Title of the task")

    list_parser = subparsers.add_parser("list", help="List all tasks")

    done_parser = subparsers.add_parser("done", help="Mark a task as done")
    done_parser.add_argument("id", type=int, help="ID of the task")

    args = parser.parse_args()
    manager = TaskManager()

    if args.command == "add":
        task = manager.add_task(args.title)
        print(f"Added task {task.id}: {task.title}")
    elif args.command == "list":
        tasks = manager.list_tasks()
        for t in tasks:
            print(f"{t.id}. [{t.status}] {t.title}")
    elif args.command == "done":
        try:
            task = manager.mark_done(args.id)
            print(f"Task {task.id} marked as done")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
