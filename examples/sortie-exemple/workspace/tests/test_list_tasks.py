import pytest
from src.task_manager import TaskManager


def test_list_tasks(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))

    # Initially, the list should be empty
    assert manager.list_tasks() == []

    # Add two tasks
    t1 = manager.add_task("Task 1")
    t2 = manager.add_task("Task 2")

    # Retrieve the list of tasks
    tasks = manager.list_tasks()

    # Verify the list contains the added tasks in order
    assert len(tasks) == 2
    assert tasks[0] is t1
    assert tasks[1] is t2
    assert [t.id for t in tasks] == [t1.id, t2.id]
    assert [t.title for t in tasks] == ["Task 1", "Task 2"]
