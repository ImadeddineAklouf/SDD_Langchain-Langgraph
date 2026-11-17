import pytest
from src.task_manager import TaskManager


def test_add_task(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))
    task = manager.add_task("Test Task")
    assert task.title == "Test Task"
    assert task.id == 1
    tasks = manager.list_tasks()
    assert len(tasks) == 1
    assert tasks[0] is task
    assert manager.get_task_by_id(task.id) is task
