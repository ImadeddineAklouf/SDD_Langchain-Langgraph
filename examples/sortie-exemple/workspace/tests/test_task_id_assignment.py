import pytest
from src.task_manager import TaskManager


def test_task_id_assignment(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))
    # No tasks initially
    assert manager.list_tasks() == []

    # Add first task
    t1 = manager.add_task("First")
    assert isinstance(t1.id, int)
    assert t1.id == 1

    # Add second task
    t2 = manager.add_task("Second")
    assert t2.id == 2

    # Add third task
    t3 = manager.add_task("Third")
    assert t3.id == 3

    # Verify order
    ids = [t.id for t in manager.list_tasks()]
    assert ids == [1, 2, 3]


def test_task_id_continuation_after_loading(tmp_path):
    file_path = tmp_path / "tasks.json"
    # Prepopulate file with two tasks
    data = [
        {"id": 1, "title": "Existing 1", "status": "pending"},
        {"id": 2, "title": "Existing 2", "status": "done"},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=4)

    manager = TaskManager(str(file_path))
    # Next id should be 3
    t3 = manager.add_task("New Task")
    assert t3.id == 3
