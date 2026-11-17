import json
import pytest
from src.task_manager import TaskManager


def test_persistence(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))
    t1 = manager.add_task("Task A")
    t2 = manager.add_task("Task B")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == t1.id
    assert data[0]["title"] == t1.title
    assert data[0]["status"] == t1.status
    assert data[1]["id"] == t2.id
    assert data[1]["title"] == t2.title
    assert data[1]["status"] == t2.status


def test_persistence_load(tmp_path):
    file_path = tmp_path / "tasks.json"
    data = [
        {"id": 1, "title": "Existing", "status": "pending"},
        {"id": 2, "title": "Done", "status": "done"}
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    manager = TaskManager(str(file_path))
    tasks = manager.list_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == 1
    assert tasks[0].title == "Existing"
    assert tasks[0].status == "pending"
    assert tasks[1].id == 2
    assert tasks[1].title == "Done"
    assert tasks[1].status == "done"


def test_clear_tasks(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))
    manager.add_task("To be cleared")
    manager.clear_tasks()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == []