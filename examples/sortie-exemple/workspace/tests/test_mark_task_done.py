import pytest
from src.task_manager import TaskManager


def test_mark_task_done(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))

    # Add a task and mark it as done
    task = manager.add_task("Sample Task")
    assert task.status == "pending"

    done_task = manager.mark_done(task.id)
    assert done_task.status == "done"
    # The returned task should be the same instance
    assert done_task is task

    # Verify that the task is persisted with updated status
    new_manager = TaskManager(str(file_path))
    persisted_task = new_manager.get_task_by_id(task.id)
    assert persisted_task is not None
    assert persisted_task.status == "done"


def test_mark_done_invalid_id(tmp_path):
    file_path = tmp_path / "tasks.json"
    manager = TaskManager(str(file_path))

    # No tasks yet, so any id should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        manager.mark_done(999)
    assert "Task with id 999 not found" in str(excinfo.value)
