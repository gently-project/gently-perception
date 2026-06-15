"""Task registry — populated lazily so importing harness.tasks is cheap."""
from __future__ import annotations

from harness.core.types_v2 import Task

_TASKS: dict[str, Task] | None = None


def get_tasks() -> dict[str, Task]:
    global _TASKS
    if _TASKS is not None:
        return _TASKS
    from harness.tasks.onset import dopaminergic

    _TASKS = {
        dopaminergic.TASK.name: dopaminergic.TASK,
    }
    return _TASKS


def get_task(name: str) -> Task:
    tasks = get_tasks()
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; known: {sorted(tasks)}")
    return tasks[name]
