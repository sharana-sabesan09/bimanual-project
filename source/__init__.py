import importlib
import pkgutil
from pathlib import Path


def import_tasks():
    tasks_path = Path(__file__).parent / "tasks"
    for _, name, _ in pkgutil.iter_modules([str(tasks_path)]):
        importlib.import_module(f"source.tasks.{name}")


import_tasks()
