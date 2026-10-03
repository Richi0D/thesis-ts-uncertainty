import importlib
import sys
import types


def reload_package(prefix: str = "ts_uncertainty", namespace: dict | None = None) -> None:
    """Re-import every module under `prefix` and rebind names in `namespace` (e.g. globals())."""
    names = sorted(m for m in sys.modules if m == prefix or m.startswith(prefix + "."))

    # 1. Drop the old modules from the cache
    for name in names:
        del sys.modules[name]

    # 2. Import them fresh (sorted order: parents before children)
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"Failed to reload {name}: {e!r}")

    # 3. Point notebook names at the new objects
    rebound = 0
    if namespace is not None:
        for var, obj in list(namespace.items()):
            if isinstance(obj, types.ModuleType):
                mod_name = obj.__name__
                if mod_name.startswith(prefix) and mod_name in sys.modules:
                    namespace[var] = sys.modules[mod_name]
                    rebound += 1
            elif isinstance(obj, (type, types.FunctionType)):
                mod_name = getattr(obj, "__module__", "") or ""
                if mod_name.startswith(prefix) and mod_name in sys.modules:
                    new_obj = getattr(sys.modules[mod_name], obj.__name__, None)
                    if new_obj is not None:
                        namespace[var] = new_obj
                        rebound += 1

    print(f"Reloaded {len(names)} modules, rebound {rebound} names.")