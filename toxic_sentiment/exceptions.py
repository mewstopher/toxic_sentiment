class ScriptPathError(Exception):
    pass


class ScriptError(ScriptPathError):
    pass


class PathError(Exception):
    pass
