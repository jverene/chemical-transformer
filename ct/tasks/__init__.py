from .tagged import TaggedArithmetic
from .mixed import MixedArithmetic

TASKS = {
    "tagged": TaggedArithmetic,
    "mixed": MixedArithmetic,
}


def get_task(name: str):
    return TASKS[name]()
