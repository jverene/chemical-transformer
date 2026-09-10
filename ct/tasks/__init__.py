from .tagged import TaggedArithmetic
from .mixed import MixedArithmetic
from .hard import TaggedHardArithmetic, MixedHardArithmetic

TASKS = {
    "tagged": TaggedArithmetic,
    "mixed": MixedArithmetic,
    "tagged-v2": TaggedHardArithmetic,
    "mixed-v2": MixedHardArithmetic,
}


def get_task(name: str):
    return TASKS[name]()
