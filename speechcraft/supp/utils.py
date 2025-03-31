import re
import unicodedata


def encode_path_safe(filename: str, allow_unicode=False):
    """
    Makes a string path safe by removing / replacing not by the os allowed patterns.
    This converts:
    spaces 2 dashes, repeated dashes 2 single dashes, remove non alphanumerics, underscores, or hyphen, string 2 lowercase
    strip
    """
    filename = str(filename)
    if allow_unicode:
        filename = unicodedata.normalize('NFKC', filename)
    else:
        filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\s-]', '', filename.lower())
    return re.sub(r'[-\s]+', '', filename).strip('-_')


def get_cpu_or_gpu() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def create_progress_tracker(original_update_func: callable, steps: list = None):
    """
    Create a progress tracking function with predefined step weights.

    :param steps: A list of tuples (step_name, step_weight)
                  Weights should sum to 100
    :return: Callable progress update function
    """
    if not steps or not isinstance(steps, list) or not all(
            isinstance(step, tuple) and len(step) == 2 for step in steps
    ):
        steps = [("one", 100)]

    total_steps = sum(step[1] for step in steps)
    if total_steps != 100:
        raise ValueError("Step weights must sum to 100")

    current_progress = 0.0
    prev_progress = 0.0
    step_index = 0

    def progress_update_func_block(x):
        nonlocal current_progress, step_index, prev_progress

        # Calculate actual progress for current step
        current_step_weight = steps[step_index][1]
        step_progress = (current_step_weight / 100) * x
        total_current_progress = current_progress + step_progress

        # Call external progress update if defined
        if original_update_func:
            if prev_progress <= total_current_progress:
                original_update_func(total_current_progress)
                prev_progress = total_current_progress

        # Move to next step if progress reaches 100%
        if x >= 100:
            current_progress += current_step_weight
            step_index = min(step_index + 1, len(steps) - 1)
            prev_progress = 0.0

        return total_current_progress

    return progress_update_func_block
