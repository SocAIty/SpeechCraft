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
