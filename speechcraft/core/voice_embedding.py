import os
import uuid
from io import BytesIO
from typing import Union

import numpy as np

from speechcraft.settings import EMBEDDINGS_DIR


class VoiceEmbedding:
    def __init__(
            self,
            codes: np.array,
            semantic_tokens: np.array,
            name: str = "new_speaker"
    ):
        self.name = name
        self.fine_prompt = codes
        self.coarse_prompt = codes[:2, :]
        self.semantic_prompt = semantic_tokens

    def save_to_speaker_lib(self):
        self.save(EMBEDDINGS_DIR)
        return self

    def save(self, folder: str):
        speaker_embedding_file = os.path.join(folder, f"{self.name}.npz")
        np.savez(speaker_embedding_file,
                 fine_prompt=self.fine_prompt,
                 coarse_prompt=self.coarse_prompt,
                 semantic_prompt=self.semantic_prompt
        )
        return speaker_embedding_file

    def to_bytes_io(self):
        f = BytesIO()
        np.savez(
           f, fine_prompt=self.fine_prompt, coarse_prompt=self.coarse_prompt,semantic_prompt=self.semantic_prompt
        )
        f.seek(0)
        return f

    @staticmethod
    def load(path_or_speaker_name: str):
        """
        Load a VoiceEmbedding from a file path or speaker nam.
        """
        if not os.path.exists(path_or_speaker_name):
            # add .npz extension
            if not path_or_speaker_name.endswith(".npz"):
                path_or_speaker_name = path_or_speaker_name + ".npz"

            # add default speaker dir
            path_or_speaker_name = os.path.join(EMBEDDINGS_DIR, path_or_speaker_name)


        if not os.path.exists(path_or_speaker_name):
            raise FileNotFoundError(f"Speaker embedding {path_or_speaker_name} not found")

        # Load data
        with np.load(path_or_speaker_name) as data:
            codes = data['fine_prompt']
            semantic_tokens = data['semantic_prompt']

        # return new VoiceEmbedding
        speaker_name = path_or_speaker_name.split("/")[-1].split(".")[0]
        return VoiceEmbedding(name=speaker_name, codes=codes, semantic_tokens=semantic_tokens)

    @staticmethod
    def load_from_bytes_io(bytes_io: BytesIO, speaker_name: str = None):
        with np.load(bytes_io) as data:
            codes = data['fine_prompt']
            semantic_tokens = data['semantic_prompt']

        if speaker_name is None or len(speaker_name) == 0:
            speaker_name = f"embedding_{uuid.uuid4()}"

        return VoiceEmbedding(name=speaker_name, codes=codes, semantic_tokens=semantic_tokens)

    @staticmethod
    def load_from_speaker_lib(speaker_name: str):
        return VoiceEmbedding.load(os.path.join(EMBEDDINGS_DIR, f"{speaker_name}.npz"))

    def __getitem__(self, key):
        """
        Support of ["semantic_tokens"] and ["fine_prompt"] and ["coarse_prompt"] syntax.
        :param key: str
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in VoiceEmbedding")

    def __setitem__(self, key, value):
        """
        Support of ["semantic_tokens"] and ["fine_prompt"] and ["coarse_prompt"] syntax.
        :param key: str
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key {key} not found in VoiceEmbedding")
