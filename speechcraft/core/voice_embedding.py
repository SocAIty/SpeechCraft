import os
import uuid
from io import BytesIO
from typing import Union, Optional

import numpy as np

from speechcraft.settings import EMBEDDINGS_DIR, DEFAULT_EMBEDDINGS_DIR


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
        sp = getattr(EMBEDDINGS_DIR, DEFAULT_EMBEDDINGS_DIR, "")
        self.save(sp)
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
    def _load_embedding_data(source: Union[str, BytesIO]) -> tuple:
        """
        Internal method to load embedding data from a file or BytesIO object.

        Args:
            source: File path or BytesIO object containing the embedding data

        Returns:
            tuple: (codes, semantic_tokens)

        Raises:
            ValueError: If the embedding data is invalid or cannot be loaded
        """
        try:
            with np.load(source) as data:
                return data['fine_prompt'], data['semantic_prompt']
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid embedding format. Error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load embedding. Error: {str(e)}")

    @staticmethod
    def _find_embedding_file(path_or_name: str) -> Optional[str]:
        """
        Search for embedding file in multiple locations.

        Args:
            path_or_name: File path or speaker name

        Returns:
            str or None: Found file path or None if not found
        """
        # Try direct path first
        if os.path.isfile(path_or_name):
            return path_or_name

        filename = os.path.basename(path_or_name)
        search_paths = []

        # Add EMBEDDINGS_DIR to search paths if it exists and is valid
        if EMBEDDINGS_DIR and os.path.isdir(EMBEDDINGS_DIR):
            search_paths.append(EMBEDDINGS_DIR)

        # Always include DEFAULT_EMBEDDINGS_DIR
        search_paths.append(DEFAULT_EMBEDDINGS_DIR)

        # Search in all paths
        for directory in search_paths:
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                return filepath

        return None

    @classmethod
    def load(cls, source: Union[str, BytesIO], speaker_name: Optional[str] = None) -> "VoiceEmbedding":
        """
        Unified method to load a VoiceEmbedding from various sources.

        Args:
            source: Can be one of:
                - Full path to .npz file (absolute or relative)
                - Speaker name
                - BytesIO object containing embedding data
            speaker_name: Optional speaker name. If not provided:
                - For file paths: extracted from filename
                - For BytesIO: generated using UUID

        Returns:
            VoiceEmbedding: Loaded voice embedding object

        Raises:
            FileNotFoundError: If embedding file cannot be found
            ValueError: If embedding data has invalid format
        """
        if isinstance(source, BytesIO):
            # Handle BytesIO input
            codes, semantic_tokens = cls._load_embedding_data(source)
            final_speaker_name = speaker_name or f"embedding_{uuid.uuid4()}"
        else:
            # Handle string input (file path or speaker name)
            # Add .npz extension if not present
            source = f"{source}.npz" if not source.endswith('.npz') else source
            filepath = cls._find_embedding_file(source)

            if not filepath:
                searched_paths = [
                    source,
                    os.path.join(EMBEDDINGS_DIR, os.path.basename(source)) if EMBEDDINGS_DIR else None,
                    os.path.join(DEFAULT_EMBEDDINGS_DIR, os.path.basename(source))
                ]
                searched_paths = [p for p in searched_paths if p]  # Remove None values
                raise FileNotFoundError(
                    f"Voice embedding not found. Searched in:\n  - " +
                    "\n  - ".join(searched_paths)
                )

            codes, semantic_tokens = cls._load_embedding_data(filepath)
            final_speaker_name = speaker_name or os.path.splitext(os.path.basename(filepath))[0]

        return cls(
            name=final_speaker_name,
            codes=codes,
            semantic_tokens=semantic_tokens
        )

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
