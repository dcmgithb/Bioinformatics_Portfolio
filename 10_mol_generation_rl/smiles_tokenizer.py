"""
SMILES Tokenizer for Generative Models
========================================
Regex-based tokenization that correctly handles:
- Two-character elements (Cl, Br, Si, Se, etc.)
- Ring closures (%, ring numbers up to %99)
- Stereochemistry (@, @@, /, \\)
- Charge notation ([NH+], [O-], [nH], etc.)
- Aromatic atoms (b, c, n, o, s, p)
- Branch notation ((, ))
- Bond types (=, #, -, :, ~)

Usage
-----
tokenizer = SmilesTokenizer()
tokenizer.build_vocab(smiles_list)
enc = tokenizer.encode("CC(=O)Oc1ccccc1C(=O)O")   # List[int]
smi = tokenizer.decode(enc)                         # str
batch = tokenizer.batch_encode(smiles_list)         # np.ndarray [B, L]
"""

import re
import json
import numpy as np
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Canonical regex for SMILES tokenisation.
# Order matters: longer patterns must come before shorter ones.
# ---------------------------------------------------------------------------
SMILES_REGEX = (
    r"(\[[^\]]+]"          # bracketed atoms [NH+], [2H], [C@@H], etc.
    r"|Br?"                # Br or B
    r"|Cl?"                # Cl or C
    r"|Si|Se|Sn"           # other two-char elements starting with S
    r"|N|O|S|P|F|I"        # single-char organic subset
    r"|b|c|n|o|s|p"        # aromatic atoms (lowercase)
    r"|\(|\)"              # branch open/close
    r"|\.|\=|\#|\-|\+"     # disconnected structure, bond types
    r"|\\|\/|:"            # stereochemistry bonds, aromatic bond
    r"|~|@|\?|>"           # misc (reaction arrow, query)
    r"|\*|\$"              # any-atom wildcards
    r"|\%[0-9]{2}"         # two-digit ring closures %10–%99
    r"|[0-9])"             # single-digit ring closures 1–9
)

_COMPILED_REGEX = re.compile(SMILES_REGEX)

# Special token strings
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SmilesTokenizer:
    """Regex-based SMILES tokenizer with vocabulary management.

    Parameters
    ----------
    max_length : int
        Maximum sequence length (including BOS and EOS tokens).
        Sequences longer than this are excluded during batch encoding.
    """

    def __init__(self, max_length: int = 128) -> None:
        self._max_length = max_length

        # Special token IDs (always indices 0-3)
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN

        # Vocab mappings (populated by build_vocab or load)
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._vocab_size: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return self._vocab_size

    @property
    def max_length(self) -> int:
        """Maximum encoded sequence length."""
        return self._max_length

    @property
    def pad_id(self) -> int:
        return self._token2id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token2id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token2id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token2id[UNK_TOKEN]

    # ------------------------------------------------------------------
    # Core tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(smiles: str) -> List[str]:
        """Split a SMILES string into a list of token strings.

        Parameters
        ----------
        smiles : str
            A valid (or candidate) SMILES string.

        Returns
        -------
        List[str]
            Ordered list of SMILES tokens, e.g. ['C', 'C', '(', '=', 'O', ')'].
        """
        return _COMPILED_REGEX.findall(smiles)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocab(self, smiles_list: List[str]) -> "SmilesTokenizer":
        """Fit vocabulary from a list of SMILES strings.

        Inserts special tokens at indices 0–3, then adds atom/bond tokens
        in the order they are first encountered.

        Parameters
        ----------
        smiles_list : List[str]
            Training corpus of SMILES strings.

        Returns
        -------
        self : SmilesTokenizer
            Returns self to allow method chaining.
        """
        seen = []
        seen_set: set = set()
        for smi in smiles_list:
            for tok in self.tokenize(smi):
                if tok not in seen_set:
                    seen.append(tok)
                    seen_set.add(tok)

        all_tokens = SPECIAL_TOKENS + seen
        self._token2id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self._id2token = {idx: tok for tok, idx in self._token2id.items()}
        self._vocab_size = len(all_tokens)
        return self

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a SMILES string to a list of token IDs.

        Parameters
        ----------
        smiles : str
            SMILES string to encode.
        add_special_tokens : bool
            If True, prepend BOS and append EOS tokens.

        Returns
        -------
        List[int]
            Token ID sequence.
        """
        tokens = self.tokenize(smiles)
        ids = [self._token2id.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a SMILES string.

        Parameters
        ----------
        token_ids : List[int]
            Sequence of token IDs to decode.
        skip_special_tokens : bool
            If True, omit PAD, BOS, EOS, UNK from output.

        Returns
        -------
        str
            Reconstructed SMILES string.
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        parts = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self._id2token.get(tid, UNK_TOKEN)
            parts.append(tok)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        smiles_list: List[str],
        max_len: Optional[int] = None,
        pad: bool = True,
    ) -> np.ndarray:
        """Encode a list of SMILES strings to a padded integer matrix.

        SMILES whose encoded length (with BOS + EOS) exceeds *max_len* are
        silently truncated to *max_len* tokens.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES to encode.
        max_len : int, optional
            Maximum sequence length. Defaults to self.max_length.
        pad : bool
            Whether to right-pad with PAD tokens.

        Returns
        -------
        np.ndarray of shape [B, L]
            Integer token ID matrix.
        """
        if max_len is None:
            max_len = self._max_length

        encoded = [self.encode(smi)[:max_len] for smi in smiles_list]

        if not pad:
            return np.array(encoded, dtype=object)

        matrix = np.full((len(encoded), max_len), self.pad_id, dtype=np.int64)
        for i, seq in enumerate(encoded):
            matrix[i, : len(seq)] = seq
        return matrix

    def batch_decode(self, token_matrix: np.ndarray, skip_special_tokens: bool = True) -> List[str]:
        """Decode a 2-D token ID matrix to a list of SMILES strings.

        Parameters
        ----------
        token_matrix : np.ndarray of shape [B, L]
        skip_special_tokens : bool

        Returns
        -------
        List[str]
        """
        return [
            self.decode(row.tolist(), skip_special_tokens=skip_special_tokens)
            for row in token_matrix
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise tokenizer state to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. 'tokenizer.json').
        """
        state = {
            "max_length": self._max_length,
            "token2id": self._token2id,
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        """Load a serialised tokenizer from disk.

        Parameters
        ----------
        path : str
            Path to a JSON file previously saved by :meth:`save`.

        Returns
        -------
        SmilesTokenizer
        """
        with open(path) as fh:
            state = json.load(fh)

        obj = cls(max_length=state["max_length"])
        obj._token2id = state["token2id"]
        obj._id2token = {int(v): k for k, v in state["token2id"].items()}
        obj._vocab_size = len(state["token2id"])
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SmilesTokenizer(vocab_size={self._vocab_size}, "
            f"max_length={self._max_length})"
        )

    def vocabulary(self) -> List[str]:
        """Return the full ordered vocabulary list."""
        return [self._id2token[i] for i in range(self._vocab_size)]
