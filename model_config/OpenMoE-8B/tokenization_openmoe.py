from transformers import T5Tokenizer
from typing import List, Optional, Tuple, Union

class OpenMoeTokenizer(T5Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_side = 'left'
        self.add_bos_token = True
        self.add_eos_token = False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.add_eos_token:
            token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if self.add_bos_token:
            token_ids_0 = [self.pad_token_id] + token_ids_0
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1