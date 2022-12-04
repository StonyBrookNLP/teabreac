import re
import types
from typing import List

from transformers import AutoTokenizer


def t5_fix_output_spacing(text: str) -> str:
    # This is mostly taken from NT5's codebase.
    text = re.sub(r" +", " ", text).strip()
    match = re.compile(r"([a-z]|,|-)(\d)")
    text = re.sub(match, r"\1 \2", text)
    match = re.compile(r"(\d|[a-z])( )?(-)( )?(\d|[a-z])")
    text = re.sub(match, r"\1\3\5", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def t5_convert_tokens_to_string(self, tokens: List[str]) -> str:
    """Converts a sequence of tokens (string) in a single string."""
    current_sub_tokens = []
    out_string = ""
    for token in tokens:
        # make sure that special tokens are not decoded using sentencepiece model
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        if token in self.all_special_tokens and token not in digits:
            out_string += self.sp_model.decode_pieces(current_sub_tokens) + token + " "
            current_sub_tokens = []
        else:
            current_sub_tokens.append(token)
    out_string += self.sp_model.decode_pieces(current_sub_tokens)
    out_string = t5_fix_output_spacing(out_string)
    return out_string.strip()


def bart_convert_tokens_to_string(self, tokens: List[str]) -> str:
    """Converts a sequence of tokens (string) in a single string."""
    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    tokens = [
        token + "Ġ"
        if token in self.all_special_tokens and token not in digits  # Ġ is for space
        else token
        for token in tokens
    ]
    text = "".join(tokens)
    text = bytearray([self.byte_decoder[c] for c in text]).decode(
        "utf-8", errors=self.errors
    )
    return text


def fix_digit_spacing(fixed_decoded_text: str) -> str:
    # TODO: This is only temporary.
    ### For bart:
    fixed_decoded_text = re.sub(r"(\d)(\D)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" ,(\d\d\d)(\D)", r",\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" ,(\d\d\d)\b", r",\1", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) ,(\d)", r"\1, \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) , ", r"\1, ", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" +", " ", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\D),(\d)", r"\1 \2", fixed_decoded_text)

    ### For bart + t5:
    fixed_decoded_text = re.sub(r", (\d\d\d)(\D)", r",\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r", (\d\d\d)\b", r",\1", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) +([-/.:%;-])", r"\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"([-/:%;-]) +(\d)", r"\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(
        r"(\d) (st|nd|rd|th|ers|°C)", r"\1\2", fixed_decoded_text
    )
    fixed_decoded_text = re.sub(r"(\d)(thousand|million)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\([\d,.]+) +\)", r"\1)", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d+) s\b", r"\1s", fixed_decoded_text)
    return fixed_decoded_text


def fix_decoded_text(self, decoded_text: str) -> str:
    fixed_decoded_text = decoded_text
    for special_token in self.added_tokens_decoder.values():
        fixed_decoded_text = fixed_decoded_text.replace(
            special_token, " " + special_token + " "
        )
    if self._bos_token:
        fixed_decoded_text = fixed_decoded_text.replace(self.bos_token, "")
    fixed_decoded_text = fixed_decoded_text.replace(self.eos_token, "")
    fixed_decoded_text = fixed_decoded_text.replace(self.pad_token, "")
    fixed_decoded_text = " ".join(re.split(r" +", fixed_decoded_text)).strip()
    fixed_decoded_text = re.sub(r" +", " ", fixed_decoded_text).strip()

    ### For bart:
    fixed_decoded_text = re.sub(r"(\d)(\D)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" ,(\d\d\d)(\D)", r",\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" ,(\d\d\d)\b", r",\1", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) ,(\d)", r"\1, \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) , ", r"\1, ", fixed_decoded_text)
    fixed_decoded_text = re.sub(r" +", " ", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\D),(\d)", r"\1 \2", fixed_decoded_text)

    ### For bart + t5:
    fixed_decoded_text = re.sub(r", (\d\d\d)(\D)", r",\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r", (\d\d\d)\b", r",\1", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d) +([-/.:%;-])", r"\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"([-/:%;-]) +(\d)", r"\1\2", fixed_decoded_text)
    fixed_decoded_text = re.sub(
        r"(\d) (st|nd|rd|th|ers|°C)", r"\1\2", fixed_decoded_text
    )
    fixed_decoded_text = re.sub(r"(\d)(thousand|million)", r"\1 \2", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\([\d,.]+) +\)", r"\1)", fixed_decoded_text)
    fixed_decoded_text = re.sub(r"(\d+) s\b", r"\1s", fixed_decoded_text)
    return fixed_decoded_text


def enable_digit_tokenization(tokenizer: AutoTokenizer) -> None:

    if tokenizer.is_fast:
        raise Exception("The monkey patch only works for the slow tokenizer currently.")

    is_t5_based = "T5Tokenizer" in str(tokenizer.__class__)
    is_bart_based = "BartTokenizer" in str(tokenizer.__class__)

    if not is_t5_based and not is_bart_based:
        raise Exception(
            f"The monkeypath only works for T5 and Bart based tokenizers currently. "
            f"Found tokenizer of class: {tokenizer.__class__}"
        )

    if is_t5_based:
        tokenizer.convert_tokens_to_string = types.MethodType(
            t5_convert_tokens_to_string, tokenizer
        )
    elif is_bart_based:
        tokenizer.convert_tokens_to_string = types.MethodType(
            bart_convert_tokens_to_string, tokenizer
        )

    tokenizer.fix_decoded_text = types.MethodType(fix_decoded_text, tokenizer)
