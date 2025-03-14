from ..activation_dataset import ActivationDataset, ProcessInfo
from .activation_transform import Activation
from .match_transform import Match
from .token_transform import Tokenize, deserialize_tokenization


class OpenMathDataset(ActivationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: move this to a unit test? or maybe utility file?
    def sanity_check_text(self, match, should_print=True):
        token_row = self.token[match["token_row_id"]]
        decoded_match = self.tokenizer.decode(
            deserialize_tokenization(token_row["serialized_tokenization"]).ids[
                slice(*match["match_token_range"])
            ]
        )
        if should_print:
            print("Token row id:", match["token_row_id"])
            print("Match text:", match["match_text"])
            print("Match context: ", match["context"])
            print("Context in prefill", match["context"] in token_row["prefill"])
            print("Match token range:", match["match_token_range"])
            print("Decoded match:", decoded_match)
            print(
                "Match text equals decoded match:",
                match["match_text"] == decoded_match,
            )
        return match["match_text"] == decoded_match

    def _process_token(self) -> ProcessInfo:
        return ProcessInfo(
            self.raw,
            Tokenize(self.tokenizer),
        )

    def _process_match(self) -> ProcessInfo:
        return ProcessInfo(
            self.token,
            Match(),
        )

    def _process_activation(self):
        # TODO: manually set up our own batches so that we are running *all* of the examples
        # for each of N prefills
        return ProcessInfo(
            self.match,
            Activation(self.model, self.tokenizer, self.token),
        )

    def _process_classification(self):
        return None
