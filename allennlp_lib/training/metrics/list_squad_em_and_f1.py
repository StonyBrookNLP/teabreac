from typing import Tuple, List

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp_lib.tools.drop import get_metrics as drop_em_and_f1
from allennlp_lib.tools.drop import _normalize_answer


@Metric.register("list_squad")
class ListSquadEmAndF1(Metric):
    def __init__(self, keep_whitespace: bool = False) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._keep_whitespace = keep_whitespace  # False (default) for arxiv version.

    @overrides
    def __call__(self, predicted_texts: List[str], all_label_texts: List[List[str]]):
        # This needs to be called for each instance separately.
        # This is not a batched call.

        if predicted_texts:
            assert isinstance(predicted_texts[0], str)

        predicted_texts = [
            predicted_text
            for predicted_text in predicted_texts
            if predicted_text.strip()
        ]

        if all_label_texts:
            assert isinstance(all_label_texts[0], (list, tuple))
            if all_label_texts[0]:
                # Empty list is also allowed for eg. for intersection module.
                assert isinstance(all_label_texts[0][0], str)

        if self._keep_whitespace:
            merged_em = 0
        else:
            remove_space = lambda lst: [e.replace(" ", "") for e in lst]
            merged_em = max(
                [
                    int(
                        set(remove_space(predicted_texts))
                        == set(remove_space(label_texts))
                    )
                    for label_texts in all_label_texts
                ]
            )

        predicted_texts = list(set([_normalize_answer(e) for e in predicted_texts]))
        all_label_texts = [
            list(set([_normalize_answer(e) for e in label_texts]))
            for label_texts in all_label_texts
        ]

        em, f1 = max(
            [
                drop_em_and_f1(predicted_texts, label_texts)
                for label_texts in all_label_texts
            ],
            key=lambda e: e[0] + e[1],
        )

        em = max(em, merged_em)

        self._total_em += em
        self._total_f1 += f1
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"ans_em": exact_match, "ans_f1": f1_score}

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"SquadEmAndF1(em={self._total_em}, f1={self._total_f1})"
