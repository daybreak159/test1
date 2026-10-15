from __future__ import annotations

import re
import string
from collections import Counter

from ..contracts import Grade
from .qa import AnswerGrader, exact_match


def locomo_f1(prediction, reference):
    """MemSkill local evaluation semantics: Porter stems; remove a/an/the/and."""
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    def tokens(text):
        text = "".join(c for c in str(text).lower() if c not in string.punctuation)
        text = re.sub(r"\b(a|an|the|and)\b", " ", text)
        return [stemmer.stem(token) for token in text.split()]

    predicted, expected = tokens(prediction), tokens(reference)
    count = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * count / (len(predicted) + len(expected)) if count else 0.0


def locomo_multihop_f1(prediction, reference):
    parts = str(reference).split(",")
    return sum(max(locomo_f1(p, r) for p in str(prediction).split(",")) for r in parts) / len(parts)


class LoCoMoGrader(AnswerGrader):
    def __call__(self, task, execution):
        if task.identity.dataset != "locomo":
            raise ValueError("LoCoMo grader cannot silently score another benchmark")
        if task.category == "5":
            return super().__call__(task, execution)
        aliases = self.labels[task.id]["answers"]
        f1 = locomo_multihop_f1 if task.category == "1" else locomo_f1
        metrics = {
            "f1": max(f1(execution.answer, answer) for answer in aliases),
            "em": max(exact_match(execution.answer, answer) for answer in aliases),
        }
        return Grade(metrics[self.quality_metric], metrics, self.version)
