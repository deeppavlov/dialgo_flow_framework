"""
Conditions
------------

This module provides condition functions for annotation processing.
"""

from typing import Optional, List

try:
    # !!! remove sklearn, use pure python instead
    from sklearn.metrics.pairwise import cosine_similarity

    sklearn_available = True
except ImportError:
    sklearn_available = False
from chatsky import Context
from chatsky.conditions.standard import BaseCondition
from chatsky.ml.models.base_model import ExtrasBaseAPIModel


class HasLabel(BaseCondition):
    """
    Use this condition, when you need to check, whether the probability
    of a particular label for the last annotated user utterance surpasses the threshold.

    :param label: String name or a reference to a DatasetItem object, or a collection thereof.
    :param namespace: Namespace key of a particular model that should detect the dataset_item.
        If not set, all namespaces will be searched for the required dataset_item.
    :param threshold: The minimal label probability that triggers a positive response
        from the function.
    """

    label: str
    model: ExtrasBaseAPIModel
    threshold: float = 0.9

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    async def call(self, ctx: Context) -> bool:
        # Predict labels for the last request
        # and store them in framework_data with uuid of the model as a key
        await self.model(ctx)
        if self.model.model_id not in ctx.framework_data.models_labels:
            return False
        if self.model.model_id is not None:
            return ctx.framework_data.models_labels.get(self.model.model_id, {}).get(self.label, 0) >= self.threshold
        scores = [item.get(self.label, 0) for item in ctx.framework_data.models_labels.values()]
        comparison_array = [item >= self.threshold for item in scores]
        return any(comparison_array)


class HasMatch(BaseCondition):
    """
    Use this condition, if you need to check whether the last request matches
    any of the pre-defined intent utterances.
    The model passed to this function should be in the fit state.

    :param model: Any model from the :py:mod:`~chatsky.ml.models.local.cosine_matchers` module.
    :param positive_examples: Utterances that the request should match.
    :param negative_examples: Utterances that the request should not match.
    :param threshold: Similarity threshold that triggers a positive response from the function.
    """

    model: ExtrasBaseAPIModel
    positive_examples: Optional[List[str]]
    negative_examples: Optional[List[str]] = []
    threshold: float = 0.9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def call(self, ctx: Context) -> bool:
        if not (ctx.last_request and ctx.last_request.text):
            return False
        input_vector = self.model.transform(ctx.last_request.text)
        positive_vectors = [self.model.transform(item) for item in self.positive_examples]
        negative_vectors = [self.model.transform(item) for item in self.negative_examples]
        positive_sims = [cosine_similarity(input_vector, item)[0][0] for item in positive_vectors]
        negative_sims = [cosine_similarity(input_vector, item)[0][0] for item in negative_vectors]
        max_pos_sim = max(positive_sims)
        max_neg_sim = 0 if len(negative_sims) == 0 else max(negative_sims)
        return bool(max_pos_sim > self.threshold > max_neg_sim)
