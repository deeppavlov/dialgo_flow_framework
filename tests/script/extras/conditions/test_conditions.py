import pytest
from chatsky import Context, Message
from chatsky.ml.utils import LABEL_KEY
from chatsky.ml.dataset import DatasetItem, Dataset
from chatsky.conditions.ml import has_cls_label, has_match
from chatsky.ml.models.base_model import ExtrasBaseModel


class DummyModel(ExtrasBaseModel):
    def __init__(self, model_id=None):
        self.model_id = model_id

    def predict(self, text):
        return {"label_a": 0.1, "label_b": 0.9}

    def __call__(self, text):
        pass


@pytest.mark.parametrize(
    ["input"],
    [
        ("a",),
        (DatasetItem(label="a", samples=["a"]),),
        (["a", "b"],),
        ([DatasetItem(label="a", samples=["a"]), DatasetItem(label="b", samples=["b"])],),
    ],
)
def test_conditions(input, testing_pipeline):
    ctx = Context(framework_states={LABEL_KEY: {"model_a": {"a": 1, "b": 1}, "model_b": {"b": 1, "c": 1}}})
    ctx.add_request(Message(text="idk something"))
    model = DummyModel(model_id="model_a")
    assert has_cls_label(input, model)(ctx, testing_pipeline) is True
    assert has_cls_label(input, model, namespace="model_a")(ctx, testing_pipeline) is True
    assert has_cls_label(input, model, threshold=1.1)(ctx, testing_pipeline) is False
    ctx2 = Context()
    assert has_cls_label(input, model)(ctx2, testing_pipeline) is False


@pytest.mark.parametrize(["input"], [(1,), (3.3,), ({"a", "b"},)])
def test_conds_invalid(input, testing_pipeline):
    with pytest.raises(NotImplementedError):
        model = DummyModel(model_id="model_a")
        _ = has_cls_label(model, input)(Context(), testing_pipeline)


# since the standart model is no longer exist (because we do not support local models) mock-model should be created)
# @pytest.mark.parametrize(
#     ["_input", "last_request", "thresh"],
#     [
#         ({"positive_examples": ["like sweets", "like candy"], "negative_examples": ["other stuff"]}, "sweets", 0.7),
#         (
#             {
#                 "positive_examples": ["good stuff", "brilliant stuff", "excellent stuff"],
#                 "negative_examples": ["negative example"],
#             },
#             "excellent, brilliant",
#             0.5,
#         ),
#     ],
# )
# def test_has_match(_input: dict, testing_pipeline, thresh, standard_model, last_request):
#     ctx = Context()
#     ctx.add_request(Message(text=last_request))
#     # Per default, we assume that the model has already been fit.
#     # For this test case we fit it manually.
#     collection = Dataset(
#         items=[DatasetItem.model_validate({"label": key, "samples": values}) for key, values in _input.items()]
#     )
#     standard_model.fit(collection)
#     result = has_match(standard_model, threshold=thresh, **_input)(ctx, testing_pipeline)
#     assert result
