import asyncio
import pytest

from chatsky.core.script_function import ConstResponse, ConstDestination, ConstCondition, ConstPriority
from chatsky.core.script_function import BasePriority, BaseCondition, BaseResponse, BaseDestination, BaseProcessing
from chatsky.core.script_function import logger
from chatsky.core import Message, Pipeline, Context, Node, Transition
from chatsky.core.node_label import AbsoluteNodeLabel, NodeLabel


class TestBaseFunctionCallWrapper:
    @pytest.mark.parametrize(
        "func_type,data,return_value",
        [
            (BaseResponse, "text", Message(text="text")),
            (BaseCondition, False, False),
            (BaseDestination, ("flow", "node"), AbsoluteNodeLabel(flow_name="flow", node_name="node")),
            (BaseProcessing, None, None),
            (BasePriority, 1.0, 1.0),
        ],
    )
    async def test_validation(self, func_type, data, return_value):
        class MyFunc(func_type):
            async def call(self, ctx):
                return data

        assert await MyFunc().wrapped_call(None) == return_value

    async def test_wrong_type(self):
        class MyProc(BasePriority):
            async def call(self, ctx):
                return "w"

        assert isinstance(await MyProc().wrapped_call(None), TypeError)

    async def test_non_async_func(self):
        class MyCondition(BaseCondition):
            def call(self, ctx):
                return True

        assert await MyCondition().wrapped_call(None) is True

    async def test_catch_exception(self, log_event_catcher):
        log_list = log_event_catcher(logger)

        class MyProc(BaseProcessing):
            async def call(self, ctx):
                raise RuntimeError()

        assert isinstance(await MyProc().wrapped_call(None), RuntimeError)
        assert len(log_list) == 1
        assert log_list[0].levelname == "ERROR"

    async def test_base_exception_not_handled(self):
        class SpecialException(BaseException):
            pass

        class MyProc(BaseProcessing):
            async def call(self, ctx):
                raise SpecialException()

        with pytest.raises(SpecialException):
            await MyProc().wrapped_call(None)


@pytest.mark.parametrize(
    "func_type,data,root_value,return_value",
    [
        (ConstResponse, "response_text", Message(text="response_text"), Message(text="response_text")),
        (ConstResponse, {"text": "response_text"}, Message(text="response_text"), Message(text="response_text")),
        (ConstResponse, Message(text="response_text"), Message(text="response_text"), Message(text="response_text")),
        (
            ConstDestination,
            ("flow", "node"),
            NodeLabel(flow_name="flow", node_name="node"),
            AbsoluteNodeLabel(flow_name="flow", node_name="node"),
        ),
        (
            ConstDestination,
            NodeLabel(flow_name="flow", node_name="node"),
            NodeLabel(flow_name="flow", node_name="node"),
            AbsoluteNodeLabel(flow_name="flow", node_name="node"),
        ),
        (ConstPriority, 1.0, 1.0, 1.0),
        (ConstPriority, None, None, None),
        (ConstCondition, False, False, False),
    ],
)
async def test_const_functions(func_type, data, root_value, return_value):
    func = func_type.model_validate(data)
    assert func.root == root_value

    assert await func.wrapped_call(None) == return_value


class TestNodeLabelValidation:
    @pytest.fixture
    def pipeline(self):
        return Pipeline(script={"flow1": {"node": {}}, "flow2": {"node": {}}}, start_label=("flow1", "node"))

    @pytest.mark.parametrize("flow_name", ("flow1", "flow2"))
    async def test_const_destination(self, context_factory, flow_name):
        const_dst = ConstDestination.model_validate("node")

        dst = await const_dst.wrapped_call(context_factory(start_label=(flow_name, "node")))
        assert dst.flow_name == flow_name

    @pytest.mark.parametrize("flow_name", ("flow1", "flow2"))
    async def test_base_destination(self, context_factory, flow_name):
        class MyDestination(BaseDestination):
            def call(self, ctx):
                return "node"

        dst = await MyDestination().wrapped_call(context_factory(start_label=(flow_name, "node")))
        assert dst.flow_name == flow_name


def test_response_from_dict_validation():
    Node.model_validate({"response": {"msg": "text"}})


def test_destination_from_dict_validation():
    Transition.model_validate({"dst": {"flow_name": "flow", "node_name": "node"}})


async def test_const_object_immutability():
    message = Message(text="text1")
    response = ConstResponse.model_validate(message)

    response_result = await response.wrapped_call(Context())

    response_result.text = "text2"

    assert message.text == "text1"


class TestTimeout:
    class SleepingFunc(BaseProcessing):
        sleeping: float

        async def call(self, ctx: Context):
            await asyncio.sleep(self.sleeping)

    async def test_timeout(self):
        ctx = Context()
        sleep_func = self.SleepingFunc(timeout=0.1, sleeping=0.05)
        assert await sleep_func.wrapped_call(ctx) is None
