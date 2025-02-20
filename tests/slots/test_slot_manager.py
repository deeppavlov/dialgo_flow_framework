import pytest

from chatsky.slots.slots import (
    SlotManager,
    RegexpSlot,
    GroupSlot,
    FunctionSlot,
    ExtractedGroupSlot,
    ExtractedValueSlot,
    SlotNotExtracted,
)
from chatsky.core import Message, Context


def faulty_func(_):
    raise SlotNotExtracted("Error.")


init_value_slot = ExtractedValueSlot.model_construct(
    is_slot_extracted=False,
    extracted_value=SlotNotExtracted("Initial slot extraction."),
    default_value=None,
)


root_slot = GroupSlot(
    person=GroupSlot(
        name=RegexpSlot(regexp=r"(?<=am ).+?(?=\.)"),
        surname=FunctionSlot(func=faulty_func),
        email=RegexpSlot(regexp=r"[a-zA-Z\.]+@[a-zA-Z\.]+"),
    ),
    msg_len=FunctionSlot(func=lambda msg: len(msg.text)),
)


extracted_slot_values = {
    "person.name": ExtractedValueSlot.model_construct(
        is_slot_extracted=True, extracted_value="Bot", default_value=None
    ),
    "person.surname": ExtractedValueSlot.model_construct(
        is_slot_extracted=False, extracted_value=SlotNotExtracted("Error."), default_value=None
    ),
    "person.email": ExtractedValueSlot.model_construct(
        is_slot_extracted=True, extracted_value="bot@bot", default_value=None
    ),
    "msg_len": ExtractedValueSlot.model_construct(is_slot_extracted=True, extracted_value=29, default_value=None),
}


extracted_slot_values["person"] = ExtractedGroupSlot(
    name=extracted_slot_values["person.name"],
    surname=extracted_slot_values["person.surname"],
    email=extracted_slot_values["person.email"],
)


unset_slot = ExtractedValueSlot.model_construct(
    is_slot_extracted=False, extracted_value=SlotNotExtracted("Slot manually unset."), default_value=None
)


init_slot_storage = ExtractedGroupSlot(
    person=ExtractedGroupSlot(
        name=init_value_slot,
        surname=init_value_slot,
        email=init_value_slot,
    ),
    msg_len=init_value_slot,
)


unset_slot_storage = ExtractedGroupSlot(
    person=ExtractedGroupSlot(
        name=unset_slot,
        surname=unset_slot,
        email=unset_slot,
    ),
    msg_len=unset_slot,
)


full_slot_storage = ExtractedGroupSlot(
    person=ExtractedGroupSlot(
        name=extracted_slot_values["person.name"],
        surname=extracted_slot_values["person.surname"],
        email=extracted_slot_values["person.email"],
    ),
    msg_len=extracted_slot_values["msg_len"],
)


@pytest.fixture(scope="function")
def context_with_request(context):
    new_ctx = context.model_copy(deep=True)
    new_ctx.requests[2] = Message(text="I am Bot. My email is bot@bot")
    return new_ctx


async def test_init_slot_storage():
    assert root_slot.init_value() == init_slot_storage


@pytest.fixture(scope="function")
def empty_slot_manager():
    manager = SlotManager()
    manager.set_root_slot(root_slot)
    return manager


@pytest.fixture(scope="function")
def extracted_slot_manager():
    slot_storage = full_slot_storage.model_copy(deep=True)
    return SlotManager(root_slot=root_slot, slot_storage=slot_storage)


@pytest.fixture(scope="function")
def fully_extracted_slot_manager():
    slot_storage = full_slot_storage.model_copy(deep=True)
    slot_storage.person.surname = ExtractedValueSlot.model_construct(
        extracted_value="Bot", is_slot_extracted=True, default_value=None
    )
    return SlotManager(root_slot=root_slot, slot_storage=slot_storage)


def test_get_slot_by_name(empty_slot_manager):
    assert empty_slot_manager.get_slot("person.name").regexp == r"(?<=am ).+?(?=\.)"
    assert empty_slot_manager.get_slot("person.email").regexp == r"[a-zA-Z\.]+@[a-zA-Z\.]+"
    assert isinstance(empty_slot_manager.get_slot("person"), GroupSlot)
    assert isinstance(empty_slot_manager.get_slot("msg_len"), FunctionSlot)

    with pytest.raises(KeyError):
        empty_slot_manager.get_slot("person.birthday")

    with pytest.raises(KeyError):
        empty_slot_manager.get_slot("intent")


@pytest.mark.parametrize(
    "slot_name,expected_slot_storage",
    [
        (
            "person.name",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=extracted_slot_values["person.name"],
                    surname=init_value_slot,
                    email=init_value_slot,
                ),
                msg_len=init_value_slot,
            ),
        ),
        (
            "person",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=extracted_slot_values["person.name"],
                    surname=extracted_slot_values["person.surname"],
                    email=extracted_slot_values["person.email"],
                ),
                msg_len=init_value_slot,
            ),
        ),
        (
            "msg_len",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=init_value_slot,
                    surname=init_value_slot,
                    email=init_value_slot,
                ),
                msg_len=extracted_slot_values["msg_len"],
            ),
        ),
    ],
)
async def test_slot_extraction(slot_name, expected_slot_storage, empty_slot_manager, context_with_request):
    await empty_slot_manager.extract_slot(slot_name, context_with_request, success_only=False)
    assert empty_slot_manager.slot_storage == expected_slot_storage


@pytest.mark.parametrize(
    "slot_name,expected_slot_storage",
    [
        (
            "person.name",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=extracted_slot_values["person.name"],
                    surname=init_value_slot,
                    email=init_value_slot,
                ),
                msg_len=init_value_slot,
            ),
        ),
        (
            "person.surname",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=init_value_slot,
                    surname=init_value_slot,
                    email=init_value_slot,
                ),
                msg_len=init_value_slot,
            ),
        ),
    ],
)
async def test_successful_extraction(slot_name, expected_slot_storage, empty_slot_manager, context_with_request):
    await empty_slot_manager.extract_slot(slot_name, context_with_request, success_only=True)
    assert empty_slot_manager.slot_storage == expected_slot_storage


async def test_extract_all(empty_slot_manager, context_with_request):
    await empty_slot_manager.extract_all(context_with_request)
    assert empty_slot_manager.slot_storage == full_slot_storage


@pytest.mark.parametrize(
    "slot_name, expected_slot_storage",
    [
        (
            "person.name",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=unset_slot,
                    surname=extracted_slot_values["person.surname"],
                    email=extracted_slot_values["person.email"],
                ),
                msg_len=extracted_slot_values["msg_len"],
            ),
        ),
        (
            "person",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=unset_slot,
                    surname=unset_slot,
                    email=unset_slot,
                ),
                msg_len=extracted_slot_values["msg_len"],
            ),
        ),
        (
            "msg_len",
            ExtractedGroupSlot(
                person=ExtractedGroupSlot(
                    name=extracted_slot_values["person.name"],
                    surname=extracted_slot_values["person.surname"],
                    email=extracted_slot_values["person.email"],
                ),
                msg_len=unset_slot,
            ),
        ),
    ],
)
def test_unset_slot(extracted_slot_manager, slot_name, expected_slot_storage):
    extracted_slot_manager.unset_slot(slot_name)
    assert extracted_slot_manager.slot_storage == expected_slot_storage


def test_unset_all(extracted_slot_manager):
    extracted_slot_manager.unset_all_slots()
    assert extracted_slot_manager.slot_storage == unset_slot_storage


@pytest.mark.parametrize("slot_name", ["person.name", "person", "msg_len"])
def test_get_extracted_slot(extracted_slot_manager, slot_name):
    assert extracted_slot_manager.get_extracted_slot(slot_name) == extracted_slot_values[slot_name]


def test_get_extracted_slot_raises(extracted_slot_manager):
    with pytest.raises(KeyError):
        extracted_slot_manager.get_extracted_slot("none.none")

    with pytest.raises(KeyError):
        extracted_slot_manager.get_extracted_slot("person.none")

    with pytest.raises(KeyError):
        extracted_slot_manager.get_extracted_slot("person.name.none")

    with pytest.raises(KeyError):
        extracted_slot_manager.get_extracted_slot("none")


def test_slot_extracted(fully_extracted_slot_manager, empty_slot_manager):
    assert fully_extracted_slot_manager.is_slot_extracted("person.name") is True
    assert fully_extracted_slot_manager.is_slot_extracted("person") is True
    with pytest.raises(KeyError):
        fully_extracted_slot_manager.is_slot_extracted("none")
    assert fully_extracted_slot_manager.all_slots_extracted() is True

    assert empty_slot_manager.is_slot_extracted("person.name") is False
    assert empty_slot_manager.is_slot_extracted("person") is False
    with pytest.raises(KeyError):
        empty_slot_manager.is_slot_extracted("none")
    assert empty_slot_manager.all_slots_extracted() is False


@pytest.mark.parametrize(
    "template,filled_value",
    [
        (
            "Your name is {person.name} {person.surname}, your email: {person.email}.",
            "Your name is Bot None, your email: bot@bot.",
        ),
    ],
)
def test_template_filling(extracted_slot_manager, template, filled_value):
    assert extracted_slot_manager.fill_template(template) == filled_value


def test_serializable():
    serialized = full_slot_storage.model_dump_json()
    assert full_slot_storage == ExtractedGroupSlot.model_validate_json(serialized)


async def test_old_slot_storage_update():
    ctx = Context.model_validate({"requests": {"items": {0: Message(text="text")}}})

    slot1 = FunctionSlot(func=lambda msg: len(msg.text) + 2, default_value="1")
    init_slot1 = slot1.init_value()
    extracted_value1 = await slot1.get_value(ctx)
    assert extracted_value1.value == 6

    slot2 = FunctionSlot(func=lambda msg: len(msg.text) + 3, default_value="2")
    init_slot2 = slot2.init_value()
    extracted_value2 = await slot2.get_value(ctx)
    assert extracted_value2.value == 7

    old_group_slot = GroupSlot.model_validate(
        {
            "0": {"0": slot1, "1": slot2},
            "1": {"0": slot1, "1": slot2},
            "2": {"0": slot1, "1": slot2},
            "3": slot1,
            "4": slot1,
            "5": slot1,
        }
    )

    manager = SlotManager()
    manager.set_root_slot(old_group_slot)

    assert manager.slot_storage == ExtractedGroupSlot.model_validate(
        {
            "0": {"0": init_slot1, "1": init_slot2},
            "1": {"0": init_slot1, "1": init_slot2},
            "2": {"0": init_slot1, "1": init_slot2},
            "3": init_slot1,
            "4": init_slot1,
            "5": init_slot1,
        }
    )

    await manager.extract_all(ctx)
    assert manager.slot_storage == ExtractedGroupSlot.model_validate(
        {
            "0": {"0": extracted_value1, "1": extracted_value2},
            "1": {"0": extracted_value1, "1": extracted_value2},
            "2": {"0": extracted_value1, "1": extracted_value2},
            "3": extracted_value1,
            "4": extracted_value1,
            "5": extracted_value1,
        }
    )

    new_group_slot = GroupSlot.model_validate(
        {
            "-1": {"0": slot1, "2": slot2},  # added
            "0": {"0": slot1, "2": slot2},
            "1": slot2,  # type changed
            # "2" -- removed
            "3": slot2,
            "4": {"0": slot1, "2": slot2},  # type changed
            # "5" -- removed
            "6": slot2,  # added
        }
    )

    manager.set_root_slot(new_group_slot)

    assert manager.slot_storage == ExtractedGroupSlot.model_validate(
        {
            "-1": {"0": init_slot1, "2": init_slot2},
            "0": {"0": extracted_value1, "2": init_slot2},
            "1": init_slot2,
            "3": extracted_value1,
            "4": {"0": init_slot1, "2": init_slot2},
            "6": init_slot2,
        }
    )
