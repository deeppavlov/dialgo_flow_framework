# %% [markdown]
"""
# 2. Partial slot extraction

This tutorial shows advanced options for slot extraction allowing
to extract only some of the slots.
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    RESPONSE,
    TRANSITIONS,
    PRE_RESPONSE,
    GLOBAL,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    processing as proc,
    responses as rsp,
)

from chatsky.slots import RegexpSlot, GroupSlot

from chatsky.utils.testing import (
    check_happy_path,
    is_interactive_mode,
)

# %% [markdown]
"""
## Extracted values

Result of successful slot extraction is the extracted value, *but*
if the extraction fails, the slot will be marked as "not extracted"
and its value will be set to the slot's `default_value` (`None` by default).

If group slot is being extracted, the extraction is considered successful
if and only if all child slots are successfully extracted.

## Success only extraction

The `Extract` function accepts `success_only` flag which makes it so
that extracted value is not saved unless extraction is successful.

This means that unsuccessfully trying to extract a slot will not overwrite
its previously extracted value.

Note that `success_only` is `True` by default.

## Partial group slot extraction

A group slot marked with `allow_partial_extraction` only saves values
of successfully extracted child slots.
Extracting such group slot is equivalent to extracting every child slot
with the `success_only` flag.

Partially extracted group slot is always considered successfully extracted
for the purposes of the `success_only` flag.

## Code explanation

In this example we showcase the behavior of
different group slot extraction settings:

Group `partial_extraction` is marked with `allow_partial_extraction`.
Any slot in this group is saved if and only if that slot is successfully
extracted.

Group `success_only_extraction` is extracted with the `success_only`
flag set to True.
Any slot in this group is saved if and only if all of the slots in the group
are successfully extracted within a single `Extract` call.

Group `success_only_false` is extracted with the `success_only` set to False.
Every slot in this group is saved (even if extraction was not successful).

Group `sub_slot_success_only_extraction` is extracted by passing all of its
child slots to the `Extract` method with the `success_only` flag set to True.
The behavior is equivalent to that of `partial_extraction`.
"""

# %%
sub_slots = {
    "date": RegexpSlot(
        regexp=r"(0?[1-9]|(?:1|2)[0-9]|3[0-1])[\.\/]"
        r"(0?[1-9]|1[0-2])[\.\/](\d{4}|\d{2})",
    ),
    "email": RegexpSlot(
        regexp=r"[\w\.-]+@[\w\.-]+\.\w{2,4}",
    ),
}

SLOTS = {
    "partial_extraction": GroupSlot(
        **sub_slots,
        allow_partial_extraction=True,
    ),
    "success_only_extraction": GroupSlot(
        **sub_slots,
    ),
    "success_only_false": GroupSlot(
        **sub_slots,
    ),
    "sub_slot_success_only_extraction": GroupSlot(
        **sub_slots,
    ),
}

script = {
    GLOBAL: {
        TRANSITIONS: [
            Tr(dst=("main", "start"), cnd=cnd.ExactMatch("/start")),
            Tr(dst=("main", "reset"), cnd=cnd.ExactMatch("/reset")),
            Tr(dst=("main", "print"), priority=0.5),
        ]
    },
    "main": {
        "start": {RESPONSE: "Hi! Send me email and date."},
        "reset": {
            PRE_RESPONSE: {"reset_slots": proc.UnsetAll()},
            RESPONSE: "All slots have been reset.",
        },
        "print": {
            PRE_RESPONSE: {
                "partial_extraction": proc.Extract("partial_extraction"),
                # partial extraction is always successful;
                # success_only doesn't matter
                "success_only_extraction": proc.Extract(
                    "success_only_extraction", success_only=True
                ),
                # success_only is True by default
                "success_only_false": proc.Extract(
                    "success_only_false", success_only=False
                ),
                "sub_slot_success_only_extraction": proc.Extract(
                    "sub_slot_success_only_extraction.email",
                    "sub_slot_success_only_extraction.date",
                    success_only=True,
                ),
            },
            RESPONSE: rsp.FilledTemplate(
                "Extracted slots:\n"
                "  Group with partial extraction:\n"
                "    {partial_extraction}\n"
                "  Group with success_only:\n"
                "    {success_only_extraction}\n"
                "  Group without success_only:\n"
                "    {success_only_false}\n"
                "  Extracting sub-slots with success_only:\n"
                "    {sub_slot_success_only_extraction}"
            ),
        },
    },
}

HAPPY_PATH = [
    ("/start", "Hi! Send me email and date."),
    (
        "Only email: email@email.com",
        "Extracted slots:\n"
        "  Group with partial extraction:\n"
        "    {'date': 'None', 'email': 'email@email.com'}\n"
        "  Group with success_only:\n"
        "    {'date': 'None', 'email': 'None'}\n"
        "  Group without success_only:\n"
        "    {'date': 'None', 'email': 'email@email.com'}\n"
        "  Extracting sub-slots with success_only:\n"
        "    {'date': 'None', 'email': 'email@email.com'}",
    ),
    (
        "Only date: 01.01.2024",
        "Extracted slots:\n"
        "  Group with partial extraction:\n"
        "    {'date': '01.01.2024', 'email': 'email@email.com'}\n"
        "  Group with success_only:\n"
        "    {'date': 'None', 'email': 'None'}\n"
        "  Group without success_only:\n"
        "    {'date': '01.01.2024', 'email': 'None'}\n"
        "  Extracting sub-slots with success_only:\n"
        "    {'date': '01.01.2024', 'email': 'email@email.com'}",
    ),
    (
        "Both email and date: another_email@email.com; 02.01.2024",
        "Extracted slots:\n"
        "  Group with partial extraction:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group with success_only:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group without success_only:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}\n"
        "  Extracting sub-slots with success_only:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}",
    ),
    (
        "Partial update (date only): 03.01.2024",
        "Extracted slots:\n"
        "  Group with partial extraction:\n"
        "    {'date': '03.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group with success_only:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group without success_only:\n"
        "    {'date': '03.01.2024', 'email': 'None'}\n"
        "  Extracting sub-slots with success_only:\n"
        "    {'date': '03.01.2024', 'email': 'another_email@email.com'}",
    ),
    (
        "No slots here but `Extract` will still be called.",
        "Extracted slots:\n"
        "  Group with partial extraction:\n"
        "    {'date': '03.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group with success_only:\n"
        "    {'date': '02.01.2024', 'email': 'another_email@email.com'}\n"
        "  Group without success_only:\n"
        "    {'date': 'None', 'email': 'None'}\n"
        "  Extracting sub-slots with success_only:\n"
        "    {'date': '03.01.2024', 'email': 'another_email@email.com'}",
    ),
]


# %%
pipeline = Pipeline(
    script=script,
    start_label=("main", "start"),
    slots=SLOTS,
)


# %%
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)

    if is_interactive_mode():
        pipeline.run()
