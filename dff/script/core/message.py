"""
Message
-------
The :py:class:`.Message` class is a universal data model for representing a message that should be supported by
DFF. It only contains types and properties that are compatible with most messaging services.
"""

from typing import Literal, Optional, List, Union
from enum import Enum, auto
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4
import abc

from pydantic import Field, field_validator, FilePath, HttpUrl, model_validator
from pydantic_core import Url

from dff.messengers.common.interface import MessengerInterfaceWithAttachments
from dff.utils.devel import JSONSerializableDict, PickleEncodedValue, JSONSerializableExtras


class DataModel(JSONSerializableExtras):
    """
    This class is a Pydantic BaseModel that can have any type and number of extras.
    """

    pass


class Session(Enum):
    """
    An enumeration that defines two possible states of a session.
    """

    ACTIVE = auto()
    FINISHED = auto()


class Command(DataModel):
    """
    This class is a subclass of DataModel and represents
    a command that can be executed in response to a user input.
    """

    pass


class Attachment(DataModel, abc.ABC):
    """
    DFF Message attachment base class.
    It is capable of serializing and validating all the model fields to JSON.
    """

    dff_attachment_type: str


class CallbackQuery(Attachment):
    """
    This class is a data model that represents a callback query attachment.
    It is sent as a response to non-message events, e.g. keyboard UI interactions.
    It has query string attribute, that represents the response data string.
    """

    query_string: Optional[str]
    dff_attachment_type: Literal["callback_query"] = "callback_query"


class Location(Attachment):
    """
    This class is a data model that represents a geographical
    location on the Earth's surface.
    It has two attributes, longitude and latitude, both of which are float values.
    If the absolute difference between the latitude and longitude values of the two
    locations is less than 0.00004, they are considered equal.
    """

    longitude: float
    latitude: float
    dff_attachment_type: Literal["location"] = "location"

    def __eq__(self, other):
        if isinstance(other, Location):
            return abs(self.latitude - other.latitude) + abs(self.longitude - other.longitude) < 0.00004
        return NotImplemented


class Contact(Attachment):
    """
    This class is a data model that represents a contact.
    It includes phone number, and user first and last name.
    """

    phone_number: str
    first_name: str
    last_name: Optional[str]
    dff_attachment_type: Literal["contact"] = "contact"


class Invoice(Attachment):
    """
    This class is a data model that represents an invoice.
    It includes title, description, currency name and amount.
    """

    title: str
    description: str
    currency: str
    amount: int
    dff_attachment_type: Literal["invoice"] = "invoice"


class PollOption(DataModel):
    """
    This class is a data model that represents a poll option.
    It includes the option name and votes number.
    """

    text: str
    votes: int = Field(default=0)
    dff_attachment_type: Literal["poll_option"] = "poll_option"


class Poll(Attachment):
    """
    This class is a data model that represents a poll.
    It includes a list of poll options.
    """

    question: str
    options: List[PollOption]
    dff_attachment_type: Literal["poll"] = "poll"


class DataAttachment(Attachment):
    """
    This class represents an attachment that can be either
    a file or a URL, along with an optional ID and title.
    This attachment can also be optionally cached for future use.
    """

    source: Optional[Union[HttpUrl, FilePath]] = None
    """Attachment source -- either a URL to a file or a local filepath."""
    title: Optional[str] = None
    """Text accompanying the data."""
    use_cache: bool = True
    """
    Whether to cache the file (only for URL and ID files).
    Disable this if you want to always respond with the most up-to-date version of the file.
    """
    cached_filename: Optional[FilePath] = None
    """This field stores a path to cached version of this file (retrieved from id or URL)."""
    id: Optional[str] = None
    """
    ID of the file on a file server.
    :py:meth:`~.MessengerInterfaceWithAttachments.populate_attachment` is used to retrieve bytes from ID.
    """

    async def _cache_attachment(self, data: bytes, directory: Path) -> None:
        """
        Cache attachment, save bytes into a file.

        :param data: attachment data bytes.
        :param directory: cache directory where attachment will be saved.
        """

        title = str(uuid4()) if self.title is None else self.title
        self.cached_filename = directory / title
        with open(self.cached_filename, "wb") as file:
            file.write(data)

    async def get_bytes(self, from_interface: MessengerInterfaceWithAttachments) -> Optional[bytes]:
        """
        Retrieve attachment bytes.
        If the attachment is represented by URL or saved in a file,
        it will be downloaded or read automatically.
        If cache use is allowed and the attachment is cached, cached file will be used.
        Otherwise, a :py:meth:`~.MessengerInterfaceWithAttachments.populate_attachment`
        will be used for receiving attachment bytes by ID and title.

        If cache use is allowed and the attachment is a URL or an ID, bytes will be cached locally.

        :param from_interface: messenger interface the attachment was received from.
        """

        if isinstance(self.source, Path):
            with open(self.source, "rb") as file:
                return file.read()
        elif self.use_cache and self.cached_filename is not None and self.cached_filename.exists():
            with open(self.cached_filename, "rb") as file:
                return file.read()
        elif isinstance(self.source, Url):
            with urlopen(self.source.unicode_string()) as url:
                attachment_data = url.read()
        else:
            attachment_data = await from_interface.populate_attachment(self)
        if self.use_cache:
            await self._cache_attachment(attachment_data, from_interface.attachments_directory)
        return attachment_data

    def __eq__(self, other):
        if isinstance(other, DataAttachment):
            if self.id != other.id:
                return False
            if self.source != other.source:
                return False
            if self.title != other.title:
                return False
            return True
        return NotImplemented

    @model_validator(mode="before")
    @classmethod
    def validate_source_or_id(cls, values: dict):
        if not isinstance(values, dict):
            raise AssertionError(f"Invalid constructor parameters: {str(values)}")
        if bool(values.get("source")) == bool(values.get("id")):
            raise AssertionError("Attachment type requires exactly one parameter, `source` or `id`, to be set.")
        return values

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, value):
        if isinstance(value, Path):
            return Path(value)
        return value


class Audio(DataAttachment):
    """Represents an audio file attachment."""

    dff_attachment_type: Literal["audio"] = "audio"


class Video(DataAttachment):
    """Represents a video file attachment."""

    dff_attachment_type: Literal["video"] = "video"


class Animation(DataAttachment):
    """Represents an animation file attachment."""

    dff_attachment_type: Literal["animation"] = "animation"


class Image(DataAttachment):
    """Represents an image file attachment."""

    dff_attachment_type: Literal["image"] = "image"


class Sticker(DataAttachment):
    """Represents a sticker as a file attachment."""

    dff_attachment_type: Literal["sticker"] = "sticker"


class Document(DataAttachment):
    """Represents a document file attachment."""

    dff_attachment_type: Literal["document"] = "document"


class VoiceMessage(DataAttachment):
    """Represents a voice message."""

    dff_attachment_type: Literal["voice_message"] = "voice_message"


class VideoMessage(DataAttachment):
    """Represents a video message."""

    dff_attachment_type: Literal["video_message"] = "video_message"


class MediaGroup(Attachment):
    """
    Represents a group of media attachments.
    Without this class attachments are sent one-by-one.

    Be mindful of limitations that certain services apply
    (e.g. Telegram does not allow audio or document files to be mixed with other types when using media groups,
    so you should send them separately by putting them directly in :py:attr:`~.Message.attachments`).
    """

    group: List[Union[Audio, Video, Image, Document]] = Field(default_factory=list)
    dff_attachment_type: Literal["media_group"] = "media_group"

    def __eq__(self, other):
        if isinstance(other, MediaGroup):
            if len(self.group) != len(other.group):
                return False
            for attachment1, attachment2 in zip(self.group, other.group):
                if attachment1 != attachment2:
                    return False
            return True
        return NotImplemented


class Message(DataModel):
    """
    Class representing a message and contains several
    class level variables to store message information.

    It includes message text, list of commands included in the message,
    list of attachments, annotations, MISC dictionary (that consists of
    user-defined parameters) and original message field that represents
    the update received from messenger interface API.
    """

    text: Optional[str] = None
    commands: Optional[List[Command]] = None
    attachments: Optional[
        List[
            Union[
                CallbackQuery,
                Location,
                Contact,
                Invoice,
                Poll,
                Audio,
                Video,
                Animation,
                Image,
                Sticker,
                Document,
                VoiceMessage,
                VideoMessage,
                MediaGroup,
            ]
        ]
    ] = None
    annotations: Optional[JSONSerializableDict] = None
    misc: Optional[JSONSerializableDict] = None
    original_message: Optional[PickleEncodedValue] = None

    def __init__(
        self,
        text: Optional[str] = None,
        commands: Optional[List[Command]] = None,
        attachments: Optional[
            List[
                Union[
                    CallbackQuery,
                    Location,
                    Contact,
                    Invoice,
                    Poll,
                    Audio,
                    Video,
                    Animation,
                    Image,
                    Sticker,
                    Document,
                    VoiceMessage,
                    VideoMessage,
                    MediaGroup,
                ]
            ]
        ] = None,
        annotations: Optional[JSONSerializableDict] = None,
        misc: Optional[JSONSerializableDict] = None,
        **kwargs,
    ):
        super().__init__(
            text=text, commands=commands, attachments=attachments, annotations=annotations, misc=misc, **kwargs
        )

    def __eq__(self, other):
        if isinstance(other, Message):
            for field in self.model_fields:
                if field not in other.model_fields:
                    return False
                if self.__getattribute__(field) != other.__getattribute__(field):
                    return False
            return True
        return NotImplemented

    def __repr__(self) -> str:
        return " ".join([f"{key}='{value}'" for key, value in self.model_dump(exclude_none=True).items()])
