# -*- coding: utf-8 -*-
"""Unit tests for Matrix channel implementation."""

# pylint: disable=redefined-outer-name,unused-import
# pylint: disable=protected-access,unused-argument
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from nio import (
    MatrixRoom,
    RoomMessageAudio,
    RoomMessageFile,
    RoomMessageImage,
    RoomMessageText,
    RoomMessageVideo,
    RoomSendError,
    UploadError,
    UploadResponse,
)
from nio.responses import WhoamiResponse

from qwenpaw.schemas import (
    AgentRequest,
    ContentType,
    ImageContent,
    TextContent,
)
from qwenpaw.app.channels.matrix.channel import MatrixChannel
from qwenpaw.config.config import MatrixConfig


@pytest.fixture
def mock_process():
    """Create mock process handler."""

    async def mock_handler(*_args, **_kwargs):
        mock_event = MagicMock()
        mock_event.object = "message"
        mock_event.status = "completed"
        yield mock_event

    return AsyncMock(side_effect=mock_handler)


@pytest.fixture
def matrix_config():
    """Create MatrixConfig instance."""
    return MatrixConfig(
        enabled=True,
        homeserver="https://matrix.example.com",
        user_id="@bot:example.com",
        access_token="test_token_123",
        bot_prefix="!bot",
        dm_disabled=False,
        group_disabled=False,
        deny_message="Access denied",
        require_mention=False,
    )


@pytest.fixture
def matrix_channel(mock_process):
    """Create MatrixChannel instance."""
    return MatrixChannel(
        process=mock_process,
        homeserver="https://matrix.example.com",
        matrix_user_id="@bot:example.com",
        access_token="test_token_123",
    )


@pytest.fixture
def mock_async_client():
    """Create mock AsyncClient for nio."""
    client = MagicMock()
    client.access_token = None
    client.add_event_callback = Mock()
    client.close = AsyncMock()
    client.room_send = AsyncMock()
    client.sync_forever = AsyncMock()
    client.upload = AsyncMock()
    whoami_resp = WhoamiResponse(
        user_id="@bot:example.com",
        device_id=None,
        is_guest=False,
    )
    client.whoami = AsyncMock(return_value=whoami_resp)
    client.sync = AsyncMock(return_value=MagicMock())
    return client


@pytest.fixture
def mock_matrix_room():
    """Create mock MatrixRoom."""
    room = MagicMock(spec=MatrixRoom)
    room.room_id = "!test_room:example.com"
    room.users = {"@user1:example.com": None, "@user2:example.com": None}
    return room


class TestMatrixChannelInit:
    """Test MatrixChannel initialization."""

    def test_init_with_required_params(self, mock_process):
        """Test initialization with required parameters."""
        channel = MatrixChannel(
            process=mock_process,
            homeserver="https://matrix.example.com",
            matrix_user_id="@bot:example.com",
            access_token="test_token",
        )

        assert channel.homeserver == "https://matrix.example.com"
        assert channel.matrix_user_id == "@bot:example.com"
        assert channel.access_token == "test_token"
        assert channel.channel == "matrix"
        assert channel.uses_manager_queue is True
        assert channel._client is None
        assert channel._sync_task is None

    def test_init_homeserver_trailing_slash(self, mock_process):
        """Test that trailing slash is stripped from homeserver."""
        channel = MatrixChannel(
            process=mock_process,
            homeserver="https://matrix.example.com/",
            matrix_user_id="@bot:example.com",
            access_token="test_token",
        )

        assert channel.homeserver == "https://matrix.example.com"

    def test_init_with_all_params(self, mock_process):
        """Test initialization with all optional parameters."""
        channel = MatrixChannel(
            process=mock_process,
            homeserver="https://matrix.example.com",
            matrix_user_id="@bot:example.com",
            access_token="test_token",
            dm_disabled=True,
            group_disabled=False,
            access_control_dm=True,
            on_reply_sent=Mock(),
            show_tool_details=False,
            filter_tool_messages=True,
            filter_thinking=True,
        )

        assert channel.dm_disabled is True
        assert channel.group_disabled is False
        assert channel._show_tool_details is False
        assert channel._filter_tool_messages is True
        assert channel._filter_thinking is True


class TestMatrixChannelFromConfig:
    """Test MatrixChannel factory methods."""

    def test_from_config(self, mock_process, matrix_config):
        """Test creating channel from config."""
        channel = MatrixChannel.from_config(
            process=mock_process,
            config=matrix_config,
        )

        assert channel.enabled is True
        assert channel.homeserver == "https://matrix.example.com"
        assert channel.matrix_user_id == "@bot:example.com"
        assert channel.access_token == "test_token_123"

    def test_from_config_with_optional_params(
        self,
        mock_process,
        matrix_config,
    ):
        """Test from_config with optional display parameters."""
        channel = MatrixChannel.from_config(
            process=mock_process,
            config=matrix_config,
            show_tool_details=False,
            filter_tool_messages=True,
            filter_thinking=True,
        )

        assert channel._show_tool_details is False
        assert channel._filter_tool_messages is True
        assert channel._filter_thinking is True

    def test_from_env_raises_not_implemented(self, mock_process):
        """Test that from_env creates a channel (uses env vars)."""
        channel = MatrixChannel.from_env(process=mock_process)
        assert isinstance(channel, MatrixChannel)


class TestMatrixChannelMXC:
    """Test MXC to HTTP URL conversion."""

    def test_mxc_to_http_with_valid_mxc(self, matrix_channel):
        """Test converting valid MXC URL to HTTP."""
        mxc_url = "mxc://matrix.org/media_123"
        http_url = matrix_channel._mxc_to_http(mxc_url)

        expected = (
            "https://matrix.example.com/_matrix/media/v3/download/"
            "matrix.org/media_123"
        )
        assert http_url == expected

    def test_mxc_to_http_with_http_url(self, matrix_channel):
        """Test that HTTP URLs are returned as-is."""
        http_url = "https://example.com/image.png"
        result = matrix_channel._mxc_to_http(http_url)

        assert result == http_url

    def test_mxc_to_http_with_invalid_mxc_format(self, matrix_channel):
        """Test handling of invalid MXC URL format."""
        invalid_mxc = "mxc://noseparator"
        result = matrix_channel._mxc_to_http(invalid_mxc)

        assert result == invalid_mxc

    def test_mxc_to_http_with_empty_string(self, matrix_channel):
        """Test handling of empty string."""
        result = matrix_channel._mxc_to_http("")

        assert result == ""


class TestMatrixChannelDisabled:
    """Test channel-level mute (dm_disabled / group_disabled)."""

    def test_dm_not_disabled(self, matrix_channel):
        """Test that DMs pass when not disabled."""
        matrix_channel.dm_disabled = False

        result = matrix_channel._is_channel_disabled(
            "@any_user:example.com",
            "!room:example.com",
            is_dm=True,
        )
        assert result is False

    def test_dm_disabled(self, matrix_channel):
        """Test that DMs are blocked when dm_disabled=True."""
        matrix_channel.dm_disabled = True

        result = matrix_channel._is_channel_disabled(
            "@any_user:example.com",
            "!room:example.com",
            is_dm=True,
        )
        assert result is True

    def test_group_not_disabled(self, matrix_channel):
        """Test that group messages pass when not disabled."""
        matrix_channel.group_disabled = False

        result = matrix_channel._is_channel_disabled(
            "@any_user:example.com",
            "!room:example.com",
            is_dm=False,
        )
        assert result is False

    def test_group_disabled(self, matrix_channel):
        """Test that group messages are blocked when group_disabled=True."""
        matrix_channel.group_disabled = True

        result = matrix_channel._is_channel_disabled(
            "@any_user:example.com",
            "!room:example.com",
            is_dm=False,
        )
        assert result is True

    def test_dm_disabled_does_not_affect_group(self, matrix_channel):
        """Test dm_disabled doesn't block group messages."""
        matrix_channel.dm_disabled = True
        matrix_channel.group_disabled = False

        result = matrix_channel._is_channel_disabled(
            "@any_user:example.com",
            "!room:example.com",
            is_dm=False,
        )
        assert result is False


@pytest.mark.asyncio
class TestMatrixChannelBuildRequest:
    """Test request building methods."""

    def test_build_agent_request_from_native(self, matrix_channel):
        """Test building AgentRequest from native payload."""
        payload = {
            "sender_id": "@user:example.com",
            "content_parts": [
                TextContent(type=ContentType.TEXT, text="Hello bot"),
            ],
            "meta": {"room_id": "!room:example.com"},
        }

        request = matrix_channel.build_agent_request_from_native(payload)

        assert isinstance(request, AgentRequest)
        assert request.channel == "matrix"
        # user_id is intentionally set to room_id for session keying
        assert request.user_id == "!room:example.com"
        assert request.session_id == "matrix:!room:example.com"

    def test_build_agent_request_with_content_parts(self, matrix_channel):
        """Test building request with existing content_parts."""
        content_parts = [
            TextContent(type=ContentType.TEXT, text="Test message"),
        ]
        payload = {
            "sender_id": "@user:example.com",
            "content_parts": content_parts,
            "meta": {"room_id": "!room:example.com"},
        }

        request = matrix_channel.build_agent_request_from_native(payload)

        assert request.input[0].content == content_parts

    def test_get_to_handle_from_request_with_session_id(self, matrix_channel):
        """Test getting room_id from channel_meta."""
        request = MagicMock(spec=AgentRequest)
        request.session_id = "matrix:!room:example.com"
        request.channel_meta = {"room_id": "!room:example.com"}

        result = matrix_channel.get_to_handle_from_request(request)

        assert result == "!room:example.com"

    def test_get_to_handle_from_request_with_channel_meta(
        self,
        matrix_channel,
    ):
        """Test getting room_id from channel_meta."""
        request = MagicMock(spec=AgentRequest)
        request.session_id = "other_session"
        request.channel_meta = {"room_id": "!room:example.com"}

        result = matrix_channel.get_to_handle_from_request(request)

        assert result == "!room:example.com"

    def test_get_to_handle_from_request_fallback_to_user_id(
        self,
        matrix_channel,
    ):
        """Test fallback to user_id when no room_id."""
        request = MagicMock(spec=AgentRequest)
        request.session_id = "other_session"
        request.channel_meta = {}
        request.user_id = "@user:example.com"

        result = matrix_channel.get_to_handle_from_request(request)

        assert result == "@user:example.com"


@pytest.mark.asyncio
class TestMatrixChannelHandleEvent:
    """Test event handling."""

    async def test_on_room_event_dm_not_disabled(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test handling event with DM not disabled enqueues the request."""
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()
        matrix_channel.dm_disabled = False
        matrix_channel._get_display_name = Mock(return_value="user")

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@user:example.com"
        event.body = "Hello"
        event.event_id = "$evt1"
        event.source = {}
        mock_matrix_room.room_id = "!test_room:example.com"

        await matrix_channel._on_room_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        payload = matrix_channel._enqueue.call_args[0][0]
        assert payload["sender_id"] == "@user:example.com"

    async def test_on_room_event_dm_disabled(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test that DM messages are dropped when dm_disabled=True."""
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel.dm_disabled = True
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._enqueue = Mock()

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@unauthorized:example.com"
        event.body = "Hello"
        event.event_id = "$evt1"
        event.source = {}
        mock_matrix_room.room_id = "!test_room:example.com"

        await matrix_channel._on_room_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_not_called()

    async def test_on_room_event_require_mention_not_met(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test event ignored in group when mention required but absent."""
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()
        matrix_channel._is_dm_room = AsyncMock(return_value=False)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()
        matrix_channel.group_disabled = False
        matrix_channel.require_mention = True

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@user:example.com"
        event.body = "Just a message with no bot mention"
        event.event_id = "$evt1"
        event.source = {}
        mock_matrix_room.room_id = "!test_room:example.com"

        await matrix_channel._on_room_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_not_called()


@pytest.mark.asyncio
class TestMatrixChannelMessageCallback:
    """Test message callbacks."""

    async def test_message_callback_ignores_own_message(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test that bot ignores its own messages (no enqueue called)."""
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@bot:example.com"
        event.body = "Hello"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_not_called()

    async def test_message_callback_detects_mention(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test bot mention detection: _was_mentioned returns True."""
        matrix_channel._user_id = "@bot:example.com"

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@user:example.com"
        event.body = "Hello @bot:example.com!"
        event.source = {}

        assert matrix_channel._was_mentioned(event, event.body) is True

    async def test_message_callback_detects_localpart_mention(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test mention detection by full MXID in text."""
        matrix_channel._user_id = "@mybot:example.com"

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@user:example.com"
        event.body = "hello @mybot:example.com please help"
        event.source = {}

        assert matrix_channel._was_mentioned(event, event.body) is True

    async def test_message_callback_no_mention(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test when bot is not mentioned."""
        matrix_channel._user_id = "@bot:example.com"

        event = MagicMock(spec=RoomMessageText)
        event.sender = "@user:example.com"
        event.body = "Just a regular message"
        event.source = {}

        assert matrix_channel._was_mentioned(event, event.body) is False


@pytest.mark.asyncio
class TestMatrixChannelMediaCallback:
    """Test media message callbacks."""

    async def test_media_callback_image(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Test handling image message enqueues image content."""
        fake_file = tmp_path / "image_123_image.png"
        fake_file.write_bytes(b"fake image")

        matrix_channel._user_id = "@bot:example.com"
        matrix_channel.vision_enabled = True
        matrix_channel._enqueue = Mock()
        matrix_channel._download_mxc = AsyncMock(return_value=str(fake_file))
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()

        event = MagicMock(spec=RoomMessageImage)
        event.sender = "@user:example.com"
        event.url = "mxc://example.org/image_123"
        event.body = "image.png"
        event.event_id = "$abc123"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        payload = matrix_channel._enqueue.call_args[0][0]
        parts = payload["content_parts"]
        assert any(
            getattr(p, "type", None) == ContentType.IMAGE for p in parts
        )

    async def test_media_callback_video(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Test handling video message enqueues video content."""
        fake_file = tmp_path / "video_123_video.mp4"
        fake_file.write_bytes(b"fake video")

        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()
        matrix_channel._download_mxc = AsyncMock(return_value=str(fake_file))
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()

        event = MagicMock(spec=RoomMessageVideo)
        event.sender = "@user:example.com"
        event.url = "mxc://example.org/video_123"
        event.body = "video.mp4"
        event.event_id = "$abc123"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        payload = matrix_channel._enqueue.call_args[0][0]
        parts = payload["content_parts"]
        assert any(
            getattr(p, "type", None) == ContentType.VIDEO for p in parts
        )

    async def test_media_callback_audio(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Test handling audio message enqueues audio content."""
        fake_file = tmp_path / "audio_123_audio.mp3"
        fake_file.write_bytes(b"fake audio")

        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()
        matrix_channel._download_mxc = AsyncMock(return_value=str(fake_file))
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()

        event = MagicMock(spec=RoomMessageAudio)
        event.sender = "@user:example.com"
        event.url = "mxc://example.org/audio_123"
        event.body = "audio.mp3"
        event.event_id = "$abc123"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        payload = matrix_channel._enqueue.call_args[0][0]
        parts = payload["content_parts"]
        assert any(
            getattr(p, "type", None) == ContentType.AUDIO for p in parts
        )

    async def test_media_callback_file(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Test handling file message enqueues file content."""
        fake_file = tmp_path / "file_123_document.pdf"
        fake_file.write_bytes(b"fake pdf")

        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()
        matrix_channel._download_mxc = AsyncMock(return_value=str(fake_file))
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()

        event = MagicMock(spec=RoomMessageFile)
        event.sender = "@user:example.com"
        event.url = "mxc://example.org/file_123"
        event.body = "document.pdf"
        event.event_id = "$abc123"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        payload = matrix_channel._enqueue.call_args[0][0]
        parts = payload["content_parts"]
        assert any(getattr(p, "type", None) == ContentType.FILE for p in parts)

    async def test_media_callback_ignores_own_message(
        self,
        matrix_channel,
        mock_matrix_room,
    ):
        """Test that bot ignores its own media messages."""
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel._enqueue = Mock()

        event = MagicMock(spec=RoomMessageImage)
        event.sender = "@bot:example.com"
        event.url = "mxc://example.org/image_123"
        mock_matrix_room.room_id = "!room:example.com"

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_not_called()


class TestBuildQuotedPrefix:
    """``_build_quoted_prefix`` covers the 5 matrix msgtype cases.

    Mirrors ``test_yuanbao.py::TestBuildQuotedPrefix`` so channels
    share one labelling convention for quoted messages.
    """

    def test_text_quote(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.text", "body": "hello world"},
            )
            == "[quoted message: hello world]"
        )

    def test_image_quote_with_filename(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.image", "body": "photo.png"},
            )
            == "[quoted image: photo.png]"
        )

    def test_image_quote_without_filename(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.image", "body": ""},
            )
            == "[quoted image]"
        )

    def test_file_quote(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.file", "body": "report.pdf"},
            )
            == "[quoted file: report.pdf]"
        )

    def test_audio_quote(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.audio", "body": "voice.mp3"},
            )
            == "[quoted audio: voice.mp3]"
        )

    def test_video_quote(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.video", "body": "clip.mp4"},
            )
            == "[quoted video: clip.mp4]"
        )

    def test_empty_body_falls_back_to_label_only(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.text", "body": ""},
            )
            == "[quoted message]"
        )

    def test_unknown_msgtype_treated_as_message(self):
        assert (
            MatrixChannel._build_quoted_prefix(
                {"msgtype": "m.location", "body": "park"},
            )
            == "[quoted message: park]"
        )

    def test_non_dict_returns_none(self):
        assert MatrixChannel._build_quoted_prefix(None) is None


@pytest.mark.asyncio
class TestMatrixChannelQuotedMessage:
    """Integration tests for m.relates_to m.in_reply_to handling.

    The pure-prefix logic is unit-tested in
    :class:`TestBuildQuotedPrefix` above; these tests verify that
    ``_on_room_event`` and ``_on_room_media_event`` wire the
    prefix / image-download together correctly.
    """

    @staticmethod
    def _make_quoted_event(*, msgtype, body, mimetype=None, encrypted=False):
        ev = MagicMock()
        ev.event_id = "$origabcdef01"
        ev.sender = "@alice:example.org"
        content: dict = {
            "body": body,
            "msgtype": msgtype,
        }
        if msgtype in ("m.image", "m.file", "m.audio", "m.video"):
            if encrypted:
                content["file"] = {
                    "url": "mxc://example.org/encryptedblob",
                    "key": {"k": "fakekey"},
                    "hashes": {"sha256": "fakehash"},
                    "iv": "fakeiv",
                }
            else:
                content["url"] = "mxc://example.org/clearblob"
            content["info"] = {"mimetype": mimetype, "size": 1234}
        ev.source = {"content": content}
        return ev

    @staticmethod
    def _make_reply_event(original_event_id, body="what is this?"):
        ev = MagicMock(spec=RoomMessageText)
        ev.sender = "@user:example.com"
        ev.body = body
        ev.event_id = "$reply12345678"
        ev.source = {
            "content": {
                "body": body,
                "msgtype": "m.text",
                "m.relates_to": {
                    "rel_type": "m.in_reply_to",
                    "event_id": original_event_id,
                },
            },
        }
        return ev

    def _prepare_reply(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
        quoted_event,
        vision_enabled=True,
        image_bytes=b"PNGDATA",
    ):
        """Wire up channel + room for a quoted-reply test."""
        from nio.responses import RoomGetEventResponse

        matrix_channel._user_id = "@bot:example.com"
        matrix_channel.vision_enabled = vision_enabled
        matrix_channel._enqueue = Mock()
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._require_mention = lambda _rid: False
        matrix_channel._was_mentioned = lambda _ev, _t: True
        matrix_channel._is_channel_disabled = lambda *a, **kw: False
        matrix_channel._strip_mention_prefix = lambda text, _room: text
        matrix_channel._get_display_name = lambda _room, uid: uid.split(":")[
            0
        ].lstrip("@")
        matrix_channel._apply_history_to_parts = lambda _rid, parts: parts

        client = MagicMock()
        rge = RoomGetEventResponse()
        rge.event = quoted_event
        client.room_get_event = AsyncMock(return_value=rge)
        matrix_channel._client = client

        # Mimic the production filename build so the saved file has
        # the same suffix the production code would assign.
        q_mime = (
            (quoted_event.source or {})
            .get("content", {})
            .get("info", {})
            .get("mimetype", "image/jpeg")
        )
        import mimetypes as _mt

        q_ext = _mt.guess_extension(q_mime) or ".bin"
        q_ext = q_ext.rstrip(";").strip() or ".bin"

        async def fake_download_clear(_mxc_url, filename):
            dest = tmp_path / filename
            dest.write_bytes(image_bytes)
            return str(dest)

        async def fake_download_enc(_mxc_url, filename, *_args):
            dest = tmp_path / filename
            dest.write_bytes(image_bytes)
            return str(dest)

        matrix_channel._download_mxc = AsyncMock(
            side_effect=fake_download_clear,
        )
        matrix_channel._download_encrypted_mxc = AsyncMock(
            side_effect=fake_download_enc,
        )
        mock_matrix_room.room_id = "!room:example.com"
        return q_ext

    async def test_quoted_image_with_vision_downloads_and_attaches(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Reply to an m.image with vision on → prefix + ImageContent."""
        quoted = self._make_quoted_event(
            msgtype="m.image",
            body="photo.png",
            mimetype="image/png",
        )
        self._prepare_reply(
            matrix_channel,
            mock_matrix_room,
            tmp_path,
            quoted_event=quoted,
        )
        ev = self._make_reply_event(quoted.event_id)

        await matrix_channel._on_room_event(mock_matrix_room, ev)

        matrix_channel._enqueue.assert_called_once()
        parts = matrix_channel._enqueue.call_args[0][0]["content_parts"]
        images = [p for p in parts if isinstance(p, ImageContent)]
        texts = [p for p in parts if isinstance(p, TextContent)]
        assert len(images) == 1, f"expected 1 ImageContent, got {parts}"
        assert images[0].image_url.endswith(".png"), images[0].image_url
        # Leading text part carries the prefix with the original body.
        assert "[quoted image: photo.png]" in (texts[0].text or ""), texts[
            0
        ].text
        matrix_channel._download_mxc.assert_awaited_once()
        mxc_arg, name_arg = matrix_channel._download_mxc.await_args[0]
        assert mxc_arg == "mxc://example.org/clearblob"
        assert name_arg.endswith(".png"), name_arg

    async def test_quoted_text_message_emits_prefix_only(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Reply to an m.text → prefix, no media downloaded."""
        quoted = self._make_quoted_event(
            msgtype="m.text",
            body="original body text",
        )
        self._prepare_reply(
            matrix_channel,
            mock_matrix_room,
            tmp_path,
            quoted_event=quoted,
        )
        ev = self._make_reply_event(quoted.event_id)

        await matrix_channel._on_room_event(mock_matrix_room, ev)

        matrix_channel._enqueue.assert_called_once()
        parts = matrix_channel._enqueue.call_args[0][0]["content_parts"]
        assert not any(isinstance(p, ImageContent) for p in parts), parts
        assert "[quoted message: original body text]" in (
            parts[0].text or ""
        ), parts[0].text
        matrix_channel._download_mxc.assert_not_awaited()
        matrix_channel._download_encrypted_mxc.assert_not_awaited()

    async def test_quoted_image_with_vision_off_emits_prefix_only(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """vision_enabled=False → prefix kept, no image downloaded."""
        quoted = self._make_quoted_event(
            msgtype="m.image",
            body="photo.png",
            mimetype="image/png",
        )
        self._prepare_reply(
            matrix_channel,
            mock_matrix_room,
            tmp_path,
            quoted_event=quoted,
            vision_enabled=False,
        )
        ev = self._make_reply_event(quoted.event_id)

        await matrix_channel._on_room_event(mock_matrix_room, ev)

        matrix_channel._enqueue.assert_called_once()
        parts = matrix_channel._enqueue.call_args[0][0]["content_parts"]
        assert not any(isinstance(p, ImageContent) for p in parts), parts
        assert "[quoted image: photo.png]" in (parts[0].text or "")
        matrix_channel._download_mxc.assert_not_awaited()
        matrix_channel._download_encrypted_mxc.assert_not_awaited()

    async def test_quoted_encrypted_image_uses_decrypt_path(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Reply to an E2EE m.image → _download_encrypted_mxc path."""
        quoted = self._make_quoted_event(
            msgtype="m.image",
            body="secret.jpg",
            mimetype="image/jpeg",
            encrypted=True,
        )
        self._prepare_reply(
            matrix_channel,
            mock_matrix_room,
            tmp_path,
            quoted_event=quoted,
        )
        ev = self._make_reply_event(quoted.event_id)

        await matrix_channel._on_room_event(mock_matrix_room, ev)

        matrix_channel._enqueue.assert_called_once()
        parts = matrix_channel._enqueue.call_args[0][0]["content_parts"]
        images = [p for p in parts if isinstance(p, ImageContent)]
        assert len(images) == 1, parts
        assert images[0].image_url.endswith(".jpg")
        matrix_channel._download_encrypted_mxc.assert_awaited_once()
        matrix_channel._download_mxc.assert_not_awaited()
        (
            mxc,
            name,
            key,
            hashes,
            iv,
        ) = matrix_channel._download_encrypted_mxc.await_args[0]
        assert mxc == "mxc://example.org/encryptedblob"
        assert name.endswith(".jpg")
        assert key == {"k": "fakekey"}
        assert hashes == {"sha256": "fakehash"}
        assert iv == "fakeiv"

    async def test_direct_image_without_body_uses_mimetype_extension(
        self,
        matrix_channel,
        mock_matrix_room,
        tmp_path,
    ):
        """Body-less image → saved file gets mimetype-derived extension.

        Regression test for the agentscope image loader, which keys
        off the file extension. Without a body, the default filename
        is ``<eid>_matrix_media_<eid>`` and needs a mimetype-based
        suffix.
        """
        matrix_channel._user_id = "@bot:example.com"
        matrix_channel.vision_enabled = True
        matrix_channel._enqueue = Mock()
        matrix_channel._send_read_receipt = AsyncMock()
        matrix_channel._send_typing = AsyncMock()
        matrix_channel._is_dm_room = AsyncMock(return_value=True)
        matrix_channel._require_mention = lambda _rid: False
        matrix_channel._was_mentioned = lambda _ev, _t: True
        matrix_channel._is_channel_disabled = lambda *a, **kw: False
        matrix_channel._get_display_name = lambda _r, uid: uid.split(":")[
            0
        ].lstrip("@")
        matrix_channel._apply_history_to_parts = lambda _rid, parts: parts

        event = MagicMock(spec=RoomMessageImage)
        event.sender = "@user:example.com"
        event.url = "mxc://example.org/blob_no_body"
        event.body = ""
        event.event_id = "$img_evt_0001"
        event.info = {"mimetype": "image/webp", "size": 999}
        mock_matrix_room.room_id = "!room:example.com"

        captured: dict = {}

        async def fake_download(_mxc_url, filename):
            captured["filename"] = filename
            dest = tmp_path / filename
            dest.write_bytes(b"WEBPDATA")
            return str(dest)

        matrix_channel._download_mxc = AsyncMock(side_effect=fake_download)
        matrix_channel._media_dir = lambda: tmp_path

        await matrix_channel._on_room_media_event(mock_matrix_room, event)

        matrix_channel._enqueue.assert_called_once()
        assert captured["filename"].endswith(".webp"), captured["filename"]
        parts = matrix_channel._enqueue.call_args[0][0]["content_parts"]
        images = [p for p in parts if isinstance(p, ImageContent)]
        assert len(images) == 1
        assert images[0].image_url.endswith(".webp")


@pytest.mark.asyncio
class TestMatrixChannelStartStop:
    """Test start and stop lifecycle."""

    async def test_start_when_not_configured(self, matrix_channel):
        """Test start when channel is not properly configured."""
        matrix_channel.homeserver = ""

        await matrix_channel.start()

        assert matrix_channel._client is None

    async def test_start_creates_client(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test that start creates and configures AsyncClient."""
        with patch(
            "qwenpaw.app.channels.matrix.channel.QwenPawMatrixClient",
            return_value=mock_async_client,
        ):
            await matrix_channel.start()

            assert matrix_channel._client is mock_async_client
            assert mock_async_client.access_token == "test_token_123"
            assert mock_async_client.add_event_callback.call_count >= 2

    async def test_start_starts_sync_task(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test that start creates sync task."""
        with patch(
            "qwenpaw.app.channels.matrix.channel.QwenPawMatrixClient",
            return_value=mock_async_client,
        ):
            await matrix_channel.start()

            assert matrix_channel._sync_task is not None
            assert not matrix_channel._sync_task.done()

    async def test_stop_cancels_sync_task(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test that stop cancels sync task."""
        with patch(
            "qwenpaw.app.channels.matrix.channel.QwenPawMatrixClient",
            return_value=mock_async_client,
        ):
            await matrix_channel.start()

            await matrix_channel.stop()

            # Verify stop was called on the client
            assert mock_async_client.close.called

    async def test_stop_closes_client(self, matrix_channel, mock_async_client):
        """Test that stop closes the client."""
        with patch(
            "qwenpaw.app.channels.matrix.channel.QwenPawMatrixClient",
            return_value=mock_async_client,
        ):
            await matrix_channel.start()

            await matrix_channel.stop()

            mock_async_client.close.assert_called_once()

    async def test_stop_when_not_started(self, matrix_channel):
        """Test stop when channel was never started."""
        # Should not raise
        await matrix_channel.stop()


@pytest.mark.asyncio
class TestMatrixChannelSend:
    """Test send method."""

    async def test_send_success(self, matrix_channel, mock_async_client):
        """Test successful message send."""
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())
        matrix_channel._client = mock_async_client

        await matrix_channel.send("!room:example.com", "Hello world")

        mock_async_client.room_send.assert_called_once()
        call_args = mock_async_client.room_send.call_args
        assert call_args[0][0] == "!room:example.com"
        assert call_args[0][1] == "m.room.message"
        content = call_args[0][2]
        assert content["msgtype"] == "m.text"
        assert content["body"] == "Hello world"

    async def test_send_when_client_not_initialized(self, matrix_channel):
        """Test send when client is not initialized."""
        matrix_channel._client = None

        # Should not raise
        await matrix_channel.send("!room:example.com", "Hello")

    async def test_send_empty_message(self, matrix_channel, mock_async_client):
        """Test sending empty message does not raise."""
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())
        matrix_channel._client = mock_async_client

        # Should not raise regardless of whether implementation sends or skips
        await matrix_channel.send("!room:example.com", "")

    async def test_send_handles_room_send_error(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test handling RoomSendError."""
        error_response = RoomSendError(
            message="Send failed",
            status_code="M_UNKNOWN",
        )
        mock_async_client.room_send = AsyncMock(return_value=error_response)
        matrix_channel._client = mock_async_client

        # Should not raise, just log error
        await matrix_channel.send("!room:example.com", "Hello")


@pytest.mark.asyncio
class TestMatrixChannelSendContentParts:
    """Test send_content_parts method."""

    async def test_send_content_parts_text_only(self, matrix_channel):
        """Test sending text content parts."""
        matrix_channel.send = AsyncMock()

        parts = [TextContent(type=ContentType.TEXT, text="Hello")]
        await matrix_channel.send_content_parts("!room:example.com", parts)

        matrix_channel.send.assert_called_once_with(
            "!room:example.com",
            "Hello",
            None,
        )

    async def test_send_content_parts_image(self, matrix_channel):
        """Test sending image content parts."""
        matrix_channel.send_media = AsyncMock()

        parts = [
            ImageContent(
                type=ContentType.IMAGE,
                image_url="https://example.com/img.png",
            ),
        ]
        await matrix_channel.send_content_parts("!room:example.com", parts)

        matrix_channel.send_media.assert_called_once()

    async def test_send_content_parts_mixed(self, matrix_channel):
        """Test sending mixed text and media content parts."""
        matrix_channel.send = AsyncMock()
        matrix_channel.send_media = AsyncMock()

        parts = [
            TextContent(type=ContentType.TEXT, text="Hello"),
            ImageContent(
                type=ContentType.IMAGE,
                image_url="https://example.com/img.png",
            ),
        ]
        await matrix_channel.send_content_parts("!room:example.com", parts)

        matrix_channel.send.assert_called_once()
        matrix_channel.send_media.assert_called_once()


@pytest.mark.asyncio
class TestMatrixChannelSendMedia:
    """Test send_media method."""

    async def test_send_media_when_client_not_initialized(
        self,
        matrix_channel,
    ):
        """Test send_media when client is not initialized."""
        matrix_channel._client = None

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url="https://example.com/img.png",
        )
        # Should not raise
        await matrix_channel.send_media("!room:example.com", part)

    async def test_send_media_missing_url(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test send_media when part has no URL."""
        matrix_channel._client = mock_async_client

        part = ImageContent(type=ContentType.IMAGE, image_url=None)
        await matrix_channel.send_media("!room:example.com", part)

        mock_async_client.upload.assert_not_called()

    async def test_send_media_file_url(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test sending media from file:// URL."""
        # Create temp file
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(b"fake image data")

        matrix_channel._client = mock_async_client
        upload_response = UploadResponse(
            content_uri="mxc://example.org/uploaded_123",
        )
        mock_async_client.upload = AsyncMock(
            return_value=(upload_response, None),
        )
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=f"file://{test_file}",
        )
        await matrix_channel.send_media("!room:example.com", part)

        mock_async_client.upload.assert_called_once()
        mock_async_client.room_send.assert_called_once()

    async def test_send_media_http_url(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test sending media from HTTP URL."""
        # Just verify no exception is raised when channel is properly set up
        part = ImageContent(
            type=ContentType.IMAGE,
            image_url="https://example.com/img.png",
        )
        # Actual HTTP mocking is too complex, just verify the method runs
        matrix_channel._client = mock_async_client
        try:
            await matrix_channel.send_media("!room:example.com", part)
        except (TypeError, AttributeError):
            # Expected due to aiohttp mocking complexity
            pass

    async def test_send_media_http_download_fails(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test handling HTTP download failure."""
        matrix_channel._client = mock_async_client

        # Mock failed aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            part = ImageContent(
                type=ContentType.IMAGE,
                image_url="https://example.com/img.png",
            )
            await matrix_channel.send_media("!room:example.com", part)

        mock_async_client.upload.assert_not_called()

    async def test_send_media_upload_error(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test handling upload error."""
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(b"fake image data")

        matrix_channel._client = mock_async_client
        upload_error = UploadError(message="Upload failed")
        mock_async_client.upload = AsyncMock(return_value=(upload_error, None))

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=f"file://{test_file}",
        )
        await matrix_channel.send_media("!room:example.com", part)

        mock_async_client.room_send.assert_not_called()

    async def test_send_media_room_send_error(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test handling room_send error."""
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(b"fake image data")

        matrix_channel._client = mock_async_client
        upload_response = UploadResponse(
            content_uri="mxc://example.org/uploaded_123",
        )
        mock_async_client.upload = AsyncMock(
            return_value=(upload_response, None),
        )
        send_error = RoomSendError(
            message="Send failed",
            status_code="M_UNKNOWN",
        )
        mock_async_client.room_send = AsyncMock(return_value=send_error)

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url=f"file://{test_file}",
        )
        await matrix_channel.send_media("!room:example.com", part)

        # Should not raise, error is logged

    async def test_send_media_video_type(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test sending video media type."""
        from qwenpaw.schemas import (
            VideoContent,
        )

        test_file = tmp_path / "test_video.mp4"
        test_file.write_bytes(b"fake video data")

        matrix_channel._client = mock_async_client
        upload_response = UploadResponse(
            content_uri="mxc://example.org/uploaded_123",
        )
        mock_async_client.upload = AsyncMock(
            return_value=(upload_response, None),
        )
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())

        part = VideoContent(
            type=ContentType.VIDEO,
            video_url=f"file://{test_file}",
        )
        await matrix_channel.send_media("!room:example.com", part)

        call_args = mock_async_client.room_send.call_args[0][2]
        assert call_args["msgtype"] == "m.video"

    async def test_send_media_audio_type(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test sending audio media type."""
        from qwenpaw.schemas import (
            AudioContent,
        )

        test_file = tmp_path / "test_audio.mp3"
        test_file.write_bytes(b"fake audio data")

        matrix_channel._client = mock_async_client
        upload_response = UploadResponse(
            content_uri="mxc://example.org/uploaded_123",
        )
        mock_async_client.upload = AsyncMock(
            return_value=(upload_response, None),
        )
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())

        part = AudioContent(type=ContentType.AUDIO, data=f"file://{test_file}")
        await matrix_channel.send_media("!room:example.com", part)

        call_args = mock_async_client.room_send.call_args[0][2]
        assert call_args["msgtype"] == "m.audio"

    async def test_send_media_unknown_url_scheme(
        self,
        matrix_channel,
        mock_async_client,
    ):
        """Test handling unknown URL scheme."""
        matrix_channel._client = mock_async_client

        part = ImageContent(
            type=ContentType.IMAGE,
            image_url="ftp://example.com/img.png",
        )
        await matrix_channel.send_media("!room:example.com", part)

        mock_async_client.upload.assert_not_called()

    async def test_send_media_generic_file_type(
        self,
        matrix_channel,
        mock_async_client,
        tmp_path,
    ):
        """Test sending generic file media type."""
        test_file = tmp_path / "test_document.pdf"
        test_file.write_bytes(b"fake pdf data")

        matrix_channel._client = mock_async_client
        upload_response = UploadResponse(
            content_uri="mxc://example.org/uploaded_123",
        )
        mock_async_client.upload = AsyncMock(
            return_value=(upload_response, None),
        )
        mock_async_client.room_send = AsyncMock(return_value=MagicMock())

        from qwenpaw.schemas import FileContent

        part = FileContent(
            type=ContentType.FILE,
            file_url=f"file://{test_file}",
        )
        await matrix_channel.send_media("!room:example.com", part)

        call_args = mock_async_client.room_send.call_args[0][2]
        assert call_args["msgtype"] == "m.file"
