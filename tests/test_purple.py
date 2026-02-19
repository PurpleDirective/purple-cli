"""
Comprehensive unit tests for Purple CLI, docs server, and memory server.

All tests run WITHOUT Ollama. External dependencies (MCP servers, Ollama,
filesystem side effects) are mocked where necessary.

Run:
    python -m pytest tests/test_purple.py -v
"""

import importlib
import importlib.util
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys

# ---------------------------------------------------------------------------
# Import targets using importlib to avoid module name collisions
# ---------------------------------------------------------------------------

# 1. CLI: purple.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
from purple import (
    _parse_xml_tool_calls,
    OllamaChat,
    MCPToolManager,
)

# 2. Docs server: docs/server.py (import as docs_server)
_docs_spec = importlib.util.spec_from_file_location(
    "docs_server",
    str(Path(__file__).resolve().parent.parent / "docs" / "server.py"),
)
docs_server = importlib.util.module_from_spec(_docs_spec)
_docs_spec.loader.exec_module(docs_server)

_validate_path = docs_server._validate_path
_convert_value = docs_server._convert_value
_resolve_output = docs_server._resolve_output

# 3. Memory server: memory/server.py (import as mem_server)
_mem_spec = importlib.util.spec_from_file_location(
    "mem_server",
    str(Path(__file__).resolve().parent.parent / "memory" / "server.py"),
)
mem_server = importlib.util.module_from_spec(_mem_spec)
_mem_spec.loader.exec_module(mem_server)


# ============================================================================
# SECTION 1: _parse_xml_tool_calls (purple.py)
# ============================================================================

class TestParseXmlToolCalls:
    """Tests for the XML tool call parser in purple.py."""

    # -- Format 1: <function=name><parameter=key>value</parameter></function>

    def test_format1_single_param(self):
        content = '<function=store_memory><parameter=content>hello world</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "store_memory"
        assert result[0]["function"]["arguments"]["content"] == "hello world"

    def test_format1_multiple_params(self):
        content = (
            '<function=store_memory>'
            '<parameter=content>test note</parameter>'
            '<parameter=type>fact</parameter>'
            '</function>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 1
        func = result[0]["function"]
        assert func["name"] == "store_memory"
        assert func["arguments"]["content"] == "test note"
        assert func["arguments"]["type"] == "fact"

    def test_format1_multiple_calls(self):
        content = (
            '<function=store_memory><parameter=content>note 1</parameter></function>'
            '<function=recall_memories><parameter=query>all</parameter></function>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "store_memory"
        assert result[1]["function"]["name"] == "recall_memories"

    def test_format1_whitespace_in_value(self):
        content = '<function=store_memory><parameter=content>  spaced value  </parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        # strip() is called on the value
        assert result[0]["function"]["arguments"]["content"] == "spaced value"

    def test_format1_empty_parameter_value(self):
        content = '<function=test_tool><parameter=key></parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["arguments"]["key"] == ""

    def test_format1_multiline_value(self):
        content = (
            '<function=store_memory>'
            '<parameter=content>line one\nline two\nline three</parameter>'
            '</function>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert "line one\nline two\nline three" in result[0]["function"]["arguments"]["content"]

    def test_format1_no_parameters(self):
        """Function with no parameters at all."""
        content = '<function=list_tools></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "list_tools"
        assert result[0]["function"]["arguments"] == {}

    def test_format1_parameter_name_with_underscores(self):
        content = '<function=my_tool><parameter=my_param_name>value</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert "my_param_name" in result[0]["function"]["arguments"]

    # -- Format 2: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    def test_format2_basic(self):
        content = '<tool_call>{"name": "recall_memories", "arguments": {"query": "test"}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "recall_memories"
        assert result[0]["function"]["arguments"]["query"] == "test"

    def test_format2_multiple_calls(self):
        content = (
            '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_format2_empty_arguments(self):
        content = '<tool_call>{"name": "list_recent", "arguments": {}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["arguments"] == {}

    def test_format2_missing_name_key(self):
        content = '<tool_call>{"arguments": {"query": "test"}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        # Missing name defaults to ""
        assert result[0]["function"]["name"] == ""

    def test_format2_missing_arguments_key(self):
        content = '<tool_call>{"name": "test_tool"}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["arguments"] == {}

    def test_format2_with_whitespace(self):
        content = '<tool_call>\n  {"name": "test", "arguments": {"a": "b"}}\n</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    def test_format2_extra_keys_ignored(self):
        """Extra keys in the JSON should not cause failure."""
        content = '<tool_call>{"name": "test", "arguments": {}, "extra": true}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    def test_format2_with_nested_json_objects(self):
        content = '<tool_call>{"name": "create_excel", "arguments": {"sheets": "[{\\"name\\": \\"Sheet1\\"}]"}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "create_excel"

    # -- Mixed content (text around XML)

    def test_text_before_format1(self):
        content = 'Let me call a tool for you. <function=recall_memories><parameter=query>search</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "recall_memories"

    def test_text_after_format1(self):
        content = '<function=test><parameter=a>b</parameter></function> I called the tool.'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    def test_text_surrounding_format2(self):
        content = 'Here is my call: <tool_call>{"name": "test", "arguments": {}}</tool_call> done.'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    def test_very_long_content_with_tool_call_at_end(self):
        """Long text before the tool call should still be found."""
        preamble = "Here is a very long explanation. " * 500
        content = preamble + '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    # -- Malformed inputs

    def test_malformed_unclosed_function_tag(self):
        content = '<function=test><parameter=a>b</parameter>'
        result = _parse_xml_tool_calls(content)
        # No closing </function> -- regex won't match
        assert result is None

    def test_malformed_unclosed_tool_call(self):
        content = '<tool_call>{"name": "test", "arguments": {}}'
        result = _parse_xml_tool_calls(content)
        assert result is None

    def test_malformed_invalid_json_in_tool_call(self):
        content = '<tool_call>{not valid json}</tool_call>'
        result = _parse_xml_tool_calls(content)
        # json.loads fails, continues -- no valid calls found
        assert result is None

    def test_malformed_partial_json(self):
        content = '<tool_call>{"name": "test", "arguments": {</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is None

    def test_malformed_mixed_valid_and_invalid_format2(self):
        """One valid, one invalid -- should return just the valid one."""
        content = (
            '<tool_call>{broken json}</tool_call>'
            '<tool_call>{"name": "good", "arguments": {"x": 1}}</tool_call>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "good"

    # -- Empty / no XML

    def test_empty_content(self):
        result = _parse_xml_tool_calls("")
        assert result is None

    def test_no_xml_plain_text(self):
        result = _parse_xml_tool_calls("This is just a normal response with no tool calls.")
        assert result is None

    def test_no_xml_with_angle_brackets(self):
        result = _parse_xml_tool_calls("The value is x < 10 and y > 5.")
        assert result is None

    def test_only_whitespace(self):
        result = _parse_xml_tool_calls("   \n\t  ")
        assert result is None

    # -- Format 1 takes priority over Format 2

    def test_format1_priority_over_format2(self):
        """If Format 1 matches, Format 2 is not checked (by design)."""
        content = (
            '<function=tool_a><parameter=x>1</parameter></function>'
            '<tool_call>{"name": "tool_b", "arguments": {}}</tool_call>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        # Format 1 found first, so Format 2 is skipped
        assert len(result) == 1
        assert result[0]["function"]["name"] == "tool_a"

    # -- Nested / tricky content

    def test_nested_angle_brackets_in_value(self):
        """Parameter value containing < and > but not matching the parameter regex."""
        content = '<function=store_memory><parameter=content>value with <b>html</b> inside</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "store_memory"


# ============================================================================
# SECTION 2: _estimate_tokens (purple.py)
# ============================================================================

class TestEstimateTokens:
    """Tests for the token estimation function in OllamaChat."""

    def _make_chat(self):
        """Create an OllamaChat with mocked dependencies."""
        mgr = MagicMock(spec=MCPToolManager)
        mgr.ollama_tools = []
        with patch("purple.load_identity", return_value="test system prompt"):
            with patch("purple.httpx.AsyncClient"):
                chat = OllamaChat(mgr)
        return chat

    def test_empty_messages(self):
        chat = self._make_chat()
        assert chat._estimate_tokens([]) == 0

    def test_single_short_message(self):
        chat = self._make_chat()
        messages = [{"role": "user", "content": "hello"}]
        # "hello" = 5 chars -> 5 // 4 = 1
        assert chat._estimate_tokens(messages) == 1

    def test_multiple_messages(self):
        chat = self._make_chat()
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 200},
        ]
        # (100 + 200) // 4 = 75
        assert chat._estimate_tokens(messages) == 75

    def test_message_with_no_content_key(self):
        chat = self._make_chat()
        messages = [{"role": "tool"}]
        # No "content" key -> defaults to "" -> len 0
        assert chat._estimate_tokens(messages) == 0

    def test_message_with_empty_content(self):
        chat = self._make_chat()
        messages = [{"role": "user", "content": ""}]
        assert chat._estimate_tokens(messages) == 0

    def test_large_message(self):
        chat = self._make_chat()
        messages = [{"role": "user", "content": "x" * 10000}]
        assert chat._estimate_tokens(messages) == 2500

    def test_mixed_content_and_no_content(self):
        chat = self._make_chat()
        messages = [
            {"role": "user", "content": "a" * 40},
            {"role": "tool"},
            {"role": "assistant", "content": "b" * 80},
        ]
        # (40 + 0 + 80) // 4 = 30
        assert chat._estimate_tokens(messages) == 30

    def test_exact_divisibility(self):
        chat = self._make_chat()
        messages = [{"role": "user", "content": "abcd"}]
        # 4 chars / 4 = 1 token exactly
        assert chat._estimate_tokens(messages) == 1

    def test_not_divisible(self):
        chat = self._make_chat()
        messages = [{"role": "user", "content": "abc"}]
        # 3 chars // 4 = 0 (integer division)
        assert chat._estimate_tokens(messages) == 0


# ============================================================================
# SECTION 3: _build_messages context trimming (purple.py)
# ============================================================================

class TestBuildMessages:
    """Tests for message list construction and context trimming."""

    def _make_chat(self, system_prompt="sys"):
        mgr = MagicMock(spec=MCPToolManager)
        mgr.ollama_tools = []
        with patch("purple.load_identity", return_value=system_prompt):
            with patch("purple.httpx.AsyncClient"):
                chat = OllamaChat(mgr)
        return chat

    def test_empty_history(self):
        chat = self._make_chat()
        msgs = chat._build_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "sys"

    def test_under_budget_no_trimming(self):
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        msgs = chat._build_messages()
        assert len(msgs) == 3  # system + 2 history
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_over_budget_trims_from_front(self):
        chat = self._make_chat()
        # MAX_HISTORY_TOKENS = 50000, each char ~0.25 tokens
        # Need total > 200000 chars so estimate > 50000 tokens
        chat._messages = [
            {"role": "user", "content": "x" * 80000},       # 20000 tokens
            {"role": "assistant", "content": "y" * 80000},   # 20000 tokens
            {"role": "user", "content": "z" * 80000},        # 20000 tokens -- total 60000
            {"role": "assistant", "content": "w" * 40000},   # 10000 tokens
        ]
        msgs = chat._build_messages()
        # After removing first msg: y(20k) + z(20k) + w(10k) = 50k, not > 50k, stops
        assert msgs[0]["role"] == "system"
        remaining_contents = [m.get("content", "") for m in msgs[1:]]
        assert "x" * 80000 not in remaining_contents

    def test_never_trims_below_2_messages(self):
        chat = self._make_chat()
        # With 3 messages all huge, it trims until len <= 2
        chat._messages = [
            {"role": "user", "content": "a" * 400000},
            {"role": "assistant", "content": "b" * 400000},
            {"role": "user", "content": "c" * 400000},
        ]
        msgs = chat._build_messages()
        history = msgs[1:]
        assert len(history) >= 2

    def test_orphaned_tool_responses_removed(self):
        """If trimming leaves a 'tool' message at the front, it should be removed."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "assistant", "content": "a" * 300000, "tool_calls": [{"function": {"name": "t"}}]},
            {"role": "tool", "content": "result of tool"},
            {"role": "user", "content": "b" * 100000},
            {"role": "assistant", "content": "c" * 40000},
        ]
        msgs = chat._build_messages()
        history = msgs[1:]
        if history:
            assert history[0]["role"] != "tool", "Orphaned tool response should be removed"

    def test_system_prompt_always_first(self):
        chat = self._make_chat(system_prompt="Identity prompt here")
        chat._messages = [
            {"role": "user", "content": "test"},
        ]
        msgs = chat._build_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Identity prompt here"

    def test_history_ordering_preserved(self):
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        msgs = chat._build_messages()
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert msgs[1]["content"] == "first"
        assert msgs[3]["content"] == "third"

    def test_single_huge_message_kept(self):
        """A single message over budget should be kept (can't trim below len > 2 with 1)."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "z" * 1000000},
        ]
        msgs = chat._build_messages()
        assert len(msgs) == 2
        assert msgs[1]["content"] == "z" * 1000000

    def test_two_messages_both_huge(self):
        """With exactly 2 messages, loop condition len(history) > 2 is False immediately."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "a" * 500000},
            {"role": "assistant", "content": "b" * 500000},
        ]
        msgs = chat._build_messages()
        assert len(msgs) == 3

    def test_multiple_consecutive_tool_messages_cleaned(self):
        """Multiple orphaned tool messages at the front should all be removed."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "tool", "content": "orphan1"},
            {"role": "tool", "content": "orphan2"},
            {"role": "user", "content": "real message"},
        ]
        msgs = chat._build_messages()
        history = msgs[1:]
        assert history[0]["role"] == "user"
        assert len(history) == 1

    def test_tool_message_not_at_front_is_kept(self):
        """Tool messages NOT at position 0 of history should be preserved."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "calling tool", "tool_calls": [{}]},
            {"role": "tool", "content": "tool result"},
            {"role": "assistant", "content": "here's the answer"},
        ]
        msgs = chat._build_messages()
        roles = [m["role"] for m in msgs]
        assert "tool" in roles  # tool message preserved since it's not at front


# ============================================================================
# SECTION 4: _mcp_to_ollama_tool (purple.py)
# ============================================================================

class TestMcpToOllamaTool:
    """Tests for MCP-to-Ollama tool schema conversion."""

    def _make_manager(self):
        return MCPToolManager()

    def test_full_schema(self):
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "store_memory"
        tool.description = "Store a memory"
        tool.inputSchema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["content", "type"],
        }
        result = mgr._mcp_to_ollama_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "store_memory"
        assert result["function"]["description"] == "Store a memory"
        assert result["function"]["parameters"]["type"] == "object"
        assert "content" in result["function"]["parameters"]["properties"]

    def test_empty_dict_schema(self):
        """Empty dict {} is falsy in Python, so it falls to the default."""
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "list_recent"
        tool.description = "List recent memories"
        tool.inputSchema = {}
        result = mgr._mcp_to_ollama_tool(tool)
        # {} is falsy, so the code takes the else branch: {"type": "object", "properties": {}}
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_none_schema_uses_default(self):
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "simple_tool"
        tool.description = "A simple tool"
        tool.inputSchema = None
        result = mgr._mcp_to_ollama_tool(tool)
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_no_description(self):
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "nodesc"
        tool.description = None
        tool.inputSchema = {"type": "object", "properties": {}}
        result = mgr._mcp_to_ollama_tool(tool)
        assert result["function"]["description"] == ""

    def test_complex_nested_schema(self):
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "create_excel"
        tool.description = "Create an Excel file"
        tool.inputSchema = {
            "type": "object",
            "properties": {
                "sheets": {
                    "type": "string",
                    "description": "JSON array of sheet configs",
                },
                "filename": {"type": "string"},
            },
            "required": ["sheets", "filename"],
        }
        result = mgr._mcp_to_ollama_tool(tool)
        assert result["function"]["name"] == "create_excel"
        assert "sheets" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["required"] == ["sheets", "filename"]

    def test_output_structure(self):
        """Verify the exact structure Ollama expects."""
        mgr = self._make_manager()
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.inputSchema = {"type": "object", "properties": {}}
        result = mgr._mcp_to_ollama_tool(tool)
        assert set(result.keys()) == {"type", "function"}
        assert set(result["function"].keys()) == {"name", "description", "parameters"}


# ============================================================================
# SECTION 5: _validate_path (docs/server.py)
# ============================================================================

class TestValidatePath:
    """Tests for path validation in the docs server."""

    def test_path_inside_home(self):
        path = Path.home() / "Documents" / "test.xlsx"
        result = _validate_path(path, "test")
        assert result is None  # Valid

    def test_path_inside_tmp(self):
        path = Path("/tmp/test_output/file.xlsx")
        result = _validate_path(path, "test")
        assert result is None  # Valid

    def test_path_outside_permitted_roots(self):
        path = Path("/etc/passwd")
        result = _validate_path(path, "read")
        assert result is not None
        assert "Error" in result
        assert "blocked" in result

    def test_path_var_directory(self):
        path = Path("/var/log/system.log")
        result = _validate_path(path, "read")
        assert result is not None
        assert "Error" in result

    def test_path_traversal_with_dotdot(self):
        """Path with .. that resolves outside permitted roots should fail."""
        path = Path.home() / ".." / ".." / "etc" / "passwd"
        result = _validate_path(path, "read")
        assert result is not None
        assert "Error" in result

    def test_path_traversal_staying_inside(self):
        """Path with .. that still resolves inside home should pass."""
        path = Path.home() / "Documents" / ".." / "Desktop" / "file.txt"
        result = _validate_path(path, "read")
        assert result is None  # Still under home

    def test_home_dir_itself(self):
        path = Path.home()
        result = _validate_path(path, "list")
        assert result is None  # Home dir itself should be allowed (resolved == root)

    def test_tmp_dir_itself(self):
        path = Path("/tmp")
        result = _validate_path(path, "list")
        assert result is None

    def test_root_directory(self):
        path = Path("/")
        result = _validate_path(path, "list")
        assert result is not None
        assert "Error" in result

    def test_usr_local_path(self):
        path = Path("/usr/local/bin/something")
        result = _validate_path(path, "read")
        assert result is not None
        assert "Error" in result

    def test_operation_name_in_error(self):
        """The operation name should appear in the error message."""
        path = Path("/etc/shadow")
        result = _validate_path(path, "read_secret")
        assert result is not None
        assert "read_secret" in result

    def test_deeply_nested_home_path(self):
        path = Path.home() / "a" / "b" / "c" / "d" / "e" / "file.txt"
        result = _validate_path(path, "write")
        assert result is None


# ============================================================================
# SECTION 6: _convert_value (docs/server.py)
# ============================================================================

class TestConvertValue:
    """Tests for value conversion in the docs server."""

    def test_integer_string(self):
        assert _convert_value("42") == 42
        assert isinstance(_convert_value("42"), int)

    def test_negative_integer_string(self):
        assert _convert_value("-7") == -7
        assert isinstance(_convert_value("-7"), int)

    def test_float_string(self):
        result = _convert_value("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_leading_zero_stays_string(self):
        """Leading zero values (zip codes, IDs) must remain strings."""
        assert _convert_value("007") == "007"
        assert isinstance(_convert_value("007"), str)

    def test_leading_zero_with_dot_converts_to_float(self):
        """0.5 has a leading zero but also a dot, so it converts to float."""
        result = _convert_value("0.5")
        assert result == 0.5
        assert isinstance(result, float)

    def test_single_zero_converts_to_int(self):
        """The string '0' should NOT be treated as leading-zero (value == '0')."""
        result = _convert_value("0")
        assert result == 0
        assert isinstance(result, int)

    def test_nan_string_stays_string(self):
        assert _convert_value("nan") == "nan"
        assert isinstance(_convert_value("nan"), str)

    def test_inf_string_stays_string(self):
        assert _convert_value("inf") == "inf"
        assert isinstance(_convert_value("inf"), str)

    def test_negative_inf_string_stays_string(self):
        assert _convert_value("-inf") == "-inf"
        assert isinstance(_convert_value("-inf"), str)

    def test_list_becomes_string(self):
        result = _convert_value([1, 2, 3])
        assert result == "[1, 2, 3]"
        assert isinstance(result, str)

    def test_dict_becomes_string(self):
        result = _convert_value({"key": "value"})
        assert isinstance(result, str)
        assert "key" in result

    def test_already_int_passthrough(self):
        assert _convert_value(42) == 42
        assert isinstance(_convert_value(42), int)

    def test_already_float_passthrough(self):
        assert _convert_value(3.14) == 3.14
        assert isinstance(_convert_value(3.14), float)

    def test_float_nan_becomes_string(self):
        result = _convert_value(float("nan"))
        assert isinstance(result, str)

    def test_float_inf_becomes_string(self):
        result = _convert_value(float("inf"))
        assert isinstance(result, str)

    def test_float_negative_inf_becomes_string(self):
        result = _convert_value(float("-inf"))
        assert isinstance(result, str)

    def test_empty_string(self):
        result = _convert_value("")
        assert result == ""
        assert isinstance(result, str)

    def test_plain_text_stays_string(self):
        assert _convert_value("hello world") == "hello world"

    def test_none_passthrough(self):
        """None is not str/int/float/list/dict -- falls through to return value."""
        assert _convert_value(None) is None

    def test_boolean_passthrough(self):
        """Booleans are instances of int in Python, so they pass the isinstance check."""
        result = _convert_value(True)
        assert result is True

    def test_zipcode_leading_zero(self):
        """Real-world zip code: 01234 should stay as string."""
        assert _convert_value("01234") == "01234"

    def test_large_integer_string(self):
        result = _convert_value("1234567890123")
        assert result == 1234567890123

    def test_scientific_notation(self):
        result = _convert_value("1.5e10")
        assert result == 1.5e10
        assert isinstance(result, float)

    def test_single_digit_string(self):
        assert _convert_value("5") == 5
        assert isinstance(_convert_value("5"), int)

    def test_negative_float_string(self):
        result = _convert_value("-2.5")
        assert result == -2.5
        assert isinstance(result, float)

    def test_string_with_spaces(self):
        """A string with spaces should not convert to a number."""
        assert _convert_value("hello world") == "hello world"

    def test_leading_zero_multi_digit(self):
        """Multiple leading zeros like account numbers."""
        assert _convert_value("00123") == "00123"
        assert isinstance(_convert_value("00123"), str)

    def test_zero_point_zero(self):
        """'0.0' has leading zero but also has a dot, should become float."""
        result = _convert_value("0.0")
        assert result == 0.0
        assert isinstance(result, float)


# ============================================================================
# SECTION 7: _resolve_output (docs/server.py)
# ============================================================================

class TestResolveOutput:
    """Tests for output directory resolution in docs server."""

    def test_valid_output_dir(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            result = _resolve_output(tmpdir, "excel")
            assert isinstance(result, Path)
            assert result.exists()
            assert "excel" in str(result)

    def test_subfolder_already_in_path(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            excel_dir = Path(tmpdir) / "excel"
            excel_dir.mkdir()
            result = _resolve_output(str(excel_dir), "excel")
            assert isinstance(result, Path)
            parts_lower = [p.lower() for p in result.parts]
            count = parts_lower.count("excel")
            assert count == 1, f"'excel' appears {count} times in {result}"

    def test_output_dir_outside_permitted_roots(self):
        result = _resolve_output("/etc/evil_output", "excel")
        assert isinstance(result, str)
        assert "Error" in result

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            target = str(Path(tmpdir) / "new_subdir")
            result = _resolve_output(target, "word")
            assert isinstance(result, Path)
            assert result.exists()
            assert "word" in str(result)

    def test_excel_subfolder_added(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            result = _resolve_output(tmpdir, "excel")
            assert isinstance(result, Path)
            assert result.name == "excel"

    def test_word_subfolder_added(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            result = _resolve_output(tmpdir, "word")
            assert isinstance(result, Path)
            assert result.name == "word"

    def test_powerpoint_subfolder_added(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            result = _resolve_output(tmpdir, "powerpoint")
            assert isinstance(result, Path)
            assert result.name == "powerpoint"

    def test_tmp_directory_valid(self):
        import shutil
        test_path = "/tmp/test_purple_output_resolve"
        try:
            result = _resolve_output(test_path, "excel")
            assert isinstance(result, Path)
        finally:
            if Path(test_path).exists():
                shutil.rmtree(test_path)

    def test_tilde_expansion(self):
        import shutil
        try:
            result = _resolve_output("~/test_purple_tilde_resolve", "excel")
            assert isinstance(result, Path)
            assert str(Path.home()) in str(result)
        finally:
            expanded = Path.home() / "test_purple_tilde_resolve"
            if expanded.exists():
                shutil.rmtree(str(expanded))


# ============================================================================
# SECTION 8: Memory server tests (memory/server.py)
# ============================================================================

class TestMemoryServer:
    """Tests for the memory server using a temporary SQLite database.

    We override the global DB_PATH so that all operations hit an
    isolated temp database instead of the real purple.db.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        """Replace DB_PATH with a temporary database for each test."""
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized

        temp_db = tmp_path / "test_memory.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    # -- _store_memory tests --

    def test_store_memory_valid_fact(self):
        result = mem_server._store_memory("Python is great", "fact")
        assert "Stored memory #" in result
        assert "[fact]" in result

    def test_store_memory_valid_preference(self):
        result = mem_server._store_memory("Prefers dark mode", "preference")
        assert "Stored memory #" in result
        assert "[preference]" in result

    def test_store_memory_valid_experience(self):
        result = mem_server._store_memory("Debugging went well", "experience")
        assert "[experience]" in result

    def test_store_memory_valid_correction(self):
        result = mem_server._store_memory("Actually 30B not 70B", "correction")
        assert "[correction]" in result

    def test_store_memory_with_tags(self):
        result = mem_server._store_memory("Tagged note", "fact", tags=["ai", "model"])
        assert "Stored memory #" in result

    def test_store_memory_invalid_type(self):
        result = mem_server._store_memory("bad type", "opinion")
        assert "Error" in result
        assert "opinion" in result

    def test_store_memory_invalid_type_empty(self):
        result = mem_server._store_memory("bad type", "")
        assert "Error" in result

    def test_store_memory_increments_id(self):
        r1 = mem_server._store_memory("First", "fact")
        r2 = mem_server._store_memory("Second", "fact")
        id1 = int(r1.split("#")[1].split(":")[0])
        id2 = int(r2.split("#")[1].split(":")[0])
        assert id2 > id1

    # -- _recall_memories tests --

    def test_recall_memories_keyword_match(self):
        mem_server._store_memory("Ollama runs on port 11434", "fact")
        result = mem_server._recall_memories("Ollama")
        assert "Found 1 memories" in result
        assert "11434" in result

    def test_recall_memories_no_match(self):
        mem_server._store_memory("Purple is local AI", "fact")
        result = mem_server._recall_memories("nonexistent_query_xyz")
        assert "No memories found" in result

    def test_recall_memories_type_filter(self):
        mem_server._store_memory("Fact about X", "fact")
        mem_server._store_memory("Preference about X", "preference")
        result = mem_server._recall_memories("about X", type="fact")
        assert "Found 1 memories" in result
        assert "fact" in result.lower()

    def test_recall_memories_type_filter_excludes(self):
        mem_server._store_memory("Data point A", "fact")
        mem_server._store_memory("Data point B", "experience")
        result = mem_server._recall_memories("Data point", type="correction")
        assert "No memories found" in result

    def test_recall_memories_limit(self):
        for i in range(10):
            mem_server._store_memory(f"Item number {i}", "fact")
        result = mem_server._recall_memories("Item number", limit=3)
        assert "Found 3 memories" in result

    def test_recall_memories_case_insensitive_like(self):
        """SQLite LIKE is case-insensitive for ASCII by default."""
        mem_server._store_memory("Purple AI system", "fact")
        result = mem_server._recall_memories("purple")
        assert "Found 1 memories" in result

    def test_recall_memories_partial_match(self):
        mem_server._store_memory("The quick brown fox jumps", "fact")
        result = mem_server._recall_memories("brown fox")
        assert "Found 1 memories" in result

    # -- _forget_memory tests --

    def test_forget_memory_valid_id(self):
        store_result = mem_server._store_memory("Temporary note", "fact")
        mem_id = int(store_result.split("#")[1].split(":")[0])
        result = mem_server._forget_memory(mem_id, "no longer needed")
        assert "Deleted memory #" in result
        assert "no longer needed" in result

    def test_forget_memory_actually_deletes(self):
        mem_server._store_memory("Will be deleted A", "fact")
        store_result = mem_server._store_memory("Will be deleted B", "fact")
        mem_id = int(store_result.split("#")[1].split(":")[0])
        mem_server._forget_memory(mem_id, "cleanup")
        # Only A should remain
        result = mem_server._recall_memories("Will be deleted")
        assert "Found 1 memories" in result
        assert "deleted A" in result

    def test_forget_memory_invalid_id(self):
        result = mem_server._forget_memory(99999, "test")
        assert "Error" in result
        assert "not found" in result

    def test_forget_memory_negative_id(self):
        result = mem_server._forget_memory(-1, "test")
        assert "Error" in result
        assert "not found" in result

    # -- _list_recent tests --

    def test_list_recent_with_entries(self):
        mem_server._store_memory("Recent note", "fact")
        result = mem_server._list_recent(hours=1)
        assert "Recent note" in result

    def test_list_recent_no_entries(self):
        result = mem_server._list_recent(hours=1)
        assert "No memories in the last" in result

    def test_list_recent_type_filter(self):
        mem_server._store_memory("Fact entry", "fact")
        mem_server._store_memory("Pref entry", "preference")
        result = mem_server._list_recent(hours=1, type="fact")
        assert "Fact entry" in result
        assert "Pref entry" not in result

    def test_list_recent_type_filter_no_match(self):
        mem_server._store_memory("Only a fact", "fact")
        result = mem_server._list_recent(hours=1, type="correction")
        assert "No memories in the last" in result

    def test_list_recent_large_window(self):
        mem_server._store_memory("Old-ish note", "experience")
        result = mem_server._list_recent(hours=8760)  # 1 year
        assert "Old-ish note" in result

    # -- Edge cases --

    def test_store_and_recall_special_characters(self):
        mem_server._store_memory("Contains 'quotes' and \"double quotes\" and % percent", "fact")
        result = mem_server._recall_memories("quotes")
        assert "Found 1 memories" in result

    def test_store_empty_content(self):
        """Empty content should still store (no NOT NULL violation for empty string)."""
        result = mem_server._store_memory("", "fact")
        assert "Stored memory #" in result

    def test_store_very_long_content(self):
        long_content = "x" * 10000
        result = mem_server._store_memory(long_content, "fact")
        assert "Stored memory #" in result
        recall = mem_server._recall_memories("x" * 50)
        assert "Found" in recall

    def test_format_row_truncation(self):
        """The _format_row function should truncate long content."""
        mem_server._store_memory("A" * 1000, "fact")
        result = mem_server._list_recent(hours=1)
        # Default truncate in _list_recent uses _format_row(r, truncate=200)
        # But _list_recent calls _format_row(r) which defaults truncate=200
        assert len(result) < 1500

    def test_store_with_sql_injection_attempt(self):
        """SQL injection via content should be safely parameterized."""
        result = mem_server._store_memory("'; DROP TABLE memories; --", "fact")
        assert "Stored memory #" in result
        # Table should still exist
        recall = mem_server._recall_memories("DROP TABLE")
        assert "Found 1 memories" in recall

    def test_recall_with_percent_in_query(self):
        """% in query is a LIKE wildcard -- should still work without crashing."""
        mem_server._store_memory("Test data point", "fact")
        result = mem_server._recall_memories("%data%")
        # The LIKE pattern becomes %{query}% = %%data%% which still matches
        assert "Found" in result or "No memories" in result  # Either is fine, no crash


# ============================================================================
# SECTION 9: Memory server -- database schema and integrity
# ============================================================================

class TestMemoryServerSchema:
    """Tests verifying the database schema and constraints."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_schema.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_table_created(self):
        conn = mem_server.get_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_wal_mode_enabled(self):
        conn = mem_server.get_db()
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        conn.close()

    def test_type_constraint_enforced_at_db_level(self):
        """The CHECK constraint should reject invalid types at the SQL level."""
        conn = mem_server.get_db()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO memories (content, type, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("test", "invalid_type", "[]", now, now),
            )
        conn.close()

    def test_indexes_created(self):
        conn = mem_server.get_db()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
        ).fetchall()
        index_names = [r[0] for r in indexes]
        assert "idx_memories_type" in index_names
        assert "idx_memories_created" in index_names
        conn.close()

    def test_initialized_flag_prevents_double_init(self):
        assert mem_server._initialized is False
        mem_server.get_db().close()
        assert mem_server._initialized is True
        # Second call should not fail
        mem_server.get_db().close()


# ============================================================================
# SECTION 10: MCPToolManager error paths
# ============================================================================

class TestMCPToolManagerCallTool:
    """Tests for tool call error handling without real MCP servers."""

    @pytest.fixture
    def manager(self):
        return MCPToolManager()

    def test_call_unknown_tool(self, manager):
        import asyncio
        result = asyncio.run(manager.call_tool("nonexistent_tool", {}))
        assert "Error" in result
        assert "not found" in result

    def test_call_tool_server_disconnected(self, manager):
        import asyncio
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        manager._tool_map["test_tool"] = ("fake_server", mock_tool)
        # Client exists in dict but is None
        manager._clients["fake_server"] = None
        result = asyncio.run(manager.call_tool("test_tool", {}))
        assert "Error" in result
        assert "not connected" in result

    def test_tool_names_property(self, manager):
        assert manager.tool_names == []
        mock_tool = MagicMock()
        manager._tool_map["my_tool"] = ("server", mock_tool)
        assert "my_tool" in manager.tool_names

    def test_ollama_tools_property(self, manager):
        assert manager.ollama_tools == []


# ============================================================================
# Main -- allow running via python test_purple.py
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
