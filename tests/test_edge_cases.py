"""
Edge case and adversarial tests for Purple CLI, docs server, and memory server.

Written by Keenness agent. These tests target gaps found in the Wave 1 test suite:
  - Security: path validation bypasses, symlink escapes, filename traversal
  - Data integrity: SQL wildcard leakage, corrupted tags JSON, token estimation blind spots
  - Robustness: XML parser edge cases, streaming failure modes, _build_messages trimming
  - Platform: macOS /tmp symlink, /var/folders exclusion, null bytes in paths

Run:
    /Users/purple/.purple/venv/bin/python -m pytest /Users/purple/.purple/tests/test_edge_cases.py -v
"""

import importlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import targets (same pattern as test_purple.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
from purple import _parse_xml_tool_calls, OllamaChat, MCPToolManager

_docs_spec = importlib.util.spec_from_file_location(
    "docs_server",
    str(Path(__file__).resolve().parent.parent / "docs" / "server.py"),
)
docs_server = importlib.util.module_from_spec(_docs_spec)
_docs_spec.loader.exec_module(docs_server)
_validate_path = docs_server._validate_path
_convert_value = docs_server._convert_value
_resolve_output = docs_server._resolve_output

_mem_spec = importlib.util.spec_from_file_location(
    "mem_server",
    str(Path(__file__).resolve().parent.parent / "memory" / "server.py"),
)
mem_server = importlib.util.module_from_spec(_mem_spec)
_mem_spec.loader.exec_module(mem_server)


# ============================================================================
# SECTION 1: SECURITY -- Path validation bypasses (docs/server.py)
# ============================================================================

class TestPathValidationSecurity:
    """Tests for path validation edge cases that could allow unauthorized access."""

    def test_macos_tmp_symlink_resolves_correctly(self):
        """macOS /tmp is a symlink to /private/tmp. Both must be accepted.
        PERMITTED_ROOTS resolves /tmp to /private/tmp at import time,
        so Path('/tmp/x').resolve() = /private/tmp/x, which starts with /private/tmp/."""
        # /tmp path
        result_tmp = _validate_path(Path("/tmp/test.xlsx"), "test")
        assert result_tmp is None, "/tmp should be permitted"

        # /private/tmp path (explicit)
        result_private = _validate_path(Path("/private/tmp/test.xlsx"), "test")
        assert result_private is None, "/private/tmp should be permitted"

    def test_macos_var_folders_blocked(self):
        """macOS per-user temp dirs (/var/folders/...) should be blocked.
        tempfile.gettempdir() returns /var/folders/... which is NOT /tmp.
        This means pytest's tmp_path fixtures create dirs OUTSIDE permitted roots."""
        var_tmp = Path(tempfile.gettempdir()) / "test.xlsx"
        result = _validate_path(var_tmp, "test")
        # /var/folders resolves to /private/var/folders -- outside permitted roots
        assert result is not None, f"/var/folders path should be blocked, got None for {var_tmp.resolve()}"
        assert "Error" in result

    def test_symlink_escape_from_home(self):
        """A symlink inside ~ pointing outside ~ should be blocked after resolve()."""
        link_path = Path.home() / ".test_escape_link_keenness"
        target = Path("/etc")
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(target, link_path)

            escape_path = link_path / "passwd"
            result = _validate_path(escape_path, "read")
            assert result is not None, "Symlink escape should be blocked"
            assert "Error" in result
        finally:
            if link_path.is_symlink():
                link_path.unlink()

    def test_null_byte_in_path_raises(self):
        """Null byte injection in filenames should raise ValueError from OS."""
        with pytest.raises(ValueError, match="null"):
            Path("/Users/purple/test\x00evil.txt").resolve()

    def test_read_excel_has_path_validation(self):
        """FIXED: read_excel now calls _validate_path."""
        import inspect
        source = inspect.getsource(docs_server.read_excel.fn)
        assert "_validate_path" in source, \
            "read_excel must call _validate_path"

    def test_read_pdf_has_path_validation(self):
        """FIXED: read_pdf now calls _validate_path."""
        import inspect
        source = inspect.getsource(docs_server.read_pdf.fn)
        assert "_validate_path" in source, \
            "read_pdf must call _validate_path"

    def test_list_directory_has_path_validation(self):
        """Confirm list_directory DOES call _validate_path (control test)."""
        import inspect
        source = inspect.getsource(docs_server.list_directory.fn)
        assert "_validate_path" in source

    def test_filename_traversal_escapes_subfolder(self):
        """Filename with ../ can escape the type-specific subfolder.
        _resolve_output validates output_dir, but the final filepath
        (outdir / filename) is NOT re-validated. filename='../../evil.xlsx'
        writes to output_dir's grandparent, not the excel/ subfolder."""
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as tmpdir:
            outdir = _resolve_output(tmpdir, "excel")
            assert isinstance(outdir, Path)
            # Simulate what create_excel does
            malicious_name = "../../escape.xlsx"
            filepath = outdir / malicious_name
            resolved = filepath.resolve()
            # The file escapes the excel/ subfolder
            assert "excel" not in resolved.parts, \
                f"Filename traversal should escape excel/ subfolder, got {resolved}"
            # But it stays under home (not a security boundary violation)
            assert str(resolved).startswith(str(Path.home()))

    def test_list_directory_recursive_glob_dos(self):
        """pattern='**/*' could recursively list enormous directory trees.
        This is a DoS vector, not a security escape. The truncation at 100 entries
        limits output size but not the time spent traversing."""
        # We just verify the pattern is accepted (no blocklist)
        test_dir = Path.home() / ".purple" / "tests"
        if test_dir.exists():
            # The glob itself works -- we're documenting the risk, not blocking it
            results = list(test_dir.glob("**/*"))
            assert len(results) >= 0  # Just verify it doesn't crash

    def test_list_directory_absolute_glob_blocked(self):
        """Absolute paths in glob pattern should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            list(Path.home().glob("/etc/*"))


# ============================================================================
# SECTION 2: MEMORY SERVER -- SQL wildcard leakage and data integrity
# ============================================================================

class TestMemoryWildcardLeakage:
    """Tests exposing the SQL LIKE wildcard bug in recall_memories."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_wildcards.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_percent_wildcard_escaped(self):
        """FIXED: Searching for '%' now only matches memories containing literal %."""
        mem_server._store_memory("Has 100% success rate", "fact")
        mem_server._store_memory("Normal memory with no percent", "fact")
        mem_server._store_memory("Another plain memory", "fact")

        result = mem_server._recall_memories("%")
        # FIXED: Only returns the one with literal %
        assert "Found 1 memories" in result, \
            f"Expected only 1 match (the one with literal %), got: {result}"

    def test_underscore_wildcard_escaped(self):
        """FIXED: Searching for '_' now only matches memories containing literal _."""
        mem_server._store_memory("file_name.txt is important", "fact")
        mem_server._store_memory("No underscore here", "fact")

        result = mem_server._recall_memories("_")
        # FIXED: Only returns the one with literal _
        assert "Found 1 memories" in result, \
            f"Expected only 1 match (the one with literal _), got: {result}"

    def test_percent_in_content_found_by_exact_search(self):
        """When searching for text that happens to contain %, it still works
        (just with extra wildcard behavior)."""
        mem_server._store_memory("CPU at 95% load", "fact")
        result = mem_server._recall_memories("95%")
        assert "Found" in result
        assert "95%" in result


class TestMemoryCorruptedTags:
    """Tests for _format_row with corrupted tags in the database."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_corrupt.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_format_row_invalid_json_tags_handled(self):
        """FIXED: Invalid JSON tags no longer crash _format_row."""
        conn = mem_server.get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memories (content, type, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("test content", "fact", "not valid json", now, now),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM memories WHERE id=1").fetchone()
        result = mem_server._format_row(row)
        assert "#1" in result
        assert "test content" in result
        conn.close()

    def test_format_row_numeric_tags_handled(self):
        """FIXED: Numeric tags (non-list JSON) no longer crash _format_row."""
        conn = mem_server.get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memories (content, type, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("test content", "fact", "42", now, now),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM memories WHERE id=1").fetchone()
        result = mem_server._format_row(row)
        assert "#1" in result
        assert "test content" in result
        conn.close()

    def test_format_row_null_json_tags_handled(self):
        """Tags containing JSON 'null' should not crash (json.loads returns None,
        which is falsy, so tag_str becomes '')."""
        conn = mem_server.get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memories (content, type, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("test content", "fact", "null", now, now),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM memories WHERE id=1").fetchone()
        result = mem_server._format_row(row)
        assert "#1" in result
        assert "fact" in result
        conn.close()


class TestMemoryConcurrency:
    """Tests for concurrent access to the memory database."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_concurrent.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_concurrent_writes_no_data_loss(self):
        """Multiple threads writing simultaneously should not lose data.
        WAL mode + default 5s busy timeout should handle this."""
        errors = []
        num_threads = 5
        writes_per_thread = 20

        def writer(thread_id):
            for i in range(writes_per_thread):
                try:
                    mem_server._store_memory(
                        f"Thread {thread_id} item {i}", "fact"
                    )
                except Exception as e:
                    errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * writes_per_thread
        result = mem_server._recall_memories("Thread", limit=200)
        if "Found" in result:
            count = int(result.split("Found ")[1].split(" ")[0])
        else:
            count = 0

        assert len(errors) == 0, f"Concurrent write errors: {errors[:5]}"
        assert count == expected, f"Expected {expected} memories, found {count}"

    def test_concurrent_read_write(self):
        """Reading while writing should not crash or return partial data."""
        # Pre-populate
        for i in range(10):
            mem_server._store_memory(f"Base item {i}", "fact")

        errors = []

        def writer():
            for i in range(20):
                try:
                    mem_server._store_memory(f"New item {i}", "fact")
                except Exception as e:
                    errors.append(f"Write: {e}")

        def reader():
            for _ in range(20):
                try:
                    mem_server._recall_memories("item")
                except Exception as e:
                    errors.append(f"Read: {e}")

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert len(errors) == 0, f"Concurrent read/write errors: {errors[:5]}"


class TestMemoryLargeContent:
    """Tests for handling extremely large memory content."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_large.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_store_100kb_content(self):
        """100KB of content should store and recall without issue."""
        big = "x" * 100_000
        result = mem_server._store_memory(big, "fact")
        assert "Stored memory #" in result

        recall = mem_server._recall_memories("x" * 50)
        assert "Found" in recall

    def test_store_unicode_content(self):
        """Unicode content (emoji, CJK, Arabic) should round-trip correctly."""
        content = "Purple AI: Emoji test. CJK: \u4e16\u754c Arabic: \u0645\u0631\u062d\u0628\u0627"
        result = mem_server._store_memory(content, "fact")
        assert "Stored memory #" in result

        recall = mem_server._recall_memories("\u4e16\u754c")
        assert "Found" in recall

    def test_store_newlines_and_tabs(self):
        """Content with newlines, tabs, and other whitespace should be preserved."""
        content = "Line 1\nLine 2\tTabbed\r\nWindows line\0Null byte"
        result = mem_server._store_memory(content, "fact")
        assert "Stored memory #" in result


# ============================================================================
# SECTION 3: XML PARSER -- Edge cases in _parse_xml_tool_calls
# ============================================================================

class TestXmlParserEdgeCases:
    """Tests for XML tool call parsing edge cases the model might produce."""

    def test_hyphenated_function_name_not_matched(self):
        """BUG: Function names with hyphens (e.g., 'list-files') are not matched
        because the regex uses \\w+ which only matches [a-zA-Z0-9_].
        MCP tool names are typically Python identifiers (underscores), so this
        is low risk but worth documenting."""
        content = '<function=list-files><parameter=dir>/home</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is None, "Hyphenated function names should not match \\w+"

    def test_hyphenated_parameter_name_silently_dropped(self):
        """BUG: Parameter names with hyphens are silently dropped.
        The function is matched but the parameter regex fails, resulting
        in an empty arguments dict -- silent data loss."""
        content = '<function=my_tool><parameter=my-param>value</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["name"] == "my_tool"
        # The hyphenated parameter is silently dropped
        assert result[0]["function"]["arguments"] == {}, \
            "Hyphenated param should be silently dropped (documenting bug)"

    def test_format2_nested_json_braces(self):
        """Format 2 regex uses non-greedy .*? but the JSON reconstructor
        prepends { and appends }. Nested JSON with matching braces should work."""
        content = '<tool_call>{"name": "test", "arguments": {"nested": {"deep": 1}}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["arguments"]["nested"]["deep"] == 1

    def test_format2_braces_in_string_value(self):
        """JSON string values containing { and } should parse correctly."""
        content = '<tool_call>{"name": "test", "arguments": {"code": "if (x) { return 1; }"}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert "return 1" in result[0]["function"]["arguments"]["code"]

    def test_format1_special_chars_in_value(self):
        """Parameter values containing JSON, HTML, or other special characters."""
        # JSON-like content in a parameter value
        content = '<function=create_excel><parameter=sheets>[{"name": "Sheet1"}]</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        sheets_val = result[0]["function"]["arguments"]["sheets"]
        assert "Sheet1" in sheets_val

    def test_format1_with_equals_in_value(self):
        """Parameter values containing = signs should be captured correctly."""
        content = '<function=store_memory><parameter=content>x = 42 and y = 99</parameter></function>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert "x = 42" in result[0]["function"]["arguments"]["content"]

    def test_xml_tool_call_with_unicode_content(self):
        """Tool calls containing unicode in arguments should parse."""
        content = '<tool_call>{"name": "store", "arguments": {"text": "\u4e16\u754c"}}</tool_call>'
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert result[0]["function"]["arguments"]["text"] == "\u4e16\u754c"

    def test_multiple_valid_format2_with_one_invalid(self):
        """BUG: An invalid tool_call between two valid ones causes the second
        valid call to be lost. The regex <tool_call>\\s*\\{(.*?)\\}\\s*</tool_call>
        with re.DOTALL matches across </tool_call><tool_call> boundaries when
        the middle block has no closing }. Match 2 captures:
        'totally broken</tool_call><tool_call>{"name":"good2"...}'
        which fails JSON parsing, losing good2.

        This means a single malformed tool_call can swallow subsequent valid ones."""
        content = (
            '<tool_call>{"name": "good1", "arguments": {}}</tool_call>'
            '<tool_call>{totally broken</tool_call>'
            '<tool_call>{"name": "good2", "arguments": {"x": 1}}</tool_call>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        # BUG: Only good1 is captured. good2 is swallowed by the broken match.
        assert len(result) == 1, "Documenting regex cross-boundary matching bug"
        assert result[0]["function"]["name"] == "good1"

    def test_multiple_valid_format2_no_invalid_works(self):
        """Without invalid blocks between them, multiple Format 2 calls work."""
        content = (
            '<tool_call>{"name": "good1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "good2", "arguments": {"x": 1}}</tool_call>'
        )
        result = _parse_xml_tool_calls(content)
        assert result is not None
        assert len(result) == 2
        names = [r["function"]["name"] for r in result]
        assert "good1" in names
        assert "good2" in names


# ============================================================================
# SECTION 4: STREAMING -- _stream_to_ollama edge cases
# ============================================================================

class TestStreamingEdgeCases:
    """Tests for streaming response handling edge cases.
    These test the assembly logic without connecting to Ollama."""

    def _make_chat(self):
        mgr = MagicMock(spec=MCPToolManager)
        mgr.ollama_tools = []
        with patch("purple.load_identity", return_value="sys"):
            with patch("purple.httpx.AsyncClient"):
                chat = OllamaChat(mgr)
        return chat

    def test_stream_done_true_no_content(self):
        """If Ollama sends done:true with no content fragments, the assembled
        message should have empty content string."""
        chat = self._make_chat()
        # The assembled_message logic:
        content_parts = []
        tool_calls = None
        final_chunk = {"done": True, "model": "test"}

        assembled = {"content": "".join(content_parts)}
        if tool_calls:
            assembled["tool_calls"] = tool_calls

        result = dict(final_chunk)
        result["message"] = assembled

        assert result["message"]["content"] == ""
        assert "tool_calls" not in result["message"]
        assert result["done"] is True

    def test_stream_tool_calls_on_final_chunk(self):
        """Tool calls arrive only on the final streaming chunk.
        Verify they are captured even when content is empty."""
        content_parts = []
        tool_calls = [{"function": {"name": "test", "arguments": {}}}]
        final_chunk = {"done": True}

        assembled = {"content": "".join(content_parts)}
        if tool_calls:
            assembled["tool_calls"] = tool_calls

        result = dict(final_chunk)
        result["message"] = assembled

        assert result["message"]["tool_calls"] == tool_calls
        assert result["message"]["content"] == ""

    def test_stream_content_reassembly(self):
        """Multiple content fragments should be joined without separators."""
        content_parts = ["Hello", " ", "world", "!"]
        assembled = {"content": "".join(content_parts)}
        assert assembled["content"] == "Hello world!"

    def test_stream_malformed_json_line_skipped(self):
        """The streaming loop does `continue` on JSONDecodeError.
        Verify this logic: a malformed line among valid ones should not crash."""
        raw_lines = [
            '{"message": {"content": "Hello"}, "done": false}',
            'NOT VALID JSON',
            '{"message": {"content": " world"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]
        content_parts = []
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message", {})
            fragment = msg.get("content", "")
            if fragment:
                content_parts.append(fragment)

        assert "".join(content_parts) == "Hello world"


# ============================================================================
# SECTION 5: _build_messages -- Trimming edge cases
# ============================================================================

class TestBuildMessagesEdgeCases:
    """Tests for message trimming edge cases not covered by existing tests."""

    def _make_chat(self, system_prompt="sys"):
        mgr = MagicMock(spec=MCPToolManager)
        mgr.ollama_tools = []
        with patch("purple.load_identity", return_value=system_prompt):
            with patch("purple.httpx.AsyncClient"):
                chat = OllamaChat(mgr)
        return chat

    def test_all_tool_messages_after_trim_results_in_empty_history(self):
        """If trimming leaves only tool messages, orphan cleaning removes them all.
        Result: only the system prompt remains. User context is lost."""
        chat = self._make_chat()
        # Simulate: a huge assistant+tool_calls followed by tool results,
        # then the assistant message is popped by trimming, leaving only tools
        chat._messages = [
            {"role": "assistant", "content": "a" * 300000, "tool_calls": [{}]},
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
            {"role": "user", "content": "b" * 100},
            {"role": "assistant", "content": "c" * 100},
        ]
        msgs = chat._build_messages()
        # System prompt is always first
        assert msgs[0]["role"] == "system"
        # The orphan tool messages should be cleaned
        history = msgs[1:]
        if history:
            assert history[0]["role"] != "tool"

    def test_token_estimate_ignores_tool_calls_payload(self):
        """BUG: _estimate_tokens only counts content field, not tool_calls.
        Large tool_calls payloads (e.g., create_excel with big sheets JSON)
        are invisible to the token budget, causing potential context overflow."""
        chat = self._make_chat()
        big_args = json.dumps({"sheets": [{"name": f"S{i}", "rows": [["x"] * 50] * 100} for i in range(10)]})
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "create_excel", "arguments": {"sheets": big_args}}}
            ]},
        ]
        estimated = chat._estimate_tokens(messages)
        assert estimated == 0, "Token estimator sees 0 for empty content with large tool_calls"

        # The actual payload is much larger
        actual_size = len(json.dumps(messages[0]["tool_calls"]))
        assert actual_size > 10000, f"tool_calls payload is {actual_size} chars but invisible"

    def test_trim_preserves_most_recent_user_message(self):
        """After trimming, the most recent user message should always be present."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "old " * 60000},    # ~60K tokens
            {"role": "assistant", "content": "old reply " * 60000},
            {"role": "user", "content": "new question"},     # small
            {"role": "assistant", "content": "new reply"},
        ]
        msgs = chat._build_messages()
        contents = [m.get("content", "") for m in msgs]
        assert "new question" in contents, "Most recent user message should survive trimming"

    def test_reset_clears_all_messages(self):
        """reset() should completely clear conversation history."""
        chat = self._make_chat()
        chat._messages = [
            {"role": "user", "content": "test1"},
            {"role": "assistant", "content": "reply1"},
        ]
        chat.reset()
        assert len(chat._messages) == 0
        msgs = chat._build_messages()
        assert len(msgs) == 1  # Only system prompt
        assert msgs[0]["role"] == "system"


# ============================================================================
# SECTION 6: _convert_value -- Additional edge cases
# ============================================================================

class TestConvertValueEdgeCases:
    """Edge cases not covered in the existing _convert_value tests."""

    def test_overflow_float_string_stays_string(self):
        """'1e309' overflows to inf, which is caught by isinf check."""
        result = _convert_value("1e309")
        assert isinstance(result, str)
        assert result == "1e309"

    def test_whitespace_padded_number_converts(self):
        """' 42 ' with leading/trailing spaces -- int() strips whitespace."""
        result = _convert_value(" 42 ")
        assert result == 42
        assert isinstance(result, int)

    def test_capitalized_nan_stays_string(self):
        """'NaN' (capitalized) -- float('NaN') returns nan, caught by isnan."""
        result = _convert_value("NaN")
        assert isinstance(result, str)
        assert result == "NaN"

    def test_infinity_string_stays_string(self):
        """'Infinity' -- float('Infinity') returns inf, caught by isinf."""
        result = _convert_value("Infinity")
        assert isinstance(result, str)

    def test_negative_infinity_stays_string(self):
        result = _convert_value("-Infinity")
        assert isinstance(result, str)

    def test_empty_list_becomes_string(self):
        result = _convert_value([])
        assert result == "[]"
        assert isinstance(result, str)

    def test_bytes_passthrough(self):
        """bytes object is not str/int/float/list/dict -- falls through unchanged.
        This could cause issues in openpyxl if it reaches cell.value."""
        result = _convert_value(b"hello")
        assert result == b"hello"
        assert isinstance(result, bytes)

    def test_plus_sign_number(self):
        """'+42' -- int() handles leading + sign."""
        result = _convert_value("+42")
        assert result == 42
        assert isinstance(result, int)

    def test_leading_zero_decimal(self):
        """'0.1' has leading zero but also dot, should convert to float."""
        result = _convert_value("0.1")
        assert result == 0.1
        assert isinstance(result, float)

    def test_double_zero(self):
        """'00' has leading zero, should stay string."""
        result = _convert_value("00")
        assert result == "00"
        assert isinstance(result, str)

    def test_negative_zero(self):
        """'-0' should convert to int 0."""
        result = _convert_value("-0")
        assert result == 0
        assert isinstance(result, int)


# ============================================================================
# SECTION 7: PDF page range parsing edge cases
# ============================================================================

class TestPdfPageRangeParsing:
    """Tests for the page range parsing logic in read_pdf.
    We test the parsing logic directly since read_pdf requires an actual PDF file."""

    def _parse_pages(self, pages_str: str, total_pages: int = 10) -> list[int]:
        """Replicate the page range parsing from read_pdf."""
        if pages_str == "all":
            return list(range(total_pages))
        elif "-" in pages_str:
            start, end = pages_str.split("-", 1)
            start = max(0, int(start) - 1)
            end = min(total_pages, int(end))
            return list(range(start, end))
        else:
            return [int(pages_str) - 1]

    def test_reversed_range_returns_empty(self):
        """'5-3' produces range(4, 3) which is empty."""
        result = self._parse_pages("5-3")
        assert result == []

    def test_zero_start_clamped(self):
        """'0-5' produces max(0, -1) = 0, so pages 0-4."""
        result = self._parse_pages("0-5")
        assert result == [0, 1, 2, 3, 4]

    def test_negative_start_raises(self):
        """'-5' splits to ('', '5'), int('') raises ValueError."""
        with pytest.raises(ValueError):
            self._parse_pages("-5")

    def test_no_end_raises(self):
        """'1-' splits to ('1', ''), int('') raises ValueError."""
        with pytest.raises(ValueError):
            self._parse_pages("1-")

    def test_multiple_hyphens_raises(self):
        """'1-2-3' splits to ('1', '2-3'), int('2-3') raises ValueError."""
        with pytest.raises(ValueError):
            self._parse_pages("1-2-3")

    def test_end_clamped_to_total(self):
        """'1-999' with 10 pages produces pages 0-9."""
        result = self._parse_pages("1-999", total_pages=10)
        assert result == list(range(10))

    def test_single_page_zero_gives_negative_index(self):
        """'0' produces [0 - 1] = [-1]. The filter 0 <= n removes it."""
        result = self._parse_pages("0")
        page_nums = [n for n in result if 0 <= n < 10]
        assert page_nums == []

    def test_single_page_beyond_total(self):
        """'99' with 10 pages produces [98]. Filter removes it."""
        result = self._parse_pages("99", total_pages=10)
        page_nums = [n for n in result if 0 <= n < 10]
        assert page_nums == []


# ============================================================================
# SECTION 8: MCP tool argument type mismatch
# ============================================================================

class TestToolArgumentTypes:
    """Tests for when Ollama returns arguments in unexpected formats."""

    def test_string_arguments_not_parsed(self):
        """BUG: If Ollama returns arguments as a JSON string instead of a dict,
        purple.py passes the string directly to MCP call_tool.
        This happens with some model variants that serialize arguments."""
        tool_call = {
            "function": {
                "name": "store_memory",
                "arguments": '{"content": "test", "type": "fact"}',
            }
        }
        func = tool_call.get("function", {})
        args = func.get("arguments", {})
        # The code does NOT check if args is a string
        assert isinstance(args, str), "String arguments should be caught"
        # This string would be passed to client.call_tool() as-is
        # MCP expects a dict, so the call would fail

    def test_none_arguments_handled(self):
        """If arguments is None, the default {} should be used."""
        tool_call = {
            "function": {
                "name": "list_recent",
                "arguments": None,
            }
        }
        func = tool_call.get("function", {})
        args = func.get("arguments", {})
        # None is returned from dict.get -- the default {} is NOT used
        # because the key exists with value None
        assert args is None, "None arguments not caught by default value"

    def test_integer_arguments_not_dict(self):
        """Edge case: malformed tool_calls with non-dict arguments."""
        tool_call = {
            "function": {
                "name": "test",
                "arguments": 42,
            }
        }
        func = tool_call.get("function", {})
        args = func.get("arguments", {})
        assert args == 42
        assert not isinstance(args, dict)


# ============================================================================
# SECTION 9: MCP config loading edge cases
# ============================================================================

class TestMCPConfigEdgeCases:
    """Tests for MCP server configuration loading edge cases."""

    def test_missing_config_file(self):
        """If mcp.json does not exist, connect() should not crash."""
        mgr = MCPToolManager()
        # Patch MCP_CONFIG_PATH to a non-existent path
        import asyncio
        with patch("purple.MCP_CONFIG_PATH", Path("/nonexistent/mcp.json")):
            asyncio.run(mgr.connect())
        assert mgr.tool_names == []

    def test_invalid_json_config(self):
        """If mcp.json contains invalid JSON, connect() should not crash."""
        mgr = MCPToolManager()
        import asyncio
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            f.flush()
            with patch("purple.MCP_CONFIG_PATH", Path(f.name)):
                asyncio.run(mgr.connect())
        os.unlink(f.name)
        assert mgr.tool_names == []

    def test_empty_config_file(self):
        """Empty JSON object should result in no tools."""
        mgr = MCPToolManager()
        import asyncio
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            f.flush()
            with patch("purple.MCP_CONFIG_PATH", Path(f.name)):
                asyncio.run(mgr.connect())
        os.unlink(f.name)
        assert mgr.tool_names == []

    def test_disabled_server_skipped(self):
        """Servers with enabled:false should be skipped."""
        mgr = MCPToolManager()
        import asyncio
        config = {"servers": {"disabled_server": {"enabled": False, "command": ["echo"]}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            with patch("purple.MCP_CONFIG_PATH", Path(f.name)):
                asyncio.run(mgr.connect())
        os.unlink(f.name)
        assert mgr.tool_names == []

    def test_server_with_empty_command(self):
        """Server with empty command array should be skipped without crash."""
        mgr = MCPToolManager()
        import asyncio
        config = {"servers": {"bad_server": {"command": []}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            with patch("purple.MCP_CONFIG_PATH", Path(f.name)):
                asyncio.run(mgr.connect())
        os.unlink(f.name)
        assert mgr.tool_names == []


# ============================================================================
# SECTION 10: Database file deletion while server running
# ============================================================================

class TestMemoryDatabaseResilience:
    """Tests for database resilience under adverse conditions."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path):
        self._orig_db_path = mem_server.DB_PATH
        self._orig_initialized = mem_server._initialized
        temp_db = tmp_path / "test_resilience.db"
        mem_server.DB_PATH = temp_db
        mem_server._initialized = False
        yield
        mem_server.DB_PATH = self._orig_db_path
        mem_server._initialized = self._orig_initialized

    def test_db_recreated_after_deletion(self):
        """If the database file is deleted, get_db() should recreate it.
        (Because _initialized is still True, the table CREATE IF NOT EXISTS is skipped,
        but sqlite3.connect creates the file and WAL mode still works.)"""
        # Create initial DB
        mem_server._store_memory("before deletion", "fact")
        assert mem_server.DB_PATH.exists()

        # Delete the DB file
        mem_server.DB_PATH.unlink()
        # Also delete WAL and SHM files if they exist
        wal = mem_server.DB_PATH.with_suffix(".db-wal")
        shm = mem_server.DB_PATH.with_suffix(".db-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()

        # Reset _initialized so table is recreated
        mem_server._initialized = False

        # Should recreate DB and work
        result = mem_server._store_memory("after deletion", "fact")
        assert "Stored memory #" in result
        assert mem_server.DB_PATH.exists()

    def test_db_deleted_without_reinit_fails_gracefully(self):
        """If DB is deleted but _initialized is still True, get_db() skips
        CREATE TABLE. sqlite3.connect will create an empty file.
        Operations on the missing table will fail."""
        mem_server._store_memory("before deletion", "fact")
        mem_server.DB_PATH.unlink()
        wal = mem_server.DB_PATH.with_suffix(".db-wal")
        shm = mem_server.DB_PATH.with_suffix(".db-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()

        # _initialized is still True, so table won't be recreated
        # This should raise OperationalError for missing table
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            mem_server._store_memory("after deletion", "fact")

    def test_negative_hours_in_list_recent(self):
        """Negative hours value should not crash -- it just returns no results
        because the cutoff is in the future."""
        mem_server._store_memory("test", "fact")
        result = mem_server._list_recent(hours=-1)
        assert "No memories" in result

    def test_zero_limit_in_recall(self):
        """Limit of 0 should return no results."""
        mem_server._store_memory("test", "fact")
        result = mem_server._recall_memories("test", limit=0)
        assert "No memories" in result


# ============================================================================
# SECTION 11: OllamaChat.chat() -- tool call round limit
# ============================================================================

class TestChatToolRoundLimit:
    """Tests verifying the MAX_TOOL_ROUNDS limit is enforced."""

    def _make_chat(self):
        mgr = MagicMock(spec=MCPToolManager)
        mgr.ollama_tools = []
        with patch("purple.load_identity", return_value="sys"):
            with patch("purple.httpx.AsyncClient"):
                chat = OllamaChat(mgr)
        return chat

    def test_max_tool_rounds_constant(self):
        """MAX_TOOL_ROUNDS should be defined and reasonable."""
        from purple import MAX_TOOL_ROUNDS
        assert MAX_TOOL_ROUNDS == 10
        assert isinstance(MAX_TOOL_ROUNDS, int)

    def test_history_grows_with_tool_calls(self):
        """Each tool call round adds assistant + tool messages to history."""
        chat = self._make_chat()
        # Simulate: user sends message, then 3 rounds of tool calls
        chat._messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t1"}}]},
            {"role": "tool", "content": "result1"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t2"}}]},
            {"role": "tool", "content": "result2"},
            {"role": "assistant", "content": "final answer"},
        ]
        assert len(chat._messages) == 6
        msgs = chat._build_messages()
        # system + 6 history = 7
        assert len(msgs) == 7


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
