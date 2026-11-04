import json

import pytest
from conftest import FakeGemmaTokenizer

from gemma_swe.calibration import Calibration, calibration_from_requests, detect_tool_content_style, fill_user_template
from gemma_swe.chat_render import RenderMismatch, render_and_mask
from gemma_swe.harness_prompt import Budget
from gemma_swe.records import assemble


def neutral_record():
    return {
        "id": "t1", "source": "unit", "repo": "acme/widgets", "problem_statement": "Fix `parse_header` crash.",
        "layout": ["./acme/", "./acme/http.py"],
        "turns": [
            {"role": "assistant", "content": "Find it.", "tool_calls": [{"name": "run_command", "arguments": {"command": "grep -rn parse_header ."}}], "train": True},
            {"role": "tool", "name": "run_command", "result": {"status": "ok", "stdout": "acme/http.py:3", "stderr": "", "exit_code": 0}},
            {"role": "assistant", "content": "Bad edit.", "tool_calls": [{"name": "edit_file", "arguments": {"filepath": "acme/http.py", "old_string": "x", "new_string": "y"}}], "train": False},
            {"role": "tool", "name": "edit_file", "result": {"status": "error", "error_type": "FileEditError", "error_message": "nf"}},
            {"role": "assistant", "content": "Submit.", "tool_calls": [{"name": "submit_patch", "arguments": {}}], "train": True},
            {"role": "tool", "name": "submit_patch", "result": {"status": "ok", "patch_size": 10, "files_changed": 1}},
            {"role": "assistant", "content": "Done.", "train": True},
        ],
    }


def _labelled(tok, r):
    return tok.decode([t for t, m in zip(r.input_ids, r.labels_mask) if m])


def _check_default_mode(tok):
    calib = Calibration(tool_content_style="raw")
    msgs, tools = assemble(neutral_record(), calib, "SYSTEM PROMPT", Budget(), "coder", "desc")
    assert msgs[0]["role"] == "system" and "SYSTEM PROMPT" in msgs[0]["content"] and 'internal name is "coder"' in msgs[0]["content"]
    assert "acme/widgets" in msgs[1]["content"] and "Fix `parse_header` crash." in msgs[1]["content"]
    assert "## Workspace Layout" in msgs[1]["content"]
    assert all(m["content"] == "" for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    r = render_and_mask(tok, msgs, tools)
    lab = _labelled(tok, r)
    assert "Find it." not in r.text  # rationale dropped in default mode
    assert "<|tool_call>call:run_command" in lab and "<|tool_call>call:submit_patch" in lab
    assert lab.count("<|tool_response>") == 2  # stop token after each trained call
    assert "Done.<turn|>" in lab and not lab.endswith("\n")
    assert "call:edit_file" not in lab  # masked turn
    assert "SYSTEM PROMPT" not in lab and "parse_header crash" not in lab
    assert "acme/http.py:3" not in lab and "response:" not in lab
    return r


def test_default_mode_bare_tool_calls(fake_tok):
    _check_default_mode(fake_tok)


def test_default_mode_real_template(real_tok):
    _check_default_mode(real_tok)


def test_content_mode_is_rejected_by_gemma_template(fake_tok):
    msgs, tools = assemble(neutral_record(), Calibration(tool_content_style="raw"), "S", Budget(), rationale_mode="content")
    with pytest.raises(RenderMismatch):
        render_and_mask(fake_tok, msgs, tools)


def test_thinking_mode(fake_tok):
    msgs, tools = assemble(neutral_record(), Calibration(tool_content_style="raw"), "S", Budget(), rationale_mode="thinking")
    r = render_and_mask(fake_tok, msgs, tools, enable_thinking=True)
    lab = _labelled(fake_tok, r)
    assert "<|channel>thought\nFind it.\n<channel|><|tool_call>call:run_command" in lab
    assert "Submit.\n<channel|><|tool_call>call:submit_patch" in lab and "Bad edit." not in lab


def test_31b_prefill_is_prompt_not_label():
    tok = FakeGemmaTokenizer(gen_prefill_empty_thought=True)
    msgs, tools = assemble(neutral_record(), Calibration(tool_content_style="raw"), "S", Budget())
    r = render_and_mask(tok, msgs, tools)
    lab = _labelled(tok, r)
    assert "<|channel>" not in lab and lab.startswith("<|tool_call>call:run_command")


def test_history_stripping_is_detected(fake_tok):
    rec = neutral_record()
    rec["turns"].append({"role": "user", "content": "Please continue"})
    rec["turns"].append({"role": "assistant", "content": "ok", "train": True})
    msgs, tools = assemble(rec, Calibration(tool_content_style="raw"), "S", Budget(), rationale_mode="thinking")
    with pytest.raises(RenderMismatch):  # thoughts before the last user message are stripped from history
        render_and_mask(fake_tok, msgs, tools, enable_thinking=True)


def test_tool_content_styles():
    env = {"status": "ok", "stdout": "a"}
    raw = Calibration(tool_content_style="raw").wrap_tool_content(env)
    wrapped = Calibration(tool_content_style="adk_result_wrap").wrap_tool_content(env)
    assert json.loads(raw) == env
    assert json.loads(json.loads(wrapped)["result"]) == env
    assert detect_tool_content_style(raw) == "raw" and detect_tool_content_style(wrapped) == "adk_result_wrap"


def test_exact_record_args_become_dicts(fake_tok):
    rec = {"format": "exact", "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "U"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "name": "get_status", "content": "{}"},
        {"role": "assistant", "content": "fin"}], "tools": [{"type": "function", "function": {"name": "get_status"}}]}
    msgs, tools = assemble(rec, Calibration(), "S", Budget())
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {} and msgs[2]["content"] == ""
    r = render_and_mask(fake_tok, msgs, tools)
    assert r.n_label_tokens > 0


def test_calibration_from_requests_and_templating():
    instr = "MY INSTRUCTION"
    sys_text = f"{instr}\n\nYou are an agent. Your internal name is \"coder\"."
    problem = "Problem X happens"
    user = f"You are evaluating a software engineering task for repository a/b.\n\nProblem Statement:\n{problem}\n\n## Task Budget\n- x\n\n## Workspace Layout\n./a/\n./b.py"
    reqs = [
        {"model": "m", "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user}], "tools": [{"type": "function", "function": {"name": "run_command"}}]},
        {"model": "m", "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user},
                                    {"role": "assistant", "content": "", "tool_calls": []},
                                    {"role": "tool", "tool_call_id": "1", "content": json.dumps({"result": json.dumps({"status": "ok"})})}]},
    ]
    c = calibration_from_requests(reqs, {"problem_statement": problem, "repo": "a/b"}, instr)
    assert c.is_captured and c.tool_content_style == "adk_result_wrap"
    assert c.get_system("NEW INSTR") == "NEW INSTR\n\nYou are an agent. Your internal name is \"coder\"."
    u = c.get_user("c/d", "Other problem", Budget(), ["./x.py"])
    assert "repository c/d" in u and "Other problem" in u and u.endswith("## Workspace Layout\n./x.py")
    assert fill_user_template(user, problem, "a/b", "c/d", "P2", None).endswith("- x")
