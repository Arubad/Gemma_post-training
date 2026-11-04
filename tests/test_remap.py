import json

import pytest

from gemma_swe.remap_openhands import Drop, RemapConfig, remap
from gemma_swe.traj_filters import passes


def tc(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def openhands_record(extra_steps=()):
    root = "/workspace/acme__widgets__1.0"
    cat = "Here's the result of running `cat -n` on {}:\n".format(root + "/acme/http.py") + "".join(
        f"{i:6d}\tline {i}\n" for i in range(1, 321))
    steps = [
        ("I'll explore the repo first. This is long prose that should be cut.", tc(0, "str_replace_editor", command="view", path=root),
         f"Here's the files and directories up to 2 levels deep in {root}, excluding hidden items:\n{root}/acme\n{root}/acme/http.py"),
        ("", tc(1, "think", thought="The bug is probably in http.py. Let me look."), "Your thought has been logged."),
        ("", tc(2, "str_replace_editor", command="view", path=root + "/acme/http.py"), cat),
        ("Reproduce.", tc(3, "str_replace_editor", command="create", path=root + "/reproduce_issue.py", file_text="import acme\nprint(1)\n"),
         "File created successfully"),
        ("Run it.", tc(4, "execute_bash", command=f"cd {root} && python reproduce_issue.py"),
         f"Traceback...\nValueError\n[The command completed with exit code 1.]\n[Current working directory: {root}]"),
        ("", tc(5, "execute_bash", command="pip install -e ."), "ok"),
        ("Fix.", tc(6, "str_replace_editor", command="str_replace", path=root + "/acme/http.py", old_str="line 7", new_str="line seven"),
         "No replacement was performed, old_str `line 7` did not appear verbatim"),
        ("Fix again.", tc(7, "str_replace_editor", command="str_replace", path=root + "/acme/http.py", old_str="line 7\n", new_str="line seven\n"),
         "The file has been edited. Here's the result..."),
        *extra_steps,
        ("Done.", tc(9, "finish", message="Fixed the header parsing. All good."), ""),
    ]
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": f"<uploaded_files>\n{root}\n</uploaded_files>\n<issue_description>\nHeader parsing crashes\n</issue_description>"}]
    for content, call, out in steps:
        msgs.append({"role": "assistant", "content": content, "tool_calls": [call]})
        msgs.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": out})
    return {"instance_id": "acme__widgets-1", "trajectory": msgs, "resolved": True, "model_patch": "--- a/acme/http.py\n+++ b/acme/http.py\n"}


def test_remap_openhands():
    r = remap(openhands_record())
    assert r["repo"] == "acme/widgets" and r["problem_statement"] == "Header parsing crashes"
    calls = [(t["tool_calls"][0]["name"], t["tool_calls"][0]["arguments"], t["train"]) for t in r["turns"] if t["role"] == "assistant" and t.get("tool_calls")]
    names = [c[0] for c in calls]
    assert names[0] == "run_command" and calls[0][1]["command"].startswith("find /workspace ")
    # 320-line view -> 3 read_file chunks within the 150-line cap
    reads = [c for c in calls if c[0] == "read_file"]
    assert [(c[1]["start_line"], c[1]["end_line"]) for c in reads] == [(1, 150), (151, 300), (301, 320)]
    # think folded into the next rationale
    first_read = next(t for t in r["turns"] if t["role"] == "assistant" and t.get("tool_calls") and t["tool_calls"][0]["name"] == "read_file")
    assert first_read["content"].startswith("The bug is probably in http.py.")
    # scratch file goes to /tmp via heredoc; later command rewritten; pip removed
    heredoc = next(c for c in calls if c[0] == "run_command" and "<< 'PYEOF'" in c[1]["command"])
    assert heredoc[1]["command"].startswith("cat > /tmp/reproduce_issue.py")
    run = next(c for c in calls if c[0] == "run_command" and "python /tmp/reproduce_issue.py" in c[1]["command"])
    assert run[1]["command"] == "cd /workspace && python /tmp/reproduce_issue.py"
    assert not any("pip install" in c[1].get("command", "") for c in calls)
    # failed edit kept but masked; successful edit trained
    edits = [c for c in calls if c[0] == "edit_file"]
    assert [e[2] for e in edits] == [False, True]
    assert names[-1] == "submit_patch"
    assert r["turns"][-1] == {"role": "assistant", "content": "Fixed the header parsing. All good.", "train": True}
    # exit code parsed into an error envelope, cwd lines removed
    obs = [t for t in r["turns"] if t["role"] == "tool" and t["name"] == "run_command"]
    err = next(o for o in obs if o["result"]["status"] == "error")
    assert err["result"]["details"]["exit_code"] == 1 and "Current working directory" not in err["result"]["details"]["stdout"]
    assert passes(r)[0]


def test_remap_drops():
    rec = openhands_record()
    rec["resolved"] = False
    with pytest.raises(Drop):
        remap(rec)
    rec = openhands_record(extra_steps=[("", tc(8, "str_replace_editor", command="undo_edit", path="/workspace/acme__widgets__1.0/acme/http.py"), "")])
    with pytest.raises(Drop):
        remap(rec)
    with pytest.raises(Drop):
        remap(openhands_record(), RemapConfig(max_tool_calls=3))
