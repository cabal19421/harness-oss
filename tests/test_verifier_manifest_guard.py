"""The verifier must not read "the diff rendered as nothing" as "nothing changed".

``verify_change`` short-circuited on ``if not diff.strip()`` with a ``skip``
(``ok=True``, ``blocking=False``) *while holding the file manifest* — gitutil's
complete ``<status> <path>`` listing of the same change. An empty diff beside a
NON-EMPTY manifest is not "nothing to verify": it is a change whose body could not
be rendered (the reproduced case: a file whose content is not UTF-8 degraded its
chunk to nothing). That shipped the change to PR/push with **zero** verifier models
asked and a DEBUG line as the only trace — the gate's own contract, "a failure to
run is not a pass", inverted.

The guard here is deliberately independent of any marker gitutil does or does not
stamp into the body: it compares the MANIFEST against the diff. A marker (today
``gitutil.DIFF_UNDECODABLE``, tomorrow whatever else lands) arms the same rule on
top of it, by convention rather than by name.
"""
from __future__ import annotations

import pytest

from harness.pipeline import gitutil
from harness.pipeline.verifier import _build_prompt, verify_change

# The two helpers this guard added are imported inside the tests that need them,
# as the sibling verifier tests do for `_build_prompt`: the module-scope import
# list stays the public entry point, so this file still COLLECTS against a
# verifier that has not been fixed yet and reports real assertion failures
# instead of one import error.

# A real unified-diff body: a judge can see app/core.py and its hunk.
_DIFF_ONE_FILE = """diff --git a/app/core.py b/app/core.py
index 1111111..2222222 100644
--- a/app/core.py
+++ b/app/core.py
@@ -1,3 +1,3 @@
 def f():
-    return 1
+    return 2
"""


def _counting_ask(answer: str = "REASONS: none\nVERDICT: PASS\nCONFIDENCE: 0.9"):
    """An ``ask`` that records every call, so "no model was asked" is checkable."""
    calls: dict = {"n": 0, "prompts": []}

    def ask(prompt: str) -> str:
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return answer

    return ask, calls


def _counting_structured(verdict: str = "PASS"):
    """The schema-validated path, also counted: neither path may run silently."""
    from harness.pipeline.backends.base import OneshotResult

    calls: dict = {"n": 0}

    def ask_structured(prompt: str, schema: dict):
        calls["n"] += 1
        return OneshotResult(ran=True, structured={"verdict": verdict},
                             schema_enforced=True, text="")

    return ask_structured, calls


# ── 1. the defect: an unrendered change must not pass as an empty one ───────────


def test_empty_diff_with_a_non_empty_manifest_abstains_instead_of_skipping():
    """The reported defect. Pre-fix this returned ``skip`` (ok, non-blocking)."""
    ask, calls = _counting_ask()
    structured, scalls = _counting_structured()
    r = verify_change(ask=ask, ask_structured=structured, task_title="t",
                      task_intent="i", diff="   \n  ",
                      file_manifest="A latin.txt\nM app/core.py")

    assert r.verdict == "abstain", r.reasons
    # Not the old answer, and not a clean gate either way.
    assert r.verdict not in ("skip", "pass")
    # `uncertain` is the flag ReviewGate routes on: it forces `high` risk, so the
    # change reaches a human instead of an auto-merge. (`ok` stays True for every
    # abstain by the documented contract — only an explicit `fail` blocks a PR —
    # which is why the escalation rides on this flag, not on `ok`.)
    assert r.uncertain is True and r.blocking is False
    # Nothing was judged, so nothing may be reported as judged.
    assert r.votes == [] and r.agreement is None and r.confidence is None
    assert calls["n"] == 0 and scalls["n"] == 0, "no verifier may be asked"

    reason = " ".join(r.reasons).lower()
    assert "latin.txt" in reason and "app/core.py" in reason   # names the entries
    assert "unshown is not absent" in reason
    assert "empty" in reason
    # The feedback a human reads says it was flagged, not that it was judged wrong.
    assert "NOT confident" in r.feedback() and "latin.txt" in r.feedback()


def test_the_abstain_also_fires_on_the_schema_validated_path_alone():
    structured, scalls = _counting_structured()
    r = verify_change(ask=None, ask_structured=structured, task_title="t",
                      task_intent="i", diff="", file_manifest="M only.py")
    assert r.verdict == "abstain" and scalls["n"] == 0


def test_a_clipped_manifest_beside_an_empty_diff_still_abstains():
    """The clip marker line is not a file — but it is still evidence of files."""
    ask, calls = _counting_ask()
    manifest = f"M a.py\n… and 37 more file(s) {gitutil.MANIFEST_CLIPPED}"
    r = verify_change(ask=ask, task_title="t", task_intent="i", diff="",
                      file_manifest=manifest)
    assert r.verdict == "abstain" and calls["n"] == 0
    joined = " ".join(r.reasons)
    assert "a.py" in joined and "clipped" in joined
    assert "1 changed file(s)" in joined, joined      # the marker is not counted


def test_an_unavailable_backend_still_fails_open_ahead_of_the_guard():
    """Precedence is unchanged: no ask callable at all is infrastructure, and a
    gate that cannot run must never wedge the pipeline (ReviewGate does not even
    call it in that state)."""
    r = verify_change(ask=None, ask_structured=None, task_title="t",
                      task_intent="i", diff="", file_manifest="M a.py")
    assert r.verdict == "skip"
    assert r.reasons == ["verifier unavailable for this backend"]


# ── 2. the honest empty case is untouched ───────────────────────────────────────


@pytest.mark.parametrize("manifest", ["", "   ", "\n\n"])
def test_empty_diff_and_empty_manifest_still_skips(manifest):
    ask, calls = _counting_ask()
    r = verify_change(ask=ask, task_title="t", task_intent="i", diff="   ",
                      file_manifest=manifest)
    assert r.verdict == "skip" and r.ok and not r.blocking
    assert r.reasons == ["empty diff — nothing to verify"]
    assert calls["n"] == 0


def test_empty_diff_with_the_manifest_argument_omitted_still_skips():
    """The default call (no manifest supplied at all) is byte-identical."""
    ask, calls = _counting_ask()
    r = verify_change(ask=ask, task_title="t", task_intent="i", diff="")
    assert r.verdict == "skip" and r.reasons == ["empty diff — nothing to verify"]
    assert calls["n"] == 0


# ── 3. a non-empty diff behaves exactly as before ───────────────────────────────


def test_non_empty_diff_is_judged_as_before():
    ask, calls = _counting_ask()
    r = verify_change(ask=ask, task_title="t", task_intent="i",
                      diff=_DIFF_ONE_FILE, file_manifest="M app/core.py")
    assert r.verdict == "pass" and r.votes == ["pass", "pass"]
    assert calls["n"] == 2                       # both adversarial verifiers ran
    p = calls["prompts"][0]
    # Every manifest entry has hunks, so the prompt keeps its teeth: no truncation
    # preamble, no mandatory-ABSTAIN escape, and the list is still authoritative.
    assert "COMPLETE and AUTHORITATIVE" in p
    assert "MUST be ABSTAIN" not in p and "TRUNCATED" not in p
    assert "unshown, NOT absent" not in p


def test_free_form_diff_text_with_a_manifest_keeps_the_old_prompt():
    """A body with no file headers is not a rendered unified diff (several
    callers and tests pass prose), so shownness is not guessed from it — the
    prompt stays exactly as it was rather than arming the rule on everything."""
    p = _build_prompt("t", "i", "a short, COMPLETE diff body", "g", True, "M a.py")
    assert "COMPLETE and AUTHORITATIVE" in p
    assert "MUST be ABSTAIN" not in p and "TRUNCATED" not in p


def test_verifier_prompt_without_a_manifest_is_unchanged():
    before = _build_prompt("t", "i", _DIFF_ONE_FILE, "g", True)
    assert _build_prompt("t", "i", _DIFF_ONE_FILE, "g", True, "") == before
    assert "MUST be ABSTAIN" not in before and "COMPLETE and AUTHORITATIVE" not in before


# ── 4. a manifest entry with no hunks arms the mandatory-ABSTAIN rule ───────────


def test_manifest_entry_missing_from_the_diff_arms_the_abstain_rule():
    """The prompt may never claim a complete diff over a file it does not show —
    with or without a truncation marker in the body."""
    manifest = "M app/core.py\nA tests/test_core.py"
    p = _build_prompt("t", "i", _DIFF_ONE_FILE, "g", True, manifest)

    assert "MUST be ABSTAIN" in p                      # absence questions abstain
    assert "TRUNCATED" in p
    # The judge is told WHICH file it cannot see, and that the LIST is still whole.
    assert "unshown, NOT absent: tests/test_core.py" in p
    assert "The file list above IS complete." in p
    assert "app/core.py" not in p.split("unshown, NOT absent:")[1].split("\n")[0]


def test_every_manifest_entry_shown_does_not_arm_the_rule():
    manifest = "M app/core.py"
    p = _build_prompt("t", "i", _DIFF_ONE_FILE, "g", True, manifest)
    assert "MUST be ABSTAIN" not in p and "unshown" not in p


def test_shownness_is_read_from_file_headers_only():
    from harness.pipeline.verifier import _paths_named_in_diff, _unshown_manifest_paths

    named = _paths_named_in_diff(_DIFF_ONE_FILE)
    assert "app/core.py" in named
    # A path that appears only in hunk CONTENT is not evidence it was rendered.
    body = _DIFF_ONE_FILE + "+    import tests.test_core  # tests/test_core.py\n"
    assert _unshown_manifest_paths("A tests/test_core.py", body) == [
        "tests/test_core.py"]


def test_hunk_content_cannot_forge_a_file_header():
    """Prefix matching alone is forgeable, and the forgery clears the gate.

    A REMOVED line whose text starts ``-- `` renders as ``--- …``; an ADDED line
    whose text starts ``++ `` renders as ``+++ …`` — ordinary content in SQL, Lua
    and Haskell, and routine in a fixture that embeds a diff. Reproduced: the
    body below never renders secret_added.py, yet a bare-prefix reader called it
    shown and left the ABSTAIN rule disarmed. Headers are therefore read
    structurally: inside a chunk header only, and only as a ``---``/``+++`` pair.
    """
    from harness.pipeline.verifier import _paths_named_in_diff, _unshown_manifest_paths

    forged = _DIFF_ONE_FILE + (
        "+-- a SQL comment that renders as --- b/secret_removed.py\n"
        "++ + lua-ish content\n"
        "+++ b/secret_added.py\n"
    )
    named = _paths_named_in_diff(forged)
    assert named == {"app/core.py"}, named
    manifest = "M app/core.py\nA secret_added.py"
    assert _unshown_manifest_paths(manifest, forged) == ["secret_added.py"]
    assert "MUST be ABSTAIN" in _build_prompt("t", "i", forged, "g", True, manifest)


def test_a_spaced_filename_does_not_mark_a_different_file_shown():
    """``diff --git a/my file.py b/my file.py`` split on whitespace into "my" and
    "file.py", so a change that really was missing file.py's chunk read as fully
    shown and the ABSTAIN rule stayed disarmed. A path may contain spaces, so the
    pair is reconstructed at its midpoint, never split on whitespace."""
    from harness.pipeline.verifier import _paths_named_in_diff, _unshown_manifest_paths

    # Verbatim git rendering (note the tab git appends after a spaced name).
    body = (
        "diff --git a/my file.py b/my file.py\n"
        "new file mode 100644\n"
        "index 0000000..a003ef7\n"
        "--- /dev/null\n"
        "+++ b/my file.py\t\n"
        "@@ -0,0 +1 @@\n"
        "+y = 1\n"
    )
    assert _paths_named_in_diff(body) == {"my file.py"}
    manifest = "A file.py\nA my file.py"
    assert _unshown_manifest_paths(manifest, body) == ["file.py"]
    assert "unshown, NOT absent: file.py" in _build_prompt(
        "t", "i", body, "g", True, manifest)


def test_a_filename_ending_in_a_space_reads_as_shown():
    """The other direction of the same parse: git delimits such a name with a TAB
    (``+++ b/trail2 \t``), so trimming whitespace renames it into a file that does
    not exist and the change reads as unshown. Over-arming is the safe direction,
    but it costs the gate its teeth on every change carrying such a file."""
    from harness.pipeline.verifier import _unshown_manifest_paths

    body = (
        "diff --git a/trail2  b/trail2 \n"
        "new file mode 100644\n"
        "index 0000000..5981f60\n"
        "--- /dev/null\n"
        "+++ b/trail2 \t\n"
        "@@ -0,0 +1 @@\n"
        "+t = 1\n"
    )
    assert _unshown_manifest_paths("A trail2 ", body) == []


def test_shownness_against_a_real_git_rendering(tmp_path):
    """The parse is pinned to what git actually emits, not to a hand-written
    sample: a rendering change upstream must fail here rather than silently clear
    files as shown."""
    import subprocess

    from harness.pipeline.verifier import _unshown_manifest_paths

    repo = tmp_path / "repo"
    repo.mkdir()

    def g(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, check=False)

    g("init", "-q", "-b", "main")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (repo / "seed.py").write_text("x = 1\n")
    g("add", "-A")
    g("commit", "-qm", "seed")
    g("checkout", "-qb", "feat")
    (repo / "my file.py").write_text("y = 1\n")
    (repo / "file.py").write_text("z = 1\n")
    g("add", "-A")
    g("commit", "-qm", "work")

    diff = gitutil.diff_text(repo, base="main")
    manifest = "\n".join(gitutil.diff_manifest(repo, base="main"))
    assert _unshown_manifest_paths(manifest, diff) == []      # both are rendered

    # Drop file.py's chunk, as a clip or an unrenderable chunk would: the file is
    # then genuinely unshown, and must be reported as such.
    chunks = diff.split("diff --git ")
    without = "diff --git ".join(c for c in chunks if not c.startswith("a/file.py "))
    assert "a/file.py" not in without
    assert _unshown_manifest_paths(manifest, without) == ["file.py"]


def test_binary_and_spaced_and_added_files_count_as_shown():
    """Names, not hunks: a binary header tells the judge the file is in the change,
    and a path containing spaces must not read as missing."""
    from harness.pipeline.verifier import _unshown_manifest_paths

    body = (
        "diff --git a/logo.png b/logo.png\n"
        "Binary files /dev/null and b/logo.png differ\n"
        "diff --git a/my file.py b/my file.py\n"
        "--- /dev/null\n"
        "+++ b/my file.py\t\n"
        "@@ -0,0 +1 @@\n"
        "+y = 1\n"
    )
    manifest = "A logo.png\nA my file.py"
    assert _unshown_manifest_paths(manifest, body) == []
    assert "MUST be ABSTAIN" not in _build_prompt("t", "i", body, "g", True, manifest)


# ── 5. gitutil's dropped-chunk marker: arms the rule, is never depended on ──────


def test_a_new_gitutil_diff_marker_arms_the_rule_by_convention(monkeypatch):
    """A ``DIFF_*`` marker added to gitutil later arms the mandatory-ABSTAIN rule
    the day it lands — no import of a symbol this module may not have."""
    monkeypatch.setattr(gitutil, "DIFF_SOMETHING_NEW",
                        "…(a chunk of this change went missing)…", raising=False)
    p = _build_prompt("t", "i", "body …(a chunk of this change went missing)…",
                      "g", True)
    assert "MUST be ABSTAIN" in p and "TRUNCATED" in p


def test_gitutil_diff_markers_stay_distinctive_enough_to_scan_for():
    """The ``DIFF_*`` convention scan is only safe while the markers are unusual
    strings. A short or ordinary future constant (``DIFF_SEP = " "``) would match
    every diff and arm the mandatory-ABSTAIN rule on all of them, quietly
    retiring the gate — so pin distinctiveness here rather than in a review."""
    markers = {name: value for name, value in vars(gitutil).items()
               if name.startswith("DIFF_") and isinstance(value, str)
               and value.strip()}
    assert "DIFF_TRUNCATED" in markers, markers
    ordinary = _DIFF_ONE_FILE + (
        "diff --git a/logo.png b/logo.png\n"
        "Binary files /dev/null and b/logo.png differ\n"
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    for name, value in markers.items():
        assert len(value.strip()) >= 12, (name, value)
        assert value.strip() not in ordinary, (name, value)


def test_the_undecodable_chunk_marker_is_judged_with_the_abstain_rule():
    """gitutil now substitutes a marker chunk for a file it cannot render, so the
    body is no longer empty — the guard above and this one are complementary, not
    alternatives, and neither depends on the other existing."""
    marker = getattr(gitutil, "DIFF_UNDECODABLE", "")
    if not marker:
        pytest.skip("gitutil has no undecodable-chunk marker — the guard above "
                    "covers the empty-body case without one")
    ask, calls = _counting_ask()
    body = f"[harness] latin.txt {marker}"
    r = verify_change(ask=ask, task_title="t", task_intent="i", diff=body,
                      file_manifest="A latin.txt")
    # A non-empty body IS judged (unchanged behaviour) …
    assert calls["n"] == 2 and r.verdict == "pass"
    # … but never as complete evidence.
    assert "MUST be ABSTAIN" in calls["prompts"][0]
