"""Commit messages the agent wrote are checked before harness publishes them.

harness scrubs every commit message IT writes (``sanitize_publication``), but
the implement prompt has the agent commit its own work, and ``GitHubPR.open``
pushed those messages as written: a GitHub token shape, terminal escape and bell
bytes and the worktree path all reached the origin byte for byte.

Rewriting the agent's commits is ruled out — a plumbing rewrite skips the
repository's ``commit-msg`` hook and breaks the binding to the reviewed commit.
So the guard is refuse-only: before the push it reads the messages of every
commit the push would publish and refuses on terminal control characters or a
HIGH-CONFIDENCE credential shape only, naming the commit and what to fix;
nothing is rewritten and nothing is pushed. The full scrub's loose keyword
pattern is deliberately not used — it refused innocent messages like
``Read api_key: settings.API_KEY_FROM_ENVIRONMENT``. A home or worktree path in
a message is reported, not refused: prose names paths legitimately.

The module-scope imports are only names that existed before this fix, so the
file collects against the unfixed code and reports assertion failures there.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.pipeline import pr as prmod
from harness.pipeline.worktree import WorktreeManager

# Credential SHAPES, assembled so this file carries no literal a scanner flags.
_GH_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
_SK_KEY = "sk-" + "A1b2C3d4E5f6G7h8J9k0LmNo"

_INNOCENT = [
    "Read api_key: settings.API_KEY_FROM_ENVIRONMENT instead of a literal",
    "Rename secret_token=generate_secret_token() helper",
    "Fix password: parsing for passwords longer than twelve characters",
    "Document the Authorization: Bearer header format in README",
    "Bump sk-learn pin note",
    "Add task-queue-worker-pool-sizing to the scheduler config",
    "Fix the risk-scoring-for-multi-file-changes regression",
    "Clone over ssh://git@github.com/owner/repo.git in CI",
    "Use https://x-access-token:${GITHUB_TOKEN}@github.com/o/r.git for CI clones",
    "Parse -----BEGIN RSA PRIVATE KEY----- headers in the key loader",
    "Rename github_pat_scope_validation_helper",
    "Handle the ghs_ prefix in token detection",
    "Keep tab\tseparated columns and CRLF\r\nline endings as they are",
    "Résumé parsing — unicode prose stays",
]

_HAZARDOUS = {
    "github-token": f"Used token {_GH_TOKEN} from .env to test the API",
    "sk-key": f"Configure the client with {_SK_KEY} for now",
    "aws-key": "Set AKIA" + "ABCDEFGHIJKLMNOP as the access key",
    "jwt": "Session eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlX2hlcmU works",
    "url-password": "Push to https://bob:hunter2pass@git.example.invalid/r.git",
    "slack-token": "Bot xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    "npm-token": "Publish with npm_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "gitlab-token": "Mirror with glpat-" + "x1Y2z3A4b5C6d7E8f9G0",
    "google-key": "Maps key AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
    "private-key": ("Fixture:\n-----BEGIN OPENSSH PRIVATE KEY-----\n"
                    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtz\n"
                    "-----END OPENSSH PRIVATE KEY-----\n"),
    "fine-grained-pat": "Token github_pat_" + "11ABCDEFG0123456789abc_" + "x" * 50 + "Z9",
    "escape-sequence": "Clear the screen \x1b[2J done",
    "bell": "Ring \x07 the bell",
    "osc-hyperlink": "See \x1b]8;;https://example.invalid/x\x07the docs\x1b]8;;\x07",
    "c1-control": "A CSI \x9b in the message",
    # A C1 byte that is not part of any UTF-8 sequence, as text decoded with
    # surrogateescape carries it: git hands a hand-made commit's raw bytes over
    # unconverted, and git itself reads such a byte as Latin-1 (a C1 control).
    "raw-c1-byte": "A raw \udc9b2J byte",
}


@pytest.mark.parametrize("message", _INNOCENT)
def test_innocent_messages_raise_no_hazard(message):
    """The narrow set refuses none of these; the full scrub's loose keyword and
    ``sk-`` patterns would have refused several."""
    from harness.pipeline.sanitize import publication_hazards

    assert publication_hazards(message) == []


@pytest.mark.parametrize("message", list(_HAZARDOUS.values()), ids=list(_HAZARDOUS))
def test_control_bytes_and_high_confidence_credentials_are_hazards(message):
    from harness.pipeline.sanitize import publication_hazards

    found = publication_hazards(message)
    assert found, message
    # The finding names a CLASS, never the matched text.
    assert all(_GH_TOKEN not in f and _SK_KEY not in f for f in found)


# ── the guard in GitHubPR.open ────────────────────────────────────────────────

_FIXTURE_GIT_PINS = ("-c", "core.ignoreStat=false", "-c", "color.ui=never",
                     "-c", "commit.gpgsign=false",
                     "-c", "tag.gpgsign=false",
                     "-c", "protocol.file.allow=always",
                     "-c", "safe.bareRepository=all",
                     "-c", "branch.autoSetupMerge=true")

_GH = r'''#!{python}
import json, sys
a = sys.argv[1:]
with open({log!r}, "a") as fh:
    fh.write(" ".join(a[:2]) + "\n")
if a[:2] == ["auth", "status"]:
    print(json.dumps({{"hosts": {{"github.com": [{{"login": "p", "state": "success"}}]}}}}))
elif a[:2] == ["pr", "list"]:
    print("[]")
elif a[:2] == ["pr", "create"]:
    sys.stdin.read()
    print("https://github.com/o/r/pull/9")
else:
    sys.exit(1)
'''


def _git(args, cwd, **kw):
    return subprocess.run(["git", *_FIXTURE_GIT_PINS, *args], cwd=str(cwd),
                          capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                               "GIT_CONFIG_NOSYSTEM": "1"}, **kw)


@pytest.fixture
def agent_commit(tmp_path, monkeypatch):
    """``commit(message)`` makes the agent's own commit in a managed worktree and
    returns ``(open_pr, sha, origin, gh_calls, worktree)``."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    origin = tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        _git(args, repo)
    (repo / "app.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "origin", "main"], repo)
    wt = WorktreeManager(repo, tmp_path / "wt", base_branch="main").ensure("t1")
    log = tmp_path / "gh.log"
    ghdir = tmp_path / "ghbin"
    ghdir.mkdir()
    (ghdir / "gh").write_text(_GH.format(python=sys.executable, log=str(log)))
    (ghdir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{ghdir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("harness.config.warn_if_outdated", lambda name, cli="": None)

    def commit(message: str):
        (Path(wt.path) / "app.py").write_text("x = 2\n")
        _git(["add", "-A"], Path(wt.path))
        _git(["commit", "-q", "--cleanup=verbatim", "-F", "-"], Path(wt.path),
             input=message)
        sha = _git(["rev-parse", "HEAD"], Path(wt.path)).stdout.strip()
        body = "## Set x\n" + prmod.attestation_trailer("t1", "low", sha, "## Set x\n")

        def open_pr() -> prmod.PrResult:
            return prmod.GitHubPR().open(repo=repo, worktree=Path(wt.path),
                                         branch=wt.branch, base="main",
                                         title="Set x [t1]", body=body, reviewed_sha=sha)

        def gh_calls() -> list[str]:
            return log.read_text().splitlines() if log.exists() else []

        return open_pr, sha, origin, gh_calls, Path(wt.path)

    return commit


def _on_origin(origin: Path, branch: str = "agent/t1") -> bool:
    res = subprocess.run(["git", "rev-parse", "--verify", "-q", f"refs/heads/{branch}"],
                         cwd=str(origin), capture_output=True, text=True, check=False)
    return res.returncode == 0


@pytest.mark.parametrize("kind", ["github-token", "osc-hyperlink", "c1-control"])
def test_a_hazardous_agent_message_refuses_the_push_and_names_the_commit(agent_commit,
                                                                       kind):
    """Unfixed: pushed — the origin's message held the token / the escape bytes
    raw. Fixed: nothing pushed, no PR, and the refusal says which commit and
    what to fix — without repeating the credential."""
    open_pr, sha, origin, gh_calls, _wt = agent_commit(
        f"Set x to 2\n\n{_HAZARDOUS[kind]}\n")

    res = open_pr()

    assert not res.ok, res.detail
    assert sha[:12] in res.detail
    assert _GH_TOKEN not in res.detail and "\x1b" not in res.detail
    assert not _on_origin(origin), "the commit was published anyway"
    assert "pr create" not in gh_calls()


def test_an_innocent_keyword_message_still_publishes(agent_commit):
    """The loose keyword pattern would have refused this one."""
    open_pr, _sha, origin, gh_calls, _wt = agent_commit(
        "Read api_key: settings.API_KEY_FROM_ENVIRONMENT instead of a literal\n")

    res = open_pr()

    assert res.ok and res.url == "https://github.com/o/r/pull/9", res.detail
    assert _on_origin(origin)
    assert "pr create" in gh_calls()


def test_a_message_naming_the_worktree_path_publishes_with_a_warning(agent_commit,
                                                                    caplog):
    """Paths are reported, not refused: prose names them legitimately."""
    root = logging.getLogger("harness")
    prior = root.propagate
    root.propagate = True
    caplog.set_level(logging.WARNING, logger="harness")
    try:
        _open_pr, _sha, origin, _calls, wt = agent_commit(
            "Set x to 2\n\nTested in the worktree first.\n")
        # Reword the message to name the worktree: same tree, new commit.
        subprocess.run(["git", *_FIXTURE_GIT_PINS, "commit", "-q", "--amend",
                        "-m", f"Set x to 2\n\nWorktree: {wt}\n"], cwd=str(wt),
                       check=True, capture_output=True)
        amended = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt),
                                 capture_output=True, text=True, check=True).stdout.strip()
        body = "## Set x\n" + prmod.attestation_trailer("t1", "low", amended, "## Set x\n")
        res = prmod.GitHubPR().open(repo=wt.parent.parent / "repo", worktree=wt,
                                    branch="agent/t1", base="main", title="Set x [t1]",
                                    body=body, reviewed_sha=amended)
    finally:
        root.propagate = prior

    assert res.ok, res.detail
    assert _on_origin(origin)
    assert any(amended[:12] in r.getMessage() and "path" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.WARNING), \
        [r.getMessage() for r in caplog.records]


# ── what the guard reads: the commits a push sends, as UTF-8 ─────────────────
#
# The repository the guard reads is agent-writable: refs/replace/* and
# info/grafts sit in the SHARED common dir, and so does the config that picks
# git's log output encoding. A push sends the real objects, so the guard reads
# those (no replace refs, no grafts), and it reads every message as UTF-8,
# whatever the repository's config says.


def _main_repo(wt: Path) -> Path:
    return wt.parent.parent / "repo"


def _open_at_tip(wt: Path) -> prmod.PrResult:
    """``GitHubPR.open`` for the worktree's branch at its current tip, as the
    review hands it over."""
    tip = _git(["rev-parse", "HEAD"], wt).stdout.strip()
    body = "## Set x\n" + prmod.attestation_trailer("t1", "low", tip, "## Set x\n")
    return prmod.GitHubPR().open(repo=_main_repo(wt), worktree=wt, branch="agent/t1",
                                 base="main", title="Set x [t1]", body=body,
                                 reviewed_sha=tip)


def _refused_for(res: prmod.PrResult, sha: str) -> bool:
    """The guard's refusal of *sha* — not some other failure on the way."""
    return (not res.ok and sha[:12] in res.detail
            and "Nothing was pushed" in res.detail)


def test_a_replace_ref_does_not_blind_the_guard(agent_commit):
    """``git replace`` puts a clean stand-in in front of the hazardous commit for
    every reader that follows replace refs, while a push sends the real object.
    Unfixed: the guard read the stand-in, and the token reached the origin."""
    open_pr, sha, origin, gh_calls, wt = agent_commit(
        f"Set x to 2\n\n{_HAZARDOUS['github-token']}\n")
    tree = _git(["rev-parse", f"{sha}^{{tree}}"], wt).stdout.strip()
    parent = _git(["rev-parse", f"{sha}~1"], wt).stdout.strip()
    stand_in = _git(["commit-tree", tree, "-p", parent, "-m", "Set x to 2"],
                    wt).stdout.strip()
    _git(["replace", sha, stand_in], wt)

    res = open_pr()

    assert _refused_for(res, sha), res.detail
    assert _GH_TOKEN not in res.detail
    assert not _on_origin(origin), "the hazardous commit was published"
    assert "pr create" not in gh_calls()


def test_a_graft_does_not_hide_a_commit_from_the_guard(agent_commit):
    """A graft that makes the next commit parentless hides the hazardous one
    from the range walk. Unfixed: the guard walked the grafted history, saw only
    the clean commit, and let the push go."""
    _open_pr, hazardous, origin, gh_calls, wt = agent_commit(
        f"Set x to 2\n\n{_HAZARDOUS['osc-hyperlink']}\n")
    (wt / "more.py").write_text("y = 1\n")
    _git(["add", "-A"], wt)
    _git(["commit", "-q", "-m", "Add y"], wt)
    tip = _git(["rev-parse", "HEAD"], wt).stdout.strip()
    (_main_repo(wt) / ".git" / "info").mkdir(exist_ok=True)
    (_main_repo(wt) / ".git" / "info" / "grafts").write_text(f"{tip}\n")

    res = _open_at_tip(wt)

    assert _refused_for(res, hazardous), res.detail
    assert not _on_origin(origin), "the hazardous commit was published"
    assert "pr create" not in gh_calls()


def test_an_innocent_cjk_message_publishes_whatever_the_log_output_encoding(agent_commit):
    """``i18n.logOutputEncoding=ISO-2022-JP`` is a legitimate setting in a
    Japanese project, and it makes git write ESC charset shifts around every
    kanji run. Unfixed: an innocent message was refused as "terminal control
    characters"."""
    open_pr, _sha, origin, gh_calls, wt = agent_commit("日本語のコミットメッセージ\n")
    _git(["config", "i18n.logOutputEncoding", "ISO-2022-JP"], _main_repo(wt))

    res = open_pr()

    assert res.ok, res.detail
    assert _on_origin(origin)
    assert "pr create" in gh_calls()


def test_a_log_output_encoding_cannot_hide_a_control_character(agent_commit):
    """``i18n.logOutputEncoding=ISO-8859-1`` turns the UTF-8 C1 control U+009B
    into the single byte 0x9b on git's output, which no longer decoded as a
    control character. Unfixed: published."""
    open_pr, sha, origin, gh_calls, wt = agent_commit(
        f"Set x to 2\n\n{_HAZARDOUS['c1-control']}\n")
    _git(["config", "i18n.logOutputEncoding", "ISO-8859-1"], _main_repo(wt))

    res = open_pr()

    assert _refused_for(res, sha), res.detail
    assert not _on_origin(origin), "the hazardous commit was published"
    assert "pr create" not in gh_calls()


def test_a_raw_c1_byte_in_a_hand_made_commit_is_refused(agent_commit):
    """``git commit`` re-encodes a message that is not UTF-8, but a commit built
    with ``hash-object`` keeps its raw bytes, and git prints them unconverted.
    Unfixed: the raw 0x9b (a C1 CSI, as git itself reads a stray byte) reached
    the guard as the four characters ``\\x9b`` and was published."""
    _open_pr, sha, origin, gh_calls, wt = agent_commit("Set x to 2\n")
    tree = _git(["rev-parse", f"{sha}^{{tree}}"], wt).stdout.strip()
    parent = _git(["rev-parse", f"{sha}~1"], wt).stdout.strip()
    raw = (f"tree {tree}\nparent {parent}\n"
           f"author t <t@t> 1700000000 +0000\ncommitter t <t@t> 1700000000 +0000\n\n"
           ).encode() + b"Set x \x9b2J to 2\n"
    made = subprocess.run(["git", *_FIXTURE_GIT_PINS, "hash-object", "-t", "commit", "-w",
                           "--stdin"], cwd=str(wt), input=raw, capture_output=True,
                          check=True).stdout.decode().strip()
    _git(["update-ref", "refs/heads/agent/t1", made, sha], wt)

    res = _open_at_tip(wt)

    assert _refused_for(res, made), res.detail
    assert "terminal control characters" in res.detail
    assert not _on_origin(origin), "the hazardous commit was published"
    assert "pr create" not in gh_calls()


def test_log_show_signature_does_not_break_the_guard(agent_commit):
    """``log.showSignature=true`` would put signature verification output in
    front of every message; the guard asks git not to."""
    open_pr, _sha, origin, _calls, wt = agent_commit("Set x to 2\n")
    _git(["config", "log.showSignature", "true"], _main_repo(wt))

    res = open_pr()

    assert res.ok, res.detail
    assert _on_origin(origin)


@pytest.fixture
def gpg_home(tmp_path, monkeypatch):
    """A throwaway GnuPG home with one unprotected signing key; returns its id.

    Every gpg run here — the fixture's, and the one ``git log`` makes to verify
    a signature — reads THIS home, never the operator's, and its daemons are
    stopped afterwards and their socket directory (under ``/run/user/<uid>``)
    removed. Skipped where gpg is not installed or cannot make a key.
    """
    gpg = shutil.which("gpg")
    if gpg is None:
        pytest.skip("gpg is not on PATH")
    home = tmp_path / "gnupg"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("GNUPGHOME", str(home))
    made = subprocess.run([gpg, "--batch", "--pinentry-mode", "loopback", "--passphrase", "",
                           "--quick-generate-key", "t <t@t>", "ed25519", "sign", "never"],
                          capture_output=True, text=True, check=False, timeout=60)
    try:
        if made.returncode:
            pytest.skip(f"gpg could not make a signing key: {made.stderr.strip()[:200]}")
        listed = subprocess.run([gpg, "--batch", "--with-colons", "--list-secret-keys"],
                                capture_output=True, text=True, check=True, timeout=60)
        fingerprint = next(line.split(":")[9] for line in listed.stdout.splitlines()
                           if line.startswith("fpr:"))
        yield fingerprint
    finally:
        for step in (["--kill", "all"], ["--remove-socketdir"]):
            subprocess.run(["gpgconf", *step], capture_output=True, check=False, timeout=60)


def test_a_signed_commit_is_named_by_its_sha_under_log_show_signature(agent_commit,
                                                                      gpg_home):
    """With ``log.showSignature=true`` and a signed commit, git prints gpg's
    verification lines on standard output ahead of the record, so they land in
    the record's SHA field: without ``--no-show-signature`` the refusal named
    ``gpg: Signatu`` instead of the commit."""
    _open_pr, _sha, origin, gh_calls, wt = agent_commit(
        f"Set x to 2\n\n{_HAZARDOUS['github-token']}\n")
    _git(["-c", "commit.gpgsign=true", "-c", f"user.signingkey={gpg_home}",
          "commit", "-q", "--amend", "--no-edit", "--cleanup=verbatim"], wt)
    signed = _git(["rev-parse", "HEAD"], wt).stdout.strip()
    assert _git(["cat-file", "commit", signed], wt).stdout.count("gpgsig") == 1  # premise
    _git(["config", "log.showSignature", "true"], _main_repo(wt))

    res = _open_at_tip(wt)

    assert _refused_for(res, signed), res.detail
    assert "gpg:" not in res.detail and _GH_TOKEN not in res.detail
    assert not _on_origin(origin), "the hazardous commit was published"
    assert "pr create" not in gh_calls()


# ── which commits the guard reads: what origin lacks, not base..branch ───────
#
# A push of the agent branch publishes every ancestor origin lacks — not only
# the task's own commits relative to the LOCAL base: also the operator's
# unpushed commits on the base, agent commits merged into the base in local
# mode, and a base ref an agent moved (refs are shared). So the guard reads
# everything the branch reaches that no refs/remotes/origin/* ref reaches, and
# history origin already has is not read again.


def _base_commit(repo: Path, message: str, *, push: bool = False) -> str:
    """A commit on the main checkout's ``main``, the task's base — pushed to
    origin when *push*. Returns its sha."""
    with (repo / "notes.md").open("a", encoding="utf-8") as fh:
        fh.write("an operator's note\n")
    _git(["add", "notes.md"], repo)
    _git(["commit", "-q", "--cleanup=verbatim", "-F", "-"], repo, input=message)
    if push:
        _git(["push", "-q", "origin", "main"], repo)
    return _git(["rev-parse", "HEAD"], repo).stdout.strip()


def _merge_the_base_in(wt: Path) -> None:
    """The agent merges ``main`` into its branch: pushing the branch publishes it."""
    _git(["merge", "-q", "--no-edit", "refs/heads/main"], wt)


def test_a_hazardous_commit_on_the_base_that_origin_lacks_refuses_the_push(agent_commit):
    """The operator committed on ``main`` and never pushed it, and the agent
    merged ``main`` into its branch. Unfixed: the guard read only
    ``main..agent/t1``, so that commit was never read, and its token reached
    the origin with the branch. Fixed: refused, naming the commit."""
    _open_pr, _sha, origin, gh_calls, wt = agent_commit("Set x to 2\n")
    repo = _main_repo(wt)
    hazardous = _base_commit(repo, f"Operator note\n\n{_HAZARDOUS['github-token']}\n")
    _merge_the_base_in(wt)

    res = _open_at_tip(wt)

    assert _refused_for(res, hazardous), res.detail
    assert _GH_TOKEN not in res.detail
    assert not _on_origin(origin), "the hazardous base commit was published"
    assert "pr create" not in gh_calls()


def test_history_origin_already_has_is_not_read_again(agent_commit):
    """The same commit, but origin has it (its ``main`` was pushed, so
    refs/remotes/origin/main reaches it): the push publishes nothing new there,
    so it must not block the task — or every later push of every task, for
    good."""
    _open_pr, _sha, origin, gh_calls, wt = agent_commit("Set x to 2\n")
    repo = _main_repo(wt)
    published = _base_commit(repo, f"Operator note\n\n{_HAZARDOUS['github-token']}\n",
                             push=True)
    _merge_the_base_in(wt)
    assert _git(["rev-parse", "refs/remotes/origin/main"], repo).stdout.strip() == published

    res = _open_at_tip(wt)

    assert res.ok and res.url == "https://github.com/o/r/pull/9", res.detail
    assert _on_origin(origin)
    assert "pr create" in gh_calls()


def test_with_no_origin_tracking_ref_the_whole_history_is_read(agent_commit):
    """Nothing here records what origin has (never fetched, or a fetch refspec
    that maps origin's branches elsewhere): unknown is never a proof, so the
    guard reads the branch's whole history rather than trust the local base.
    A hazardous commit origin may well have refuses the push until a fetch
    records that it does."""
    _open_pr, _sha, origin, gh_calls, wt = agent_commit("Set x to 2\n")
    repo = _main_repo(wt)
    published = _base_commit(repo, f"Operator note\n\n{_HAZARDOUS['osc-hyperlink']}\n",
                             push=True)
    _merge_the_base_in(wt)
    tracking = _git(["for-each-ref", "--format=%(refname)", "refs/remotes/origin/"],
                    repo).stdout.split()
    assert tracking == ["refs/remotes/origin/main"]
    _git(["update-ref", "-d", "refs/remotes/origin/main"], repo)

    res = _open_at_tip(wt)

    assert _refused_for(res, published), res.detail
    assert not _on_origin(origin), "the branch was published unread"
    assert "pr create" not in gh_calls()

    _git(["fetch", "-q", "origin"], repo)              # origin's main is recorded again
    assert _open_at_tip(wt).ok


@pytest.mark.parametrize("refs", ["as-this-git-reads-them", "skipped-when-broken"])
@pytest.mark.parametrize("case", ["garbage-in-a-loose-ref", "a-ref-naming-a-missing-object",
                                  "control-dangling-origin-head"])
def test_an_unreadable_origin_tracking_ref_refuses_the_push_until_it_is_repaired(
        agent_commit, monkeypatch, case, refs):
    """Every refs/remotes/origin/* ref bounds the walk, so on current git one
    git cannot read fails it: the push is refused with git's own diagnosis.
    Fail closed — such a ref can only block publication, never widen it. A git
    that skips a broken ref instead (older gits did, with a warning;
    ``GIT_REF_PARANOIA=0`` asks any git to) walks a range that ref does not
    bound — wider, never narrower — so the hazardous commit origin lacks is
    still read, and still refuses the push. Either way nothing is published.
    With the ref removed the walk runs on every git and names that commit; once
    origin has it, the push goes out. The control: a dangling
    refs/remotes/origin/HEAD symref (common after a default-branch rename) is
    skipped, and blocks nothing. The premise is checked before the push: git
    itself reports the planted ref (a reftable ref store never reads a loose
    ref file, so the case is skipped there), and a git that does not honour
    ``GIT_REF_PARANOIA=0`` skips that case. After it, harness must have read
    the ref the way git's own walk does."""
    if refs == "skipped-when-broken":
        monkeypatch.setenv("GIT_REF_PARANOIA", "0")
    _open_pr, _sha, origin, gh_calls, wt = agent_commit("Set x to 2\n")
    repo = _main_repo(wt)
    if case == "control-dangling-origin-head":
        _git(["symbolic-ref", "refs/remotes/origin/HEAD",
              "refs/remotes/origin/old-default"], repo)
        res = _open_at_tip(wt)
        assert res.ok and res.url == "https://github.com/o/r/pull/9", res.detail
        assert _on_origin(origin)
        return
    hazardous = _base_commit(repo, f"Operator note\n\n{_HAZARDOUS['github-token']}\n")
    _merge_the_base_in(wt)
    common = Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"],
                       repo).stdout.rstrip("\n"))
    broken = common / "refs" / "remotes" / "origin" / "broken"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("garbage\n" if case == "garbage-in-a-loose-ref"
                      else "1234567890abcdef1234567890abcdef12345678\n")
    # The premise, before any push: the file is a ref git reads. Without it every
    # branch below passes on the ordinary walk and never meets a broken ref.
    if _git(["rev-parse", "--show-ref-format"], repo).stdout.strip() == "reftable":
        pytest.skip("a reftable ref store never reads a loose ref file, so no broken "
                    "tracking ref can be planted this way")

    def walk(**paranoia: str) -> subprocess.CompletedProcess:
        """The guard's walk (``--not --remotes=origin``), run by git alone."""
        return subprocess.run(["git", *_FIXTURE_GIT_PINS, "rev-list", "HEAD", "--not",
                               "--remotes=origin"], cwd=str(repo), capture_output=True,
                              text=True, check=False,
                              env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                                   "GIT_CONFIG_NOSYSTEM": "1", **paranoia})

    seen = walk(GIT_REF_PARANOIA="1")
    assert "origin/broken" in seen.stderr, seen.stderr       # git reports the planted ref
    unasked = walk()                          # this git's own walk, in harness's env
    if refs == "skipped-when-broken" and unasked.returncode:
        pytest.skip("this git does not honour GIT_REF_PARANOIA=0: its walk still dies "
                    f"on the broken ref ({unasked.stderr.strip()})")

    res = _open_at_tip(wt)

    assert ("could not be read" in res.detail) == bool(unasked.returncode), res.detail
    if "could not be read" in res.detail:     # current git: the walk dies on that ref
        assert refs == "as-this-git-reads-them", res.detail
        assert not res.ok and "origin/broken" in res.detail, res.detail  # git's own words
    else:                                     # a git that skips it: the wider walk
        assert _refused_for(res, hazardous), res.detail
    assert _GH_TOKEN not in res.detail
    assert not _on_origin(origin), "published past a tracking ref the walk could not read"
    assert "pr create" not in gh_calls()

    broken.unlink()                           # repaired: the walk runs on every git
    assert _refused_for(_open_at_tip(wt), hazardous)
    _git(["push", "-q", "origin", "main"], repo)      # origin has that history now
    assert _open_at_tip(wt).ok
