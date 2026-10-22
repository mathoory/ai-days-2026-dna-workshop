"""Figures of the lab tutorial deck "Agentic Accelerated Discovery" (slides/agents_lab.qmd).

    python figs_agents.py [name ...]     (any env with matplotlib; writes ../assets/custom/ag_*.png)

Colours: the agent ACCENT blue, its tools grey, you INK, files and reports GOOD_C green, the GPU server HELD_C orange.
"""
import sys

from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403

TOOL = "#EEF0F2"


def save_raw(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def blk(ax, x, y, w, h, text, fc=ACCENT, tc="white", fs=22, bold=True, ec="none", lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    if text:
        label(ax, x + w / 2, y + h / 2, text, fs=fs, color=tc, bold=bold)


def region(ax, x, y, w, h, name, color=MID):
    """An outlined area for a machine (laptop, server); its name on the top edge."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc="none", ec=color, lw=2,
                                ls=(0, (5, 3))))
    label(ax, x + 2, y + h - 3.2, name, fs=20, color=color, ha="left", bold=True)


def file_icon(ax, x, y, name, color=GOOD_C, fs=19):
    ax.add_patch(Rectangle((x, y - 2.6), 3.2, 5.2, fc="white", ec=color, lw=2))
    ax.plot([x + 0.6, x + 2.6], [y + 1.0, y + 1.0], color=color, lw=1.4)
    ax.plot([x + 0.6, x + 2.6], [y - 0.4, y - 0.4], color=color, lw=1.4)
    label(ax, x + 4.4, y, name, fs=fs, color=color, ha="left", family=MONO)


# ------------------------------------------------------------------ 1.2 a chatbot answers, an agent acts
def chat_agent():
    names = []
    for k in (0, 1):
        fig, ax = canvas(16, 8.6, ylim=(0, 86))
        label(ax, 2, 81, "Chatbot", fs=26, bold=True, ha="left")
        label(ax, 13, 66, "You", fs=26, bold=True)
        blk(ax, 62, 60, 34, 12, "Language model", fc=MID, fs=22)
        arrow(ax, 20, 69, 60, 69, color=INK, lw=2.5, ms=22)
        label(ax, 40, 72.5, "question", fs=19, color=MUTED)
        arrow(ax, 60, 63, 20, 63, color=INK, lw=2.5, ms=22)
        label(ax, 40, 59.5, "text", fs=19, color=MUTED)
        label(ax, 102, 66, "you copy, run, see the error,\npaste it back", fs=20, color=MUTED, ha="left")
        if k == 1:
            ax.plot([2, 158], [50, 50], color=HAIR, lw=1.2)
            label(ax, 2, 44, "Agent", fs=26, bold=True, ha="left")
            label(ax, 13, 22, "You", fs=26, bold=True)
            arrow(ax, 20, 22, 36, 22, color=INK, lw=2.5, ms=22)
            label(ax, 28, 26, "a goal", fs=19, color=MUTED)
            blk(ax, 38, 8, 40, 28, "", fc=ACCENT)
            label(ax, 58, 26, "Agent", fs=28, color="white", bold=True)
            label(ax, 58, 16, "language model\nthat can act", fs=19, color="white")
            tools = ["Files", "Terminal", "Python", "Web", "GPU server"]
            for i, t in enumerate(tools):
                yy = 38 - i * 8
                blk(ax, 118, yy - 3, 32, 6, t, fc=TOOL, tc=INK, fs=19, bold=False)
            arrow(ax, 80, 26, 114, 26, color=ACCENT, lw=2.5, ms=22)
            label(ax, 97, 29.5, "acts", fs=19, color=ACCENT, bold=True)
            arrow(ax, 114, 18, 80, 18, color=MUTED, lw=2.5, ms=22)
            label(ax, 97, 14.5, "sees the result", fs=19, color=MUTED)
            label(ax, 58, 3, "asks before actions you have not allowed", fs=19, color=INK)
        nm = f"ag_chat_agent_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 1.3 the agent loop, with what it looks like
def agent_loop():
    fig, ax = canvas(16, 8.4, ylim=(0, 84))
    nodes = {"Plan": (52, 70), "Act": (84, 42), "Observe": (52, 14)}
    for n, (x, y) in nodes.items():
        blk(ax, x - 13, y - 6, 26, 12, n, fc=ACCENT, fs=24)
    arrow(ax, 64, 64, 80, 49, color=ACCENT, lw=3, ms=24)
    arrow(ax, 80, 35, 64, 20, color=ACCENT, lw=3, ms=24)
    arrow(ax, 44, 20, 44, 64, color=ACCENT, lw=3, ms=24)
    label(ax, 41, 42, "repeat", fs=20, color=ACCENT, ha="right", bold=True)
    label(ax, 2, 81, "Goal", fs=22, bold=True, ha="left")
    label(ax, 2, 75, "plot GC against U", fs=20, color=MUTED, ha="left")
    arrow(ax, 33, 75, 38.5, 72, color=INK, lw=2.5, ms=20)
    ex = [(76, "read the two FASTA files, count letters", INK),
          (42, "$ python gc_u.py", MONO),
          (14, "FileNotFoundError: ...positives.fa", MONO)]
    for (y, t, f), x in zip(ex, (68, 100, 68)):
        label(ax, x, y, t, fs=20, color=MUTED, ha="left", family=MONO if f == MONO else None)
    label(ax, 68, 7, "next plan: fix the path, run again", fs=20, color=MUTED, ha="left")
    file_icon(ax, 118, 76, "gc_u.png", fs=20)
    label(ax, 118, 82.5, "Done", fs=22, bold=True, ha="left")
    save(fig, "ag_loop")


# ------------------------------------------------------------------ 2.1 setting up, four steps
def setup_steps():
    fig, ax = canvas(16, 5.2, ylim=(0, 52))
    steps = [("Install", "Claude Desktop\nclaude.ai/download"),
             ("Sign in", "a paid Claude plan\n(Pro or a lab seat)"),
             ("Open a folder", "Code tab:\nchoose an empty folder"),
             ("Prompt", "paste the task,\napprove each step")]
    r = 6
    for k, (h, d) in enumerate(steps):
        x = 2 + k * 40
        ax.add_patch(plt.Circle((x + r, 40), r, color=ACCENT))
        ax.text(x + r, 40, str(k + 1), ha="center", va="center_baseline", fontsize=30, color="white", fontweight="bold")
        label(ax, x + 2 * r + 2, 40, h, fs=26, bold=True, ha="left")
        label(ax, x, 22, d, fs=21, color=MUTED, ha="left", va="center")
    save(fig, "ag_setup")


# ------------------------------------------------------------------ 3.1 what a skill is
def skill_anatomy():
    fig, ax = canvas(16, 8.4, ylim=(0, 84))
    tree = [("academic-figure-skill/", INK, True), ("  SKILL.md", ACCENT, True), ("  references/", INK, False),
            ("    typography.md", MUTED, False), ("    color-palettes.md", MUTED, False), ("  scripts/", INK, False),
            ("  assets/", INK, False)]
    for i, (t, c, b) in enumerate(tree):
        label(ax, 2, 76 - i * 9, t, fs=23, color=c, ha="left", family=MONO, bold=b)
    label(ax, 2, 6, "~/.claude/skills/", fs=21, color=MUTED, ha="left", family=MONO)
    stages = [("Always in context", "name and one-line description:\nwhen to use this skill", 68),
              ("When a task matches", "the instructions in SKILL.md\nare loaded", 42),
              ("Only if needed", "references and scripts\nare read or run", 16)]
    for i, (h, d, y) in enumerate(stages):
        blk(ax, 70, y - 8, 86, 16, "", fc=ACCENT if i == 0 else (ACCENT_L if i == 1 else TOOL))
        label(ax, 74, y + 3, h, fs=23, color="white" if i == 0 else INK, bold=True, ha="left")
        label(ax, 74, y - 3.5, d.replace("\n", " "), fs=19, color="white" if i == 0 else MUTED, ha="left")
        if i < 2:
            arrow(ax, 113, y - 8.5, 113, y - 17.5, color=MUTED, lw=2.5, ms=20)
    arrow(ax, 50, 67, 68, 67, color=MUTED, lw=2.5, ms=20)
    save(fig, "ag_skill")


# ------------------------------------------------------------------ 4.1 the context window
def context_window():
    names = []
    parts = [("Instructions", 10, MID), ("Your messages", 12, INK), ("Files read", 26, ACCENT),
             ("Command output", 34, ACCENT_L)]
    for k in (0, 1):
        fig, ax = canvas(16, 6.2, ylim=(0, 62))
        label(ax, 2, 57, "Context window: everything the model sees right now", fs=23, bold=True, ha="left")
        x = 2
        for n, w, c in parts:
            ax.add_patch(Rectangle((x, 30), w * 1.56, 14, fc=c, ec="white", lw=2))
            label(ax, x + w * 0.78, 37, n, fs=17, color="white" if c in (MID, INK, ACCENT) else INK, bold=True)
            x += w * 1.56
        ax.add_patch(Rectangle((x, 30), 158 - x, 14, fc="white", ec=HAIR, lw=2))
        label(ax, (x + 158) / 2, 37, "free", fs=18, color=MUTED)
        label(ax, 2, 24, "fills as the agent works; every reply reads all of it", fs=20, color=MUTED, ha="left")
        if k == 1:
            ax.add_patch(Rectangle((2, 4), 24, 12, fc=MID, ec="white", lw=2))
            label(ax, 14, 10, "summary", fs=18, color="white", bold=True)
            ax.add_patch(Rectangle((26, 4), 132, 12, fc="white", ec=HAIR, lw=2))
            label(ax, 92, 10, "when full: compaction, the history becomes a summary and details are lost", fs=19,
                  color=INK)
        nm = f"ag_context_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 4.2 subagents against new sessions
def subagents_sessions():
    names = []
    for k in (0, 1):
        fig, ax = canvas(16, 8.6, ylim=(0, 86))
        label(ax, 2, 82, "Subagents: one session delegates", fs=24, bold=True, ha="left")
        blk(ax, 14, 58, 48, 12, "Main session", fc=ACCENT, fs=22)
        for i in range(3):
            x = 2 + i * 26
            blk(ax, x, 20, 22, 12, f"Subagent {i + 1}", fc=ACCENT_L, tc=INK, fs=19)
            arrow(ax, 30 + (i - 1) * 9, 57.5, x + 9, 32.5, color=ACCENT, lw=2, ms=18)
            arrow(ax, x + 15, 32.5, 34 + (i - 1) * 9, 57.5, color=MUTED, lw=2, ms=18)
        label(ax, 2, 46, "task", fs=18, color=ACCENT, ha="left", bold=True)
        label(ax, 76, 46, "short\nresult", fs=18, color=MUTED, ha="right")
        label(ax, 2, 9, "each reads on its own; only the result\nenters the main context", fs=19, color=MUTED, ha="left")
        if k == 1:
            ax.plot([84, 84], [4, 84], color=HAIR, lw=1.2)
            label(ax, 88, 82, "New sessions: you hand over", fs=24, bold=True, ha="left")
            blk(ax, 88, 58, 30, 12, "Session 1", fc=ACCENT, fs=21)
            blk(ax, 126, 58, 30, 12, "Session 2", fc=ACCENT, fs=21)
            file_icon(ax, 106, 36, "handoff.md", fs=19)
            arrow(ax, 103, 57.5, 108, 40, color=GOOD_C, lw=2, ms=18)
            arrow(ax, 124, 40, 141, 57.5, color=GOOD_C, lw=2, ms=18)
            label(ax, 98, 47, "writes", fs=17, color=GOOD_C, ha="right")
            label(ax, 136, 47, "reads", fs=17, color=GOOD_C, ha="left")
            file_icon(ax, 106, 22, "wiki/", fs=19)
            label(ax, 88, 9, "a new session starts empty and knows\nonly what files or you tell it", fs=19,
                  color=MUTED, ha="left")
        nm = f"ag_sessions_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 4.3 a literature review, fanned out
def lit_review():
    fig, ax = canvas(16, 8.4, ylim=(0, 84))
    blk(ax, 30, 68, 100, 13, "", fc=ACCENT)
    label(ax, 80, 77.5, "Main session", fs=22, color="white", bold=True)
    label(ax, 80, 71.5, "review deep models for RNA-binding prediction, 30 papers", fs=19, color="white")
    topics = ["CNNs", "Language\nmodels", "Benchmarks", "Datasets", "Interpretation"]
    for i, t in enumerate(topics):
        x = 2 + i * 31.6
        arrow(ax, 80 + (i - 2) * 12, 67.5, x + 13.5, 52.5, color=ACCENT, lw=2, ms=18)
        blk(ax, x, 40, 27, 12, t, fc=ACCENT_L, tc=INK, fs=18)
        label(ax, x + 13.5, 36, "6 papers", fs=17, color=MUTED)
        file_icon(ax, x + 3, 27, f"lit/{['cnn', 'lm', 'bench', 'data', 'interp'][i]}.md", fs=16)
    arrow(ax, 80, 21, 80, 14, color=GOOD_C, lw=2.5, ms=20)
    file_icon(ax, 66, 7, "review.md", fs=22)
    label(ax, 98, 7, "+ one comparison table", fs=20, color=GOOD_C, ha="left")
    save(fig, "ag_litreview")


# ------------------------------------------------------------------ 5.1 reasoning effort
def effort():
    fig, ax = canvas(16, 6.4, ylim=(0, 64))
    levels = [("Low", 1), ("Medium", 2), ("High", 3.5), ("Extra high", 5.5), ("Max", 8)]
    for i, (n, v) in enumerate(levels):
        x = 3 + i * 31.4
        ax.add_patch(Rectangle((x, 26), 25, v * 3.2, fc=ACCENT if i == 0 else ACCENT_L, ec="none"))
        label(ax, x + 12.5, 20, n, fs=22, bold=True, color=ACCENT if i == 0 else INK)
    label(ax, 2, 56, "thinking before each step: time and tokens", fs=21, color=MUTED, ha="left")
    ex = ["scripts, plots,\nfile edits", "a new analysis", "a bug you\ncannot find", "a hard design\nchoice", "rarely needed"]
    for i, t in enumerate(ex):
        label(ax, 3 + i * 31.4 + 12.5, 7, t, fs=18, color=MUTED)
    save(fig, "ag_effort")


# ------------------------------------------------------------------ 6.1 nightly runs
def nightly():
    fig, ax = canvas(16, 6.4, ylim=(0, 64))
    hours = list(range(18, 24)) + list(range(0, 9))
    x0, dx = 8, 10
    ax.plot([x0, x0 + dx * (len(hours) - 1)], [30, 30], color=INK, lw=2)
    for i, h in enumerate(hours):
        x = x0 + i * dx
        ax.plot([x, x], [29, 31], color=INK, lw=2)
        label(ax, x, 25, f"{h:02d}:00", fs=15, color=MUTED)
        if 0 < i < len(hours) - 1:
            ax.add_patch(plt.Circle((x, 36), 1.3, color=ACCENT))
    for j, (a, b) in enumerate([(0.3, 5.5), (1, 9.2), (5.7, 12.6)]):
        ax.add_patch(Rectangle((x0 + a * dx, 44 + j * 5), (b - a) * dx, 3.4, fc=HELD_C, ec="none"))
    label(ax, x0 + 13.4 * dx, 51, "jobs on the GPU", fs=18, color=HELD_C, ha="left")
    label(ax, x0, 6, "you: plan with the\nagent, launch", fs=19, ha="left", va="bottom")
    label(ax, x0 + 7 * dx, 15.5, "job-heartbeat: a scheduled prompt wakes the agent; it reads the job's status,", fs=18,
          color=ACCENT)
    label(ax, x0 + 7 * dx, 10.5, "fixes or alerts. A watchdog alerts if no agent is awake.", fs=18, color=ACCENT)
    file_icon(ax, x0 + 12.4 * dx, 10, "report.md", fs=19)
    label(ax, x0 + 12.4 * dx, 3, "08:00 morning report", fs=18, color=GOOD_C, ha="left")
    save(fig, "ag_nightly")


# ------------------------------------------------------------------ 6.2 experiment tracking
def clearml():
    fig, ax = canvas(16, 6.6, ylim=(0, 66))
    for i in range(4):
        blk(ax, 2, 52 - i * 13, 34, 9, f"run {i + 1}: lr {['1e-3', '3e-4', '1e-4', '3e-5'][i]}", fc=HELD_C, fs=18)
        arrow(ax, 37, 56.5 - i * 13, 54, 33 + (1.5 - i) * 4, color=MUTED, lw=2, ms=18)
    blk(ax, 55, 22, 40, 22, "", fc=INK)
    label(ax, 75, 37, "ClearML server", fs=22, color="white", bold=True)
    label(ax, 75, 28, "metrics, settings,\nmodel files", fs=18, color="white")
    arrow(ax, 96, 38, 112, 50, color=MUTED, lw=2.5, ms=20)
    arrow(ax, 96, 28, 112, 16, color=MUTED, lw=2.5, ms=20)
    label(ax, 114, 52, "You: the web dashboard", fs=21, ha="left", bold=True)
    label(ax, 114, 46, "curves, compare runs", fs=19, color=MUTED, ha="left")
    label(ax, 114, 18, "Agent: clearml-scientific-", fs=21, ha="left", bold=True, color=ACCENT)
    label(ax, 114, 12.5, "publishing skill", fs=21, ha="left", bold=True, color=ACCENT)
    label(ax, 114, 6.5, "logs runs, publishes the report", fs=19, color=MUTED, ha="left")
    save(fig, "ag_clearml")


# ------------------------------------------------------------------ 7.1 where Claude runs
def where_runs():
    names = []
    for k in (0, 1):
        fig, ax = canvas(16, 9, ylim=(0, 90))
        label(ax, 2, 86, "A. Claude on your PC, code runs on the server", fs=23, bold=True, ha="left")
        region(ax, 2, 50, 54, 30, "Your PC")
        blk(ax, 8, 54, 42, 16, "Claude", fc=ACCENT, fs=24)
        region(ax, 100, 50, 58, 30, "GPU server", color=HELD_C)
        blk(ax, 106, 54, 46, 16, "Python, data, GPU", fc=TOOL, tc=INK, fs=21, bold=False)
        arrow(ax, 51, 66, 105, 66, color=ACCENT, lw=2.5, ms=22)
        label(ax, 78, 69.5, "ssh: runs commands", fs=18, color=ACCENT)
        arrow(ax, 105, 58, 51, 58, color=MUTED, lw=2.5, ms=22)
        label(ax, 78, 54.5, "output, files", fs=18, color=MUTED)
        if k == 1:
            ax.plot([2, 158], [45, 45], color=HAIR, lw=1.2)
            label(ax, 2, 40, "B. Claude on the server, you connect to it", fs=23, bold=True, ha="left")
            region(ax, 2, 2, 54, 32, "Your PC")
            blk(ax, 7, 17, 44, 10, "WSL terminal", fc=INK, fs=20)
            blk(ax, 7, 5, 44, 9, "Claude app or phone", fc=TOOL, tc=INK, fs=18, bold=False)
            region(ax, 100, 2, 58, 34, "GPU server", color=HELD_C)
            ax.add_patch(FancyBboxPatch((104, 5), 50, 22, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                        ec=INK, lw=2))
            label(ax, 106, 24, "tmux session", fs=17, color=MUTED, ha="left")
            blk(ax, 108, 8, 22, 11, "Claude", fc=ACCENT, fs=20)
            blk(ax, 132, 8, 20, 11, "GPU", fc=TOOL, tc=INK, fs=19)
            arrow(ax, 130.5, 13.5, 131.5, 13.5, color=MUTED, lw=2, ms=14)
            arrow(ax, 52, 22, 104, 22, color=INK, lw=2.5, ms=22)
            label(ax, 78, 25.5, "ssh, then tmux attach", fs=18, color=INK)
            arrow(ax, 52, 9.5, 107, 11, color=ACCENT, lw=2.5, ms=22, ls=(0, (4, 3)))
            label(ax, 78, 6, "Remote Control", fs=18, color=ACCENT)
        nm = f"ag_where_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 8.1 working in phases
def phases():
    fig, ax = canvas(16, 7.2, ylim=(0, 72))
    ph = ["Data", "Baseline", "Model", "Analysis"]
    rows = [("plan", "you approve", INK), ("run", "the agent works", ACCENT), ("report.md", "", GOOD_C),
            ("review", "you read it", INK)]
    for i, p in enumerate(ph):
        x = 2 + i * 40
        blk(ax, x, 56, 32, 11, f"{i + 1}  {p}", fc=ACCENT, fs=23)
        for j, (a, b, c) in enumerate(rows):
            y = 47 - j * 8
            if a == "report.md":
                file_icon(ax, x + 1, y, a, fs=19)
            else:
                label(ax, x + 1, y, a, fs=20, color=c, ha="left", bold=True)
                label(ax, x + 14, y, b, fs=17, color=MUTED, ha="left")
        if i < 3:
            arrow(ax, x + 33, 61.5, x + 39, 61.5, color=MUTED, lw=2.5, ms=20)
    ax.add_patch(Rectangle((2, 4), 152, 9, fc=FAINT, ec="none"))
    label(ax, 78, 8.5, "after every phase: wiki/FINDINGS.md, DECISIONS.md, RISKS.md updated", fs=20, color=GOOD_C)
    save(fig, "ag_phases")


# ------------------------------------------------------------------ 8.2 the shared wiki
def wiki():
    fig, ax = canvas(16, 8.6, ylim=(0, 86))
    tree = [("project/", "", INK), ("  CLAUDE.md", "how we work here", ACCENT), ("  wiki/", "", INK),
            ("    FINDINGS.md", "what we learned", GOOD_C), ("    DECISIONS.md", "what we chose, and why", GOOD_C),
            ("    RISKS.md", "what could be wrong", GOOD_C), ("  phases/", "", INK),
            ("    01_data/report.md", "", MUTED), ("    02_baseline/report.md", "", MUTED)]
    for i, (t, d, c) in enumerate(tree):
        y = 80 - i * 8.6
        label(ax, 2, y, t, fs=20, color=c, ha="left", family=MONO, bold=c != MUTED)
        if d:
            label(ax, 44, y, d, fs=17, color=MUTED, ha="left")
    ex = [("FINDINGS", "2026-10-22  GC and U alone give AUROC 0.66\n(1,000 held-out).  phases/02_baseline/report.md"),
          ("DECISIONS", "2026-10-22  Train on composition-matched\nnegatives.  Why: the GC shortcut."),
          ("RISKS", "Negatives come from the same genes.\nCheck: matched control.  Status: open")]
    for i, (h, t) in enumerate(ex):
        y = 76 - i * 26
        label(ax, 86, y, h, fs=19, color=GOOD_C, ha="left", bold=True)
        label(ax, 86, y - 9, t, fs=16, color=INK, ha="left", family=MONO)
    save(fig, "ag_wiki")


# ------------------------------------------------------------------ 9.2 the CNN of task 2, trained on a CPU (measured 22.10)
def cnn_curve():
    auc = [0.792, 0.819, 0.842, 0.855, 0.860, 0.862, 0.863, 0.864, 0.863, 0.866]
    sec = [7.4, 7.8, 8.2, 7.5, 7.5, 7.8, 7.7, 7.4, 7.4, 7.7]
    fig = newfig(); fig.set_size_inches(16, 7.6)
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.8])
    ep = range(1, 11)
    ax.plot(ep, auc, color=CNN_C, lw=3.5, marker="o", ms=10)
    for e, a in ((1, auc[0]), (10, auc[-1])):
        ax.annotate(f"{a:.2f}", (e, a), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=22, color=CNN_C)
    ax.set_xticks(list(ep)); ax.set_xlabel("Epoch (one pass over 92,346 sequences)"); ax.set_ylabel("Held-out AUROC")
    ax.set_ylim(0.75, 0.9); ygrid(ax)
    ax.text(10, 0.765, f"{sum(sec) / len(sec):.0f} s per epoch on a laptop CPU, {sum(sec):.0f} s for all 10",
            ha="right", fontsize=22, color=MUTED)
    save(fig, "ag_cnn_curve")


# ------------------------------------------------------------------ 1.3 model and harness
def harness():
    fig, ax = canvas(16, 8.6, ylim=(0, 86))
    blk(ax, 2, 50, 156, 32, "", fc=FAINT)
    label(ax, 6, 76, "Agent", fs=26, bold=True, ha="left")
    blk(ax, 6, 54, 50, 17, "", fc=ACCENT)
    label(ax, 31, 66, "Language model", fs=24, color="white", bold=True)
    label(ax, 31, 58.5, "chooses the next step", fs=19, color="white")
    label(ax, 62, 62.5, "+", fs=40, color=INK, bold=True)
    blk(ax, 70, 54, 84, 17, "", fc=INK)
    label(ax, 112, 66, "Harness", fs=24, color="white", bold=True)
    label(ax, 112, 58.5, "tools, permissions, context, skills, memory files", fs=19, color="white")
    rows = [("Claude Code", "Fable, Opus, Sonnet, Haiku", "Anthropic"), ("Codex", "GPT-6 Astra, Sol, Luna", "OpenAI")]
    label(ax, 70, 42, "Harness", fs=21, color=MUTED, bold=True, ha="left")
    label(ax, 6, 42, "Models", fs=21, color=MUTED, bold=True, ha="left")
    for i, (h, m, co) in enumerate(rows):
        y = 31 - i * 15
        label(ax, 6, y, m, fs=22, color=ACCENT, ha="left", bold=True)
        label(ax, 6, y - 6, co, fs=18, color=MUTED, ha="left")
        arrow(ax, 56, y - 2, 68, y - 2, color=MUTED, lw=2, ms=18)
        label(ax, 70, y - 2, h, fs=24, color=INK, ha="left", bold=True)
    label(ax, 112, 23, "same harness, a different model:\nchoose it per task", fs=19, color=MUTED, ha="left")
    save(fig, "ag_harness")


# ------------------------------------------------------------------ 1.4 the models, by price
def model_prices():
    rows = [("Top", [("Claude Fable 5.1", 50, "Anthropic"), ("GPT-6 Astra", 50, "OpenAI")]),
            ("Large", [("Claude Opus 5.5", 20, "Anthropic")]),
            ("Mid", [("Claude Sonnet 5.5", 10, "Anthropic"), ("GPT-6 Sol", 10, "OpenAI")]),
            ("Small", [("Claude Haiku 5.5", 0.5, "Anthropic"), ("GPT-6 Luna", 0.5, "OpenAI")])]
    fig = newfig(); fig.set_size_inches(16, 8.2)
    ax = fig.add_axes([0.2, 0.13, 0.62, 0.8])
    y, ticks, labels, tiers = 0, [], [], []
    for tier, ms in rows:
        y0 = y
        for name, out, co in ms:
            c = ACCENT if co == "Anthropic" else MID
            ax.barh(y, out, color=c, height=0.7)
            ax.text(out * 1.12, y, f"${out:g}", va="center", fontsize=22, color=c)
            ticks.append(y); labels.append(name); y += 1
        tiers.append(((y0 + y - 1) / 2, tier)); y += 0.6
    ax.set_xscale("log"); ax.set_xlim(0.2, 200)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=22); ax.invert_yaxis()
    for lab, (n) in zip(ax.get_yticklabels(), labels):
        lab.set_color(ACCENT if n.startswith("Claude") else MUTED)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xticks([0.5, 1, 10, 50]); ax.set_xticklabels(["$0.5", "$1", "$10", "$50"], fontsize=20)
    ax.set_xlabel("Price per million output tokens (log scale)")
    for yy, tier in tiers:
        ax.text(260, yy, tier, va="center", fontsize=22, color=INK, fontweight="bold")
    xgrid(ax)
    save(fig, "ag_models")


# ------------------------------------------------------------------ 1.7 git, the undo button
def git_undo():
    fig, ax = canvas(16, 5.2, ylim=(0, 52))
    steps = [("set up .venv", ACCENT), ("download data", ACCENT), ("gc_u.png", ACCENT), ("train CNN", HELD_C)]
    xs = [12, 52, 92, 132]
    ax.plot([xs[0], xs[-1]], [34, 34], color=MID, lw=3)
    for (t, c), x in zip(steps, xs):
        ax.add_patch(plt.Circle((x, 34), 3.4, color=c, zorder=3))
        label(ax, x, 44, t, fs=22, color=INK if c == ACCENT else HELD_C, bold=True)
    for sgn in (1, -1):
        ax.plot([132 - 2.4, 132 + 2.4], [34 - 2.4 * sgn, 34 + 2.4 * sgn], color="white", lw=3, zorder=4)
    ax.annotate("", (94, 29), (130, 29), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, mutation_scale=24,
                                                          connectionstyle="arc3,rad=-0.45"))
    label(ax, 112, 6, "the script broke: back to the last commit that worked", fs=20, color=INK)
    label(ax, 2, 22, "one commit after\nevery step that works", fs=19, color=MUTED, ha="left")
    save(fig, "ag_git")


# ------------------------------------------------------------------ 2.3 the anatomy of the first prompt
def prompt_anatomy():
    fig, ax = canvas(16, 8.4, ylim=(0, 84))
    parts = [("Goal", "one scatter plot of the AGO2 RNA dataset", ACCENT),
             ("Inputs", "the two FASTA files, their links, lower case, T for U", INK),
             ("How", "a .venv with numpy and matplotlib; git, a commit per step", MUTED),
             ("Definitions", "GC fraction = (G + C) / length;  U fraction = (U + T) / length", HELD_C),
             ("Constraints", "5,000 per group, seed 0, blue and grey, random order, no windows", GOOD_C),
             ("Done when", "gc_u.png exists", TUNED_PURPLE),
             ("Report", "two sentences: do the groups differ?", INK)]
    for i, (h, t, c) in enumerate(parts):
        y = 78 - i * 11.6
        ax.add_patch(Rectangle((2, y - 4.2), 1.4, 8.4, fc=c, ec="none"))
        label(ax, 6, y, h, fs=23, color=c, bold=True, ha="left")
        label(ax, 42, y, t, fs=21, color=INK, ha="left")
    save(fig, "ag_anatomy")


# ------------------------------------------------------------------ 3.2 what the prompt left open (from the two test runs)
def coinflip():
    fig, ax = canvas(16, 8.4, ylim=(0, 84))
    rows = [("How many points", "5,000 per group", "all 92,346"),
            ("Colours", "matplotlib blue, grey", "#2166AC, #999999"),
            ("Font", "DejaVu Sans", "Arial, 7 to 8 pt"),
            ("Figure size", "default, 7 x 6 in", "89 mm, one column"),
            ("Legend", "inside the plot", "outside, with n"),
            ("Statistics", "none", "Mann-Whitney, AUC, report")]
    label(ax, 2, 80, "Not in the prompt", fs=22, color=MUTED, bold=True, ha="left")
    label(ax, 56, 80, "Agent, no skill", fs=22, color=MID, bold=True, ha="left")
    label(ax, 108, 80, "Agent with the figure skill", fs=22, color=ACCENT, bold=True, ha="left")
    ax.plot([2, 158], [75.5, 75.5], color=HAIR, lw=1.2)
    for i, (d, a, b) in enumerate(rows):
        y = 68 - i * 11.4
        label(ax, 2, y, d, fs=22, color=INK, ha="left")
        label(ax, 56, y, a, fs=21, color=MID, ha="left")
        label(ax, 108, y, b, fs=21, color=ACCENT, ha="left")
    save(fig, "ag_coinflip")


# ------------------------------------------------------------------ 3.3 the lab's skills (gold-srv2, research-skills/)
def lab_skills():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    sk = [("research-workflow", "phases, backlog, experiments, claim review, closing a phase"),
          ("research-vault", "the project's notes: links, templates, phase history"),
          ("scientific-reporting", "a report built from its evidence: figures, captions, checks"),
          ("clearml-scientific-publishing", "logs runs and publishes the report to ClearML"),
          ("job-heartbeat", "watches long jobs overnight, alerts, fills idle GPU time")]
    for i, (n, d) in enumerate(sk):
        y = 70 - i * 13.5
        label(ax, 2, y, n, fs=23, color=ACCENT, ha="left", family=MONO, bold=True)
        label(ax, 2, y - 6, d, fs=20, color=MUTED, ha="left")
    save(fig, "ag_labskills")


# ------------------------------------------------------------------ 3.8 check its work: the numbers to expect
def checks():
    rows = [("Shuffled labels (expected)", 0.50, MID), ("GC and U only", 0.66, MID), ("CNN, 10 epochs", 0.866, CNN_C)]
    fig = newfig(); fig.set_size_inches(16, 6)
    ax = fig.add_axes([0.33, 0.18, 0.6, 0.78])
    for i, (n, v, c) in enumerate(rows):
        ax.barh(i, v - 0.4, left=0.4, color=c, height=0.62, alpha=0.45 if "expected" in n else 1)
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=24, color=c)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=23); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.4, 0.95); ax.set_xlabel("Held-out AUROC (1,000 sequences)"); xgrid(ax)
    save(fig, "ag_checks")

JOBS = {"git_undo": git_undo, "prompt_anatomy": prompt_anatomy, "coinflip": coinflip, "lab_skills": lab_skills,
        "checks": checks, "harness": harness, "model_prices": model_prices, "cnn_curve": cnn_curve, "chat_agent": chat_agent, "agent_loop": agent_loop, "setup_steps": setup_steps, "skill_anatomy": skill_anatomy,
        "context_window": context_window, "subagents_sessions": subagents_sessions, "lit_review": lit_review,
        "effort": effort, "nightly": nightly, "clearml": clearml, "where_runs": where_runs, "phases": phases,
        "wiki": wiki}

if __name__ == "__main__":
    want = set(sys.argv[1:])
    for n, job in JOBS.items():
        if not want or n in want:
            job()
