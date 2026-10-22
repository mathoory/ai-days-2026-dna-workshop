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
    fig, ax = canvas(16, 6.6, ylim=(0, 66))
    steps = [("Install", "Claude Desktop, claude.ai/download"), ("Sign in", "with your lab seat"),
             ("Open a folder", "Code tab: an empty folder"), ("Prompt", "paste the task, approve each step")]
    r = 7
    for k, (h, d) in enumerate(steps):
        x, y = 2 + (k % 2) * 80, 50 - (k // 2) * 31
        ax.add_patch(plt.Circle((x + r, y), r, color=ACCENT))
        ax.text(x + r, y, str(k + 1), ha="center", va="center_baseline", fontsize=34, color="white", fontweight="bold")
        label(ax, x + 2 * r + 3, y + 3.5, h, fs=30, bold=True, ha="left")
        label(ax, x + 2 * r + 3, y - 4.5, d, fs=23, color=MUTED, ha="left")
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
        fig, ax = canvas(16, 7.6, ylim=(0, 76))
        label(ax, 2, 72, "Before each reply the model reads all of this", fs=23, bold=True, ha="left")
        x = 2
        for n, w, c in parts:
            ax.add_patch(Rectangle((x, 46), w * 1.56, 16, fc=c, ec="white", lw=2))
            label(ax, x + w * 0.78, 54, n, fs=18, color="white" if c in (MID, INK, ACCENT) else INK, bold=True)
            x += w * 1.56
        ax.add_patch(Rectangle((x, 46), 158 - x, 16, fc="white", ec=HAIR, lw=2))
        label(ax, (x + 158) / 2, 54, "free", fs=19, color=MUTED)
        label(ax, 2, 40, "it fills as the agent works", fs=20, color=MUTED, ha="left")
        if k == 1:
            label(ax, 2, 28, "Compaction", fs=27, color=HELD_C, bold=True, ha="left")
            label(ax, 32, 28, "when it is full, the history so far is replaced by a short summary", fs=21, color=INK,
                  ha="left")
            arrow(ax, 152, 45, 152, 20, color=HELD_C, lw=3, ms=24)
            ax.add_patch(Rectangle((2, 4), 30, 14, fc=HELD_C, ec="white", lw=2))
            label(ax, 17, 11, "summary", fs=20, color="white", bold=True)
            ax.add_patch(Rectangle((32, 4), 126, 14, fc="white", ec=HAIR, lw=2))
            label(ax, 95, 11, "free again; exact numbers and early files may be gone", fs=19, color=MUTED)
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
            label(ax, 88, 82, "Sessions: files and messages", fs=24, bold=True, ha="left")
            blk(ax, 88, 58, 30, 12, "Session 1", fc=ACCENT, fs=21)
            blk(ax, 126, 58, 30, 12, "Session 2", fc=ACCENT, fs=21)
            ax.annotate("", (125.5, 64), (118.5, 64), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2.5,
                                                                    mutation_scale=20))
            label(ax, 122, 74, "messages", fs=18, color=INK, bold=True)
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
        ax.add_patch(Rectangle((x, 26), 25, v * 3.2, fc=ACCENT if i == 1 else ACCENT_L, ec="none"))
        label(ax, x + 12.5, 20, n, fs=22, bold=True, color=ACCENT if i == 1 else INK)
        if i == 1:                                       # where to be most of the time
            ax.add_patch(Rectangle((x - 3, 1.5), 31, 48, fill=False, ec=BASE["U"], lw=3.5))
    label(ax, 2, 56, "thinking before each step: time and tokens", fs=21, color=MUTED, ha="left")
    ex = ["small, mechanical\nedits", "most lab work:\nscripts, plots, analyses", "a bug you\ncannot find", "a hard design\nchoice", "rarely needed"]
    for i, t in enumerate(ex):
        label(ax, 3 + i * 31.4 + 12.5, 7, t, fs=18, color=MUTED)
    save(fig, "ag_effort")


# ------------------------------------------------------------------ 6.1 nightly runs
def nightly():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    hours = list(range(18, 24)) + list(range(0, 9))
    x0, dx = 8, 10
    ax.plot([x0, x0 + dx * (len(hours) - 1)], [34, 34], color=INK, lw=2.5)
    for i, h in enumerate(hours):
        x = x0 + i * dx
        ax.plot([x, x], [32.5, 35.5], color=INK, lw=2.5)
        label(ax, x, 28, f"{h:02d}", fs=19, color=MUTED)
        if 0 < i < len(hours) - 1:
            ax.add_patch(plt.Circle((x, 41), 1.8, color=ACCENT))
    for j, (a, b) in enumerate([(0.3, 5.5), (1, 9.2), (5.7, 12.6)]):
        ax.add_patch(Rectangle((x0 + a * dx, 50 + j * 7), (b - a) * dx, 5, fc=HELD_C, ec="none"))
    label(ax, x0 + 13.4 * dx, 64, "jobs on\nthe GPU", fs=22, color=HELD_C, ha="left")
    label(ax, x0, 9, "you: plan with the\nagent, launch", fs=22, ha="left")
    label(ax, x0 + 6.6 * dx, 17, "job-heartbeat: a scheduled prompt wakes the agent;", fs=21, color=ACCENT)
    label(ax, x0 + 6.6 * dx, 11, "it reads the job's status, fixes or alerts.", fs=21, color=ACCENT)
    label(ax, x0 + 6.6 * dx, 5, "A watchdog alerts if no agent is awake.", fs=21, color=ACCENT)
    file_icon(ax, x0 + 12.2 * dx, 14, "report.md", fs=22)
    label(ax, x0 + 12.2 * dx, 6, "08:00 report", fs=21, color=GOOD_C, ha="left")
    save(fig, "ag_nightly")


# ------------------------------------------------------------------ 6.2 experiment tracking
def clearml():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    for i in range(4):
        blk(ax, 2, 62 - i * 16, 34, 12, f"run {i + 1}: lr {['1e-3', '3e-4', '1e-4', '3e-5'][i]}", fc=HELD_C, fs=21)
        arrow(ax, 37, 68 - i * 16, 54, 38 + (1.5 - i) * 5, color=MUTED, lw=2, ms=18)
    blk(ax, 55, 24, 40, 28, "", fc=INK)
    label(ax, 75, 43, "ClearML server", fs=25, color="white", bold=True)
    label(ax, 75, 32, "metrics, settings,\nmodel files", fs=21, color="white")
    arrow(ax, 96, 44, 110, 58, color=MUTED, lw=2.5, ms=20)
    arrow(ax, 96, 32, 110, 18, color=MUTED, lw=2.5, ms=20)
    label(ax, 112, 63, "You: the web", fs=24, ha="left", bold=True)
    label(ax, 112, 56, "dashboard", fs=24, ha="left", bold=True)
    label(ax, 112, 49, "curves, compare runs", fs=21, color=MUTED, ha="left")
    label(ax, 112, 21, "Agent: clearml-", fs=24, ha="left", bold=True, color=ACCENT)
    label(ax, 112, 14, "scientific-publishing", fs=24, ha="left", bold=True, color=ACCENT)
    label(ax, 112, 7, "logs runs, publishes the report", fs=21, color=MUTED, ha="left")
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
            blk(ax, 7, 6.5, 44, 9, "Claude app or phone", fc=TOOL, tc=INK, fs=18, bold=False)
            region(ax, 100, 2, 58, 34, "GPU server", color=HELD_C)
            ax.add_patch(FancyBboxPatch((104, 5), 50, 22, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                        ec=INK, lw=2))
            label(ax, 106, 24, "tmux session", fs=17, color=MUTED, ha="left")
            blk(ax, 108, 5.5, 22, 11, "Claude", fc=ACCENT, fs=20)
            blk(ax, 132, 5.5, 20, 11, "GPU", fc=TOOL, tc=INK, fs=19)
            arrow(ax, 130.5, 11, 131.5, 11, color=MUTED, lw=2, ms=14)
            arrow(ax, 52, 22, 104, 22, color=INK, lw=2.5, ms=22)
            label(ax, 78, 25.5, "ssh, then tmux attach", fs=18, color=INK)
            arrow(ax, 51.5, 11, 107.5, 11, color=ACCENT, lw=2.5, ms=22, ls=(0, (4, 3)))
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
    """One real step of Task 1, as messages between the model (blue) and the harness (black)."""
    fig, ax = canvas(16, 8.8, ylim=(0, 88))
    xm, xh = 30, 130                                    # the two lanes
    blk(ax, xm - 24, 74, 48, 11, "Language model", fc=ACCENT, fs=23)
    blk(ax, xh - 26, 74, 52, 11, "Harness (Claude Code)", fc=INK, fs=23)
    for x in (xm, xh):
        ax.plot([x, x], [6, 73], color=HAIR, lw=2, zorder=0)
    msgs = [(xh, xm, "your prompt, CLAUDE.md, the tool list", INK),
            (xm, xh, "Bash(\"python gc_u.py\")", ACCENT),
            (None, xh, "asks you: allow?  then runs it", INK),
            (xh, xm, "FileNotFoundError: ...positives.fa", INK),
            (xm, xh, "Edit(gc_u.py, fix the path)", ACCENT)]
    for i, (a, b, t, c) in enumerate(msgs):
        y = 64 - i * 13
        if a is None:
            ax.add_patch(FancyBboxPatch((xh - 40, y - 4.5), 66, 9, boxstyle="round,pad=0,rounding_size=1.2", fc=FAINT,
                                        ec=INK, lw=1.5))
            label(ax, xh - 7, y, t, fs=19, color=INK)
            continue
        arrow(ax, a + (-2 if a > b else 2), y - 2.5, b + (2 if a > b else -2), y - 2.5, color=c, lw=2.5, ms=22)
        label(ax, (a + b) / 2, y + 1.5, t, fs=19, color=c, family=MONO if "(" in t or "Error" in t else None)
    label(ax, xm, 3, "chooses the next step", fs=19, color=ACCENT)
    label(ax, xh, 3, "runs tools, asks, keeps the context", fs=19, color=INK)
    save(fig, "ag_harness")


# ------------------------------------------------------------------ 1.5 git as the project's history
def git_history():
    fig, ax = canvas(16, 6.4, ylim=(0, 64))
    log = [("c4d81aa", "gc_u.py: GC and U fractions, scatter plot", "10:06"),
           ("7be19d0", "download the AGO2 training files", "10:04"),
           ("a1f3c2e", "set up .venv with numpy and matplotlib", "10:02")]
    label(ax, 2, 59, "git log", fs=21, color=MUTED, ha="left", family=MONO)
    for i, (h, m, tm) in enumerate(log):
        y = 49 - i * 9
        label(ax, 2, y, h, fs=20, color=HELD_C, ha="left", family=MONO)
        label(ax, 20, y, m, fs=21, color=INK, ha="left")
        label(ax, 92, y, tm, fs=19, color=MUTED, ha="left")
    label(ax, 2, 18, "git diff 7be19d0 c4d81aa", fs=21, color=MUTED, ha="left", family=MONO)
    label(ax, 2, 10, "+ gc = (s.count(\"G\") + s.count(\"C\")) / len(s)", fs=20, color=GOOD_C, ha="left", family=MONO)
    label(ax, 2, 3, "- gc = s.count(\"GC\") / len(s)", fs=20, color=HELD_C, ha="left", family=MONO)
    for i, (h, d) in enumerate([("what changed", "line by line"), ("when, and why", "time and message"),
                                ("any version back", "one command")]):
        y = 50 - i * 17
        label(ax, 112, y + 2.5, h, fs=22, color=INK, ha="left", bold=True)
        label(ax, 112, y - 3.5, d, fs=19, color=MUTED, ha="left")
    save(fig, "ag_githistory")


# ------------------------------------------------------------------ 1.6 virtual environments
def venv():
    fig, ax = canvas(16, 7.4, ylim=(0, 74))
    region(ax, 2, 4, 156, 66, "Your computer")
    blk(ax, 8, 8, 40, 12, "system Python", fc=TOOL, tc=MUTED, fs=20, bold=False)
    label(ax, 28, 25, "left untouched", fs=18, color=MUTED)
    projs = [("ago2-agents/", ".venv", "numpy 2.1, torch 2.6", 56), ("thesis-2024/", ".venv", "numpy 1.26, torch 2.1", 56)]
    for i, (p, e, pk, y) in enumerate(projs):
        x = 56 + i * 50
        file_icon(ax, x, 60, p, color=INK, fs=20)
        blk(ax, x, 30, 44, 22, "", fc=ACCENT_L)
        label(ax, x + 2, 46, e, fs=21, color=ACCENT, ha="left", family=MONO, bold=True)
        label(ax, x + 2, 39, "its own Python", fs=18, color=INK, ha="left")
        label(ax, x + 2, 33.5, pk, fs=18, color=INK, ha="left")
    label(ax, 105, 16, "each project its own versions; delete the folder and it is gone", fs=19, color=MUTED)
    label(ax, 105, 9, "conda: the same idea, also installs Python itself and non-Python tools", fs=19, color=GOOD_C)
    save(fig, "ag_venv")


# ------------------------------------------------------------------ 6.8 the models: score against price
def model_prices():
    """benchlm.ai overall score (8 weighted categories, verified 20.10.2026) against API output price."""
    ms = [("Fable 5.1", 50, 83.25, "A"), ("Opus 5.5", 20, 86.69, "A"), ("Sonnet 5.5", 10, 82.25, "A"),
          ("Haiku 5.5", 0.5, 67.23, "A"), ("GPT-6 Astra", 50, 84.03, "O"), ("GPT-6 Sol", 10, 78.58, "O"),
          ("GPT-6 Luna", 0.5, 66.55, "O")]
    fig = newfig(); fig.set_size_inches(16, 8.4)
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.84])
    for fam, c in (("A", ACCENT), ("O", MID)):
        pts = sorted([(p, s) for _, p, s, f in ms if f == fam])
        ax.plot([p for p, _ in pts], [s for _, s in pts], color=c, lw=2, alpha=0.5, zorder=1)
    off = {"Fable 5.1": (12, -26), "Opus 5.5": (12, 6), "Sonnet 5.5": (12, 8), "Haiku 5.5": (12, 6),
           "GPT-6 Astra": (12, 6), "GPT-6 Sol": (12, -24), "GPT-6 Luna": (12, -22)}
    for n, p, s, f in ms:
        c = ACCENT if f == "A" else MID
        ax.scatter(p, s, s=260, color=c, zorder=3)
        ax.annotate(f"{n}  {s:.1f}", (p, s), xytext=off[n], textcoords="offset points", fontsize=22, color=c,
                    fontweight="bold" if f == "A" else "normal")
    ax.set_xscale("log"); ax.set_xlim(0.3, 160)
    ax.set_xticks([0.5, 1, 10, 20, 50]); ax.set_xticklabels(["$0.50", "$1", "$10", "$20", "$50"], fontsize=20)
    ax.set_ylim(62, 90)
    ax.set_xlabel("Price per million output tokens (log scale)")
    ax.set_ylabel("benchlm.ai overall score")
    ygrid(ax)
    ax.text(0.33, 88.5, "Claude", fontsize=24, color=ACCENT, fontweight="bold", va="top")
    ax.text(0.33, 86.3, "GPT-6", fontsize=24, color=MID, fontweight="bold", va="top")
    save(fig, "ag_models")


# ------------------------------------------------------------------ 1.7 git, the undo button
def git_undo():
    fig, ax = canvas(16, 7.2, ylim=(0, 72))
    steps = [("set up .venv", ACCENT), ("download data", ACCENT), ("gc_u.png", ACCENT), ("train CNN", HELD_C)]
    xs = [12, 52, 92, 132]
    ax.plot([xs[0], xs[-1]], [44, 44], color=MID, lw=4)
    for (tx, c), x in zip(steps, xs):
        ax.add_patch(plt.Circle((x, 44), 5, color=c, zorder=3))
        label(ax, x, 58, tx, fs=27, color=INK if c == ACCENT else HELD_C, bold=True)
    for sgn in (1, -1):
        ax.plot([132 - 3.4, 132 + 3.4], [44 - 3.4 * sgn, 44 + 3.4 * sgn], color="white", lw=4, zorder=4)
    ax.annotate("", (94, 37), (130, 37), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3, mutation_scale=30,
                                                          connectionstyle="arc3,rad=-0.45"))
    label(ax, 112, 10, "the script broke: back to the last commit that worked", fs=24, color=INK)
    label(ax, 2, 26, "one commit after\nevery step that works", fs=23, color=MUTED, ha="left")
    save(fig, "ag_git")


# ------------------------------------------------------------------ 2.3 the anatomy of the first prompt
def prompt_anatomy():
    """The Task 1 prompt, each line coloured by the part it plays; the parts listed on the left."""
    C = {"How": MUTED, "Goal": ACCENT, "Inputs": BASE["U"], "Definitions": HELD_C, "Constraints": GOOD_C,
         "Done when": TUNED_PURPLE, "Report": INK}
    lines = [("How", "I am new to coding agents. Work step by step ..."),
             ("How", "Make this folder a git repository and commit after each step that works."),
             ("Goal", "Goal: one scatter plot of the AGO2 RNA dataset."),
             ("How", "1. Set up a Python environment with numpy and matplotlib in .venv ..."),
             ("Inputs", "2. Download the two training files (FASTA; lower case, some use T for U):"),
             ("Inputs", "   https://huggingface.co/.../CLIPSEQ_AGO2.train.positives.fa   (bound)"),
             ("Inputs", "   https://huggingface.co/.../CLIPSEQ_AGO2.train.negatives.fa   (unbound)"),
             ("Definitions", "3. GC fraction = (G + C) / length, U fraction = (U + T) / length."),
             ("Constraints", "4. GC (x) against U (y): 5,000 per group (seed 0), bound in blue,"),
             ("Constraints", "   unbound in grey, small transparent dots, random order. No windows."),
             ("Done when", "5. Save the plot as gc_u.png"),
             ("Report", "   and tell me in two sentences whether the groups differ.")]
    fig, ax = canvas(16, 6.8, ylim=(0, 68))
    for i, (h, c) in enumerate(C.items()):                 # the parts, on the left
        y = 62 - i * 9.4
        ax.add_patch(Rectangle((1, y - 3.2), 1.4, 6.4, fc=c, ec="none"))
        label(ax, 4.5, y, h, fs=22, color=c, bold=True, ha="left")
    ax.add_patch(Rectangle((28, 1), 132, 66, fc="#F8F9FA", ec="none"))
    for i, (h, s) in enumerate(lines):                     # the prompt itself, on the right
        y = 63.5 - i * 5.5
        ax.add_patch(Rectangle((28, y - 2.75), 1.1, 5.5, fc=C[h], ec="none"))
        label(ax, 31, y, s, fs=18, color=C[h], ha="left", family=MONO, bold=h in ("Goal", "Done when"))
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


# ------------------------------------------------------------------ 2.3 permissions and plan mode
def control():
    fig, ax = canvas(16, 8.2, ylim=(0, 82))
    for x0 in (2, 84):
        ax.add_patch(FancyBboxPatch((x0, 18), 74, 62, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=MID,
                                    lw=2))
    label(ax, 6, 72, "The agent asks", fs=26, color=MUTED, ha="left", bold=True)
    label(ax, 6, 58, "Run:  python gc_u.py", fs=28, color=INK, ha="left", family=MONO)
    for i, (tx, c, tc) in enumerate([("Allow once", ACCENT, "white"), ("Always", TOOL, INK), ("Deny", TOOL, INK)]):
        blk(ax, 6 + i * 23, 28, 21, 14, tx, fc=c, tc=tc, fs=23)
    label(ax, 39, 8, "a permission covers the command,\nnot what the script does", fs=23, color=MUTED)
    label(ax, 88, 72, "Plan mode", fs=26, color=MUTED, ha="left", bold=True)
    steps = [("reads", ACCENT), ("proposes a plan", ACCENT), ("you accept", INK), ("then edits", GOOD_C)]
    for i, (tx, c) in enumerate(steps):
        y = 60 - i * 11
        ax.add_patch(plt.Circle((92, y), 2, color=c))
        label(ax, 97, y, tx, fs=26, color=c, ha="left", bold=c == INK)
    label(ax, 121, 8, "for anything longer\nthan a few steps", fs=23, color=MUTED)
    save(fig, "ag_control")


# ------------------------------------------------------------------ 4.4 when it goes in circles
def circles():
    import numpy as np
    fig, ax = canvas(16, 7.4, ylim=(0, 74))
    cx, cy, r = 26, 37, 18
    th = np.linspace(0.25, 2 * np.pi - 0.25, 200)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=HELD_C, lw=4)
    arrow(ax, cx + r * np.cos(th[-3]), cy + r * np.sin(th[-3]), cx + r * np.cos(th[-1]), cy + r * np.sin(th[-1]),
          color=HELD_C, lw=4, ms=28)
    label(ax, cx, cy + 4, "fix", fs=24, color=HELD_C, bold=True)
    label(ax, cx, cy - 4, "same error", fs=22, color=HELD_C)
    outs = [("Esc", "stops it at once"), ("A new session", "with what was tried and what failed"),
            ("Causes before edits", "list them and how to test each"), ("Higher effort", "for this one problem"),
            ("You", "after a few failed fixes")]
    for i, (h, d) in enumerate(outs):
        y = 66 - i * 14
        arrow(ax, 47, cy + (y - cy) * 0.35, 62, y, color=MUTED, lw=2, ms=18)
        label(ax, 64, y + 2.6, h, fs=23, color=INK, ha="left", bold=True)
        label(ax, 64, y - 3.4, d, fs=19, color=MUTED, ha="left")
    save(fig, "ag_circles")


# ------------------------------------------------------------------ 4.5 when not to use an agent
def when_not():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    qs = [("Can you check the result?", "no", "do not use an agent"),
          ("Unpublished or sensitive data?", "yes", "decide first (first slide)"),
          ("A change you already know?", "yes", "type it yourself"),
          ("Do you want to learn the method?", "yes", "do it yourself")]
    for i, (q, a, out) in enumerate(qs):
        y = 68 - i * 14
        blk(ax, 2, y - 5, 62, 10, q, fc=TOOL, tc=INK, fs=20, bold=False)
        arrow(ax, 65, y, 82, y, color=HELD_C, lw=2.5, ms=20)
        label(ax, 73.5, y + 3, a, fs=18, color=HELD_C, bold=True)
        label(ax, 84, y, out, fs=21, color=HELD_C, ha="left")
        if i < 3:
            arrow(ax, 33, y - 5.5, 33, y - 8.5, color=MUTED, lw=2, ms=16)
    blk(ax, 2, 4, 62, 9, "Use the agent", fc=ACCENT, fs=22)
    arrow(ax, 33, 68 - 3 * 14 - 5.5, 33, 13.5, color=MUTED, lw=2, ms=16)
    label(ax, 84, 11, "each seat has a budget per few hours and per week,", fs=18, color=MUTED, ha="left")
    label(ax, 84, 6, "and top models and subagents use it faster", fs=18, color=MUTED, ha="left")
    save(fig, "ag_whennot")


# ------------------------------------------------------------------ 6.7 sharing in the lab
def sharing():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    region(ax, 52, 20, 56, 46, "Lab repository", color=INK)
    for i, t in enumerate([".claude/skills/", "research-skills/", "prompts.md", "hooks"]):
        file_icon(ax, 58, 52 - i * 8.5, t, color=ACCENT if i < 2 else GOOD_C, fs=20)
    for i, (x, y) in enumerate([(14, 58), (14, 28), (146, 43)]):
        ax.add_patch(plt.Circle((x, y), 5, color=INK))
        label(ax, x, y, "you" if i == 0 else "lab", fs=17, color="white", bold=True)
        arrow(ax, x + (6 if x < 80 else -6), y, 51 if x < 80 else 109, 43 + (y - 43) * 0.4, color=MUTED, lw=2, ms=18)
    ax.annotate("", (14, 34), (14, 52), arrowprops=dict(arrowstyle="<|-|>", color=GOOD_C, lw=2.5, mutation_scale=20))
    label(ax, 20, 43, "review each\nother's report.md", fs=18, color=GOOD_C, ha="left")
    label(ax, 80, 12, "Connectors (MCP): Box, Drive, ClearML. Third-party code with access to your data", fs=19,
          color=HELD_C)
    label(ax, 80, 5, "Hooks: rules the harness enforces, such as no rm in data/raw", fs=19, color=MUTED)
    save(fig, "ag_sharing")


# ------------------------------------------------------------------ 6.9 where the data goes
def data_flow():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    region(ax, 2, 6, 70, 64, "Your PC")
    for i, t in enumerate(["files it opens", "command output", "your prompts"]):
        blk(ax, 7, 50 - i * 10, 34, 8, t, fc=TOOL, tc=INK, fs=19, bold=False)
    arrow(ax, 42, 44, 98, 44, color=ACCENT, lw=3, ms=24)
    label(ax, 86, 48, "sent", fs=19, color=ACCENT, bold=True)
    blk(ax, 100, 34, 56, 20, "", fc=INK)
    label(ax, 128, 47.5, "Anthropic", fs=22, color="white", bold=True)
    label(ax, 128, 39.5, "kept 30 days (commercial terms)", fs=17, color="white")
    file_icon(ax, 7, 16, "~/.claude/projects: transcripts, plain text", color=MUTED, fs=17)
    label(ax, 100, 24, "data/raw read-only", fs=19, color=GOOD_C, ha="left", bold=True)
    label(ax, 100, 17, "secrets never in the repo or a prompt", fs=19, color=HELD_C, ha="left", bold=True)
    label(ax, 100, 10, "one login per person", fs=19, color=INK, ha="left", bold=True)
    save(fig, "ag_dataflow")


# ------------------------------------------------------------------ 2.5 speak the prompt
def dictation():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    for i, (n, v, c) in enumerate([("Typing", 40, MID), ("Speaking", 150, ACCENT)]):
        y = 64 - i * 18
        label(ax, 2, y, n, fs=30, color=c, ha="left", bold=True)
        ax.add_patch(Rectangle((32, y - 6), v * 0.55, 12, fc=c, ec="none"))
        label(ax, 32 + v * 0.55 + 2, y, f"about {v} words\nper minute", fs=24, color=c, ha="left")
    ax.plot([2, 158], [30, 30], color=HAIR, lw=1.2)
    for i, (os_, key) in enumerate([("Windows", "Win + H"), ("macOS", "press Fn twice")]):
        x = 2 + i * 80
        label(ax, x, 18, os_, fs=26, color=MUTED, ha="left")
        label(ax, x + 30, 18, key, fs=30, color=INK, ha="left", family=MONO, bold=True)
    label(ax, 2, 5, "then read the text before you send it", fs=24, color=HELD_C, ha="left")
    save(fig, "ag_dictation")


# ------------------------------------------------------------------ 3.6 the vague CNN prompt: what two agents chose (tests 22.10)
def vibe_choices():
    fig, ax = canvas(16, 8.6, ylim=(0, 86))
    cols = [(2, "Not in the prompt", MUTED), (48, "Agent A", ACCENT), (86, "Agent B", ACCENT), (124, "Our recipe", CNN_C)]
    for x, h, c in cols:
        label(ax, x, 81, h, fs=22, color=c, bold=True, ha="left")
    ax.plot([2, 158], [76.5, 76.5], color=HAIR, lw=1.2)
    rows = [("Sequence length", "340, crop and pad", "375, pad", "501, pad"),
            ("Network", "2 conv + 2 linear", "2 conv + 1 linear", "2 conv + 1 linear"),
            ("Trainable weights", "33,729", "36,033", "2,491"),
            ("Epochs", "8, kept epoch 6", "8", "10"),
            ("Validation split", "10%", "10%", "none"),
            ("Shuffled-label control", "none", "none", "asked for"),
            ("Held-out AUROC", "0.785", "0.798", "0.866"),
            ("Training time (CPU)", "5.5 min", "4.2 min", "1.3 min")]
    for i, r in enumerate(rows):
        y = 70 - i * 9
        for (x, _, c), v in zip(cols, r):
            bold = i == 6 and x > 2
            label(ax, x, y, v, fs=21, color=INK if x == 2 else c, ha="left", bold=bold)
    save(fig, "ag_vibe")


JOBS = {"vibe_choices": vibe_choices, "dictation": dictation, "git_history": git_history, "venv": venv, "control": control, "circles": circles, "when_not": when_not, "sharing": sharing, "data_flow": data_flow,
        "git_undo": git_undo, "prompt_anatomy": prompt_anatomy, "coinflip": coinflip, "lab_skills": lab_skills,
        "checks": checks, "harness": harness, "model_prices": model_prices, "cnn_curve": cnn_curve, "chat_agent": chat_agent, "agent_loop": agent_loop, "setup_steps": setup_steps, "skill_anatomy": skill_anatomy,
        "context_window": context_window, "subagents_sessions": subagents_sessions, "lit_review": lit_review,
        "effort": effort, "nightly": nightly, "clearml": clearml, "where_runs": where_runs, "phases": phases,
        "wiki": wiki}

if __name__ == "__main__":
    want = set(sys.argv[1:])
    for n, job in JOBS.items():
        if not want or n in want:
            job()
