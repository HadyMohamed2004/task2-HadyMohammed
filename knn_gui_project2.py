# ============================================================
#  DecodeLabs — AI
#  Project 2: Data Classification Using AI  (GUI)
#  Run: python gui_p2.py
#  Requires: tkinter, matplotlib, scikit-learn, numpy
# ============================================================

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Import the classifier module ─────────────────────────────
from classifier import run_full_pipeline, predict_single

# ── Palette ───────────────────────────────────────────────────
BG       = "#0b0e14"
SURFACE  = "#13171f"
SURFACE2 = "#1a1f2c"
SURFACE3 = "#232840"
ACCENT   = "#4f9cf9"
ACCENT2  = "#38d9a9"
WARN     = "#f7c948"
TEXT     = "#dde3f0"
MUTED    = "#4a5270"
BORDER   = "#252a3a"

CLASS_COLORS = ["#4f9cf9", "#38d9a9", "#f7804a"]

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})

FONT_TITLE = ("Courier New", 14, "bold")
FONT_HEAD  = ("Courier New", 11, "bold")
FONT_BODY  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_MONO  = ("Courier New", 10)


class KNNApp:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.results = None
        self._setup_window()
        self._build_layout()
        self._start_pipeline()

    def _setup_window(self):
        self.root.title("DecodeLabs AI — Project 2: KNN Classifier")
        self.root.geometry("1180x780")
        self.root.minsize(900, 650)
        self.root.configure(bg=BG)

    def _build_layout(self):
        self._build_header()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",       background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",   background=SURFACE2, foreground=MUTED,
                        font=FONT_BODY, padding=(16, 6), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", SURFACE3)],
                  foreground=[("selected", ACCENT)])

        self.tab_overview  = tk.Frame(self.notebook, bg=BG)
        self.tab_k_search  = tk.Frame(self.notebook, bg=BG)
        self.tab_confusion = tk.Frame(self.notebook, bg=BG)
        self.tab_predict   = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_overview,  text="  📊  Overview  ")
        self.notebook.add(self.tab_k_search,  text="  🔍  K Search  ")
        self.notebook.add(self.tab_confusion, text="  🔲  Confusion Matrix  ")
        self.notebook.add(self.tab_predict,   text="  🌸  Predict  ")

        self._build_loading()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=SURFACE, pady=12,
                       highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="◈  DecodeLabs AI — Project 2",
                 font=FONT_TITLE, fg=ACCENT, bg=SURFACE).pack(side=tk.LEFT, padx=20)
        self.status_lbl = tk.Label(hdr, text="⏳ Running pipeline...",
                                   font=FONT_SMALL, fg=WARN, bg=SURFACE)
        self.status_lbl.pack(side=tk.RIGHT, padx=20)
        tk.Label(hdr, text="KNN Classifier  •  Iris Dataset  •  sklearn",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE).pack(side=tk.RIGHT, padx=10)

    def _build_loading(self):
        self.loading_frame = tk.Frame(self.tab_overview, bg=BG)
        self.loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Label(self.loading_frame, text="⚙  Running Pipeline...",
                 font=FONT_HEAD, fg=ACCENT, bg=BG).place(relx=0.5, rely=0.44, anchor="center")
        tk.Label(self.loading_frame,
                 text="Load → Scale → K-Search → Train → Evaluate",
                 font=FONT_SMALL, fg=MUTED, bg=BG).place(relx=0.5, rely=0.51, anchor="center")

    def _start_pipeline(self):
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        results = run_full_pipeline()
        self.root.after(0, self._on_pipeline_done, results)

    def _on_pipeline_done(self, results):
        self.results = results
        self.loading_frame.destroy()
        self.status_lbl.config(text="✅ Pipeline complete", fg=ACCENT2)
        self._build_overview()
        self._build_k_search()
        self._build_confusion()
        self._build_predict()

    # ── TAB 1: Overview ────────────────────────────────────────
    def _build_overview(self):
        r    = self.results
        root = self.tab_overview
        root.columnconfigure(0, weight=2)
        root.columnconfigure(1, weight=3)
        root.rowconfigure(0, weight=1)

        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        cards = [
            ("Accuracy",     f"{r['accuracy']*100:.2f}%", ACCENT2),
            ("F1 Score",     f"{r['f1']:.4f}",            ACCENT),
            ("Optimal K",    str(r["optimal_k"]),         WARN),
            ("Training Set", "120 samples",               TEXT),
            ("Test Set",     "30 samples",                TEXT),
            ("Features",     "4  (scaled)",               TEXT),
            ("Classes",      "3  (balanced)",             TEXT),
            ("Algorithm",    "K-Nearest Neighbors",       MUTED),
        ]
        for label, value, color in cards:
            card = tk.Frame(left, bg=SURFACE2,
                            highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill=tk.X, pady=4, padx=4)
            tk.Label(card, text=label, font=FONT_SMALL,
                     fg=MUTED, bg=SURFACE2, anchor="w").pack(fill=tk.X, padx=12, pady=(8, 0))
            tk.Label(card, text=value, font=FONT_HEAD,
                     fg=color, bg=SURFACE2, anchor="w").pack(fill=tk.X, padx=12, pady=(0, 8))

        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)

        report  = r["report"]
        classes = r["class_names"]
        prec    = [report[c]["precision"] for c in classes]
        rec     = [report[c]["recall"]    for c in classes]
        f1s     = [report[c]["f1-score"]  for c in classes]

        x  = np.arange(len(classes))
        w  = 0.25
        fig = Figure(figsize=(5, 4), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.bar(x - w, prec, w, label="Precision", color=ACCENT,  alpha=0.85)
        ax.bar(x,     rec,  w, label="Recall",    color=ACCENT2, alpha=0.85)
        ax.bar(x + w, f1s,  w, label="F1",        color=WARN,    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in classes], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title("Per-Class Metrics", color=TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT)
        ax.grid(axis="y")

        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── TAB 2: K Search ────────────────────────────────────────
    def _build_k_search(self):
        r      = self.results
        k_vals = [k for k, _ in r["k_results"]]
        f1s    = [f for _, f in r["k_results"]]
        best_k = r["optimal_k"]
        best_f = r["k_results"][best_k - 1][1]

        fig = Figure(figsize=(8, 4.2), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.plot(k_vals, f1s, color=ACCENT, linewidth=2.5, marker="o", markersize=5)
        ax.axvline(best_k, color=WARN, linestyle="--", linewidth=1.5,
                   label=f"Optimal K={best_k}")
        ax.scatter([best_k], [best_f], color=WARN, s=100, zorder=5)
        ax.annotate(f"  K={best_k}  F1={best_f:.4f}",
                    xy=(best_k, best_f), xytext=(best_k + 0.5, best_f - 0.007),
                    color=WARN, fontsize=9)
        ax.set_xlabel("K (Number of Neighbours)", fontsize=10)
        ax.set_ylabel("Weighted F1 Score",         fontsize=10)
        ax.set_title("Elbow Method — K vs F1 Score", color=TEXT, pad=12, fontsize=12)
        ax.set_xticks(k_vals)
        ax.legend(fontsize=9, facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT)
        ax.grid(True)

        frame = tk.Frame(self.tab_k_search, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # K table
        tbl = tk.Frame(self.tab_k_search, bg=SURFACE2,
                       highlightbackground=BORDER, highlightthickness=1)
        tbl.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(tbl, text="K Search Results", font=FONT_SMALL,
                 fg=MUTED, bg=SURFACE2).pack(anchor="w", padx=14, pady=(8, 4))
        grid = tk.Frame(tbl, bg=SURFACE2)
        grid.pack(fill=tk.X, padx=14, pady=(0, 10))
        sorted_res = sorted(r["k_results"], key=lambda x: -x[1])
        for idx, (k, f1) in enumerate(r["k_results"]):
            rank  = next(i+1 for i, (kk, _) in enumerate(sorted_res) if kk == k)
            col   = WARN if k == best_k else (ACCENT2 if rank <= 3 else TEXT)
            flag  = " OPTIMAL" if k == best_k else ""
            tk.Label(grid, text=f"K={k:<2}  {f1:.4f}  #{rank}{flag}",
                     font=FONT_SMALL, fg=col, bg=SURFACE2,
                     anchor="w").grid(row=idx // 4, column=idx % 4, sticky="w", padx=10, pady=1)

    # ── TAB 3: Confusion Matrix ─────────────────────────────────
    def _build_confusion(self):
        r      = self.results
        cm     = r["cm"]
        labels = [c.capitalize() for c in r["class_names"]]

        fig = Figure(figsize=(6.5, 4.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.55 else TEXT
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("Actual Class",    fontsize=11)
        ax.set_title("Confusion Matrix — Heatmap", color=TEXT, pad=14, fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalised Value", color=TEXT, fontsize=9)

        frame = tk.Frame(self.tab_confusion, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        leg = tk.Frame(self.tab_confusion, bg=SURFACE2,
                       highlightbackground=BORDER, highlightthickness=1)
        leg.pack(fill=tk.X, padx=12, pady=(0, 12))
        for icon, desc in [
            ("✅ Diagonal",      "Correct predictions"),
            ("❌ Off-diagonal",  "Misclassifications"),
            ("💡 Cell value",    "Raw count of test samples"),
        ]:
            row = tk.Frame(leg, bg=SURFACE2)
            row.pack(anchor="w", padx=14, pady=3)
            tk.Label(row, text=f"{icon:<20}", font=FONT_SMALL, fg=ACCENT, bg=SURFACE2).pack(side=tk.LEFT)
            tk.Label(row, text=desc, font=FONT_SMALL, fg=TEXT, bg=SURFACE2).pack(side=tk.LEFT)

    # ── TAB 4: Live Predictor ───────────────────────────────────
    def _build_predict(self):
        r      = self.results
        labels = r["feature_names"]

        outer = tk.Frame(self.tab_predict, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)

        tk.Label(outer, text="🌸  Live Iris Predictor",
                 font=FONT_HEAD, fg=ACCENT, bg=BG).pack(anchor="w", pady=(0, 4))
        tk.Label(outer, text="Adjust the sliders (cm) then hit CLASSIFY",
                 font=FONT_SMALL, fg=MUTED, bg=BG).pack(anchor="w", pady=(0, 16))

        self.feature_vars = []
        ranges   = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
        defaults = [5.8, 3.0, 4.0, 1.2]

        for label, (lo, hi), default in zip(labels, ranges, defaults):
            row = tk.Frame(outer, bg=BG)
            row.pack(fill=tk.X, pady=6)
            tk.Label(row, text=label, font=FONT_BODY,
                     fg=TEXT, bg=BG, width=26, anchor="w").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            self.feature_vars.append(var)
            tk.Scale(row, variable=var, from_=lo, to=hi,
                     resolution=0.1, orient=tk.HORIZONTAL, length=280,
                     bg=BG, fg=TEXT, troughcolor=SURFACE3,
                     activebackground=ACCENT, highlightthickness=0,
                     bd=0, font=FONT_SMALL, showvalue=False).pack(side=tk.LEFT, padx=8)
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     fg=ACCENT2, bg=BG, width=5).pack(side=tk.LEFT)

        tk.Button(outer, text="⚡  CLASSIFY",
                  font=FONT_HEAD, fg=BG, bg=ACCENT,
                  activebackground=ACCENT2, activeforeground=BG,
                  relief=tk.FLAT, padx=24, pady=10, cursor="hand2", bd=0,
                  command=self._do_predict).pack(pady=(20, 0), anchor="w")

        self.result_card = tk.Frame(outer, bg=SURFACE2,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.result_card.pack(fill=tk.X, pady=16)
        self.result_lbl = tk.Label(self.result_card,
                                   text="→ Result will appear here",
                                   font=FONT_HEAD, fg=MUTED, bg=SURFACE2)
        self.result_lbl.pack(pady=12, padx=16, anchor="w")

        self.prob_bars   = []
        self.prob_labels = []
        for i, cls in enumerate(r["class_names"]):
            row = tk.Frame(outer, bg=BG)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=f"{cls.capitalize():<14}",
                     font=FONT_SMALL, fg=TEXT, bg=BG).pack(side=tk.LEFT)
            bar_bg = tk.Frame(row, bg=SURFACE2, height=14, width=300,
                              highlightbackground=BORDER, highlightthickness=1)
            bar_bg.pack(side=tk.LEFT, padx=6)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=CLASS_COLORS[i], height=14, width=0)
            bar.pack(side=tk.LEFT, fill=tk.Y)
            lbl = tk.Label(row, text="", font=FONT_SMALL,
                           fg=CLASS_COLORS[i], bg=BG)
            lbl.pack(side=tk.LEFT, padx=4)
            self.prob_bars.append((bar, bar_bg))
            self.prob_labels.append(lbl)

    def _do_predict(self):
        features         = [v.get() for v in self.feature_vars]
        cls, proba       = predict_single(
            self.results["model"], self.results["scaler"],
            features, self.results["class_names"]
        )
        color = CLASS_COLORS[self.results["class_names"].index(cls)]
        self.result_lbl.config(
            text=f"→  Predicted: {cls.upper()}  ({proba.max()*100:.1f}% confidence)",
            fg=color
        )
        for i, (prob, (bar, bar_bg)) in enumerate(zip(proba, self.prob_bars)):
            bar.config(width=max(int(prob * 300), 0))
            self.prob_labels[i].config(text=f"{prob*100:.1f}%")


def main():
    root = tk.Tk()
    KNNApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()