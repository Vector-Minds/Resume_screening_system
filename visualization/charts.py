import matplotlib.pyplot as plt

def plot_match_scores(results):
    fig, ax = plt.subplots(figsize=(10, max(3, len(results) * 0.6)))

    resume_names = [r["Resume Name"].replace(".pdf", "") for r in results]
    match_scores = [r["Match Score (%)"] for r in results]

    colors = ["#2ecc71" if s >= 70 else "#f39c12" if s >= 40 else "#e74c3c" for s in match_scores]

    bars = ax.barh(resume_names, match_scores, color=colors, edgecolor="black")

    for i, (bar, score) in enumerate(zip(bars, match_scores)):
        ax.text(score + 1.5, i, f"{score}%", va="center", fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Match Score (%)")
    ax.set_title("Resume Match Scores")
    ax.grid(axis="x", alpha=0.3)

    return fig