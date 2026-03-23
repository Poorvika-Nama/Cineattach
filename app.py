from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE     = os.path.dirname(__file__)
pipeline = joblib.load(os.path.join(BASE, "regression_pipeline.pkl"))

CSV_PATH = os.path.join(BASE, "Movie_vs_Emotional_Attachment_Survey_2024-2026.csv")
df_raw   = pd.read_csv(CSV_PATH)
df_raw   = df_raw.rename(columns={
    "Movie/Book that affected you most recently"                           : "movie_title",
    "The storytelling in this movie was emotionally powerful."             : "ps",
    "The story made me feel strong emotions."                              : "se",
    "The emotional moments felt genuine and realistic."                    : "fg",
    "The characters' emotional experiences were clearly expressed."        : "ce",
    "The story focused strongly on characters' feelings and relationships.": "ff",
    "Emotional Storytelling Score"                                         : "es",
    "Audience Attachment Score"                                            : "attachment_score",
    "Age"                                                                  : "age",
    "Gender"                                                               : "gender",
    "How many times watched"                                               : "times_watched",
    "When watched most recently"                                           : "when_watched",
})

# ── Per-movie averages ─────────────────────────────────────
MOVIE_DATA = (
    df_raw.groupby("movie_title")
          .agg(
              ps             = ("ps",               "mean"),
              se             = ("se",               "mean"),
              fg             = ("fg",               "mean"),
              ce             = ("ce",               "mean"),
              ff             = ("ff",               "mean"),
              es             = ("es",               "mean"),
              avg_attachment = ("attachment_score", "mean"),
              pop            = ("movie_title",      "count"),
          )
          .round(3)
          .to_dict(orient="index")
)

MOVIE_LIST      = sorted(MOVIE_DATA.keys())
DEFAULT_POP     = int(df_raw["movie_title"].value_counts().median())
DEFAULT_AVG_ATT = round(float(df_raw["attachment_score"].mean()), 3)
DEFAULT_RATINGS = {
    "ps": round(float(df_raw["ps"].mean()), 3),
    "se": round(float(df_raw["se"].mean()), 3),
    "fg": round(float(df_raw["fg"].mean()), 3),
    "ce": round(float(df_raw["ce"].mean()), 3),
    "ff": round(float(df_raw["ff"].mean()), 3),
    "es": round(float(df_raw["es"].mean()), 3),
    "avg_attachment": DEFAULT_AVG_ATT,
    "pop": DEFAULT_POP,
}

# ── Global survey benchmarks (pre-computed) ───────────────
AGES    = ["Under 18", "18–20", "21–23", "24 or above"]
GENDERS = ["Female", "Male", "Prefer not to say"]

# Overall avg attachment by age and gender (actual survey data)
age_gender_avg = (
    df_raw.groupby(["age", "gender"])["attachment_score"]
          .mean().round(3)
)

# Score percentiles from the full 10k dataset
PERCENTILES = {
    int(p): round(float(df_raw["attachment_score"].quantile(p/100)), 2)
    for p in range(0, 101, 5)
}

print(f"✅ Loaded {len(MOVIE_LIST)} movies from CSV")


def score_label(score):
    if score < 2.0:  return "Very Low",  "#E24B4A"
    if score < 3.0:  return "Low",       "#EF9F27"
    if score < 3.75: return "Moderate",  "#378ADD"
    if score < 4.5:  return "High",      "#1D9E75"
    return             "Very High",      "#534AB7"


def get_percentile_rank(score):
    """What % of survey respondents scored BELOW this score."""
    below = (df_raw["attachment_score"] < score).mean()
    return round(float(below) * 100, 1)


@app.route("/")
def index():
    return render_template("index.html", movies=MOVIE_LIST)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f          = request.form
        movie_name = f.get("movie_name", "").strip()
        age        = f["age"]
        gender     = f["gender"]

        m     = MOVIE_DATA.get(movie_name, DEFAULT_RATINGS)
        known = movie_name in MOVIE_DATA

        times_watched     = "1 time"
        when_watched      = "1–6 months ago"
        times_map         = {"1 time":1,"2–3 times":2,"More than 3 times":3}
        when_map          = {"More than 6 months ago":1,"1–6 months ago":2,"Within last 1 month":3}
        rewatch_x_recency = times_map[times_watched] * when_map[when_watched]

        # ── Prediction for user's exact inputs ────────────
        def predict_row(a, g):
            row = pd.DataFrame([{
                "age"                  : a,
                "times_watched"        : times_watched,
                "when_watched"         : when_watched,
                "gender"               : g,
                "ps"                   : m["ps"],
                "se"                   : m["se"],
                "fg"                   : m["fg"],
                "ce"                   : m["ce"],
                "ff"                   : m["ff"],
                "emotion_score"        : m["es"],
                "movie_popularity"     : m["pop"],
                "rewatch_x_recency"    : rewatch_x_recency,
                "movie_avg_attachment" : m["avg_attachment"],
            }])
            return round(float(np.clip(pipeline.predict(row)[0], 1.0, 5.0)), 2)

        score      = predict_row(age, gender)
        label, col = score_label(score)

        # ── Chart 1: predicted score by age (user's gender fixed) ──
        age_chart = {
            "labels": AGES,
            "values": [predict_row(a, gender) for a in AGES],
            "highlight": AGES.index(age),
        }

        # ── Chart 2: predicted score by gender (user's age fixed) ──
        gender_chart = {
            "labels": GENDERS,
            "values": [predict_row(age, g) for g in GENDERS],
            "highlight": GENDERS.index(gender),
        }

        # ── Chart 3: distribution — where does user's score sit ──
        dist_labels = [f"{v:.1f}" for v in [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]]
        bins        = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]
        hist, _     = np.histogram(df_raw["attachment_score"].values, bins=bins)
        dist_values = [int(v) for v in hist]

        # Which bin does the user's score fall in?
        user_bin = int(np.digitize(score, bins) - 1)
        user_bin = max(0, min(user_bin, len(dist_values)-1))

        dist_chart = {
            "labels"    : dist_labels,
            "values"    : dist_values,
            "highlight" : user_bin,
        }

        pct_rank = get_percentile_rank(score)

        return jsonify({
            "score"        : score,
            "label"        : label,
            "color"        : col,
            "percent"      : round((score - 1) / 4 * 100, 1),
            "pct_rank"     : pct_rank,
            "movie_name"   : movie_name if movie_name else "Your movie",
            "known_movie"  : known,
            "emotion_score": round(m["es"], 2),
            "popularity"   : int(m["pop"]),
            "age_chart"    : age_chart,
            "gender_chart" : gender_chart,
            "dist_chart"   : dist_chart,
            "user_age"     : age,
            "user_gender"  : gender,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
