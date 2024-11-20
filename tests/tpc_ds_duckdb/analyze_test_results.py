from pathlib import Path
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tomllib
from os import environ

# https://github.com/python/cpython/issues/125235#issuecomment-2412948604
if not environ.get("TCL_LIBRARY"):
    sys_path = Path(sys.base_prefix)
    environ["TCL_LIBRARY"] = str(sys_path / "tcl" / "tcl8.6")


def analyze(show: bool = False):

    results = []
    root = Path(__file__).parent
    for filename in os.listdir(root):
        if filename.endswith(".log"):
            with open(root / filename, "r") as f:
                try:
                    loaded = tomllib.loads(f.read())
                except json.decoder.JSONDecodeError:
                    print(f"Error loading {filename}")
                    continue
                results.append(loaded)

    df = pd.DataFrame.from_records(results)
    df = df.dropna(
        subset=[
            "cases",
        ]
    )
    df = df.drop(
        columns=[
            "parse_time",
            "exec_time",
            "comp_time",
            "gen_length",
            "generated_sql",
        ]
    )
    rows = []
    for _, row in df.iterrows():
        for case_type, success_rate in row["cases"].items():
            rows.append(
                {
                    "query_id": row["query_id"],
                    "case": row["model"] + "-" + case_type,
                    "success_rate": success_rate,
                    "avg_duration": sum(row["durations"].get(case_type, [1]))
                    / len(row["durations"].get(case_type, [1])),
                }
            )

    # Create the transformed DataFrame
    df_transformed = pd.DataFrame(rows)

    # Plot the bar chart
    df_transformed["query_id"] = pd.Categorical(df_transformed["query_id"])
    sns.catplot(
        data=df_transformed,
        kind="bar",
        x="success_rate",
        y="query_id",  # `query_id` on the y-axis
        hue="case",  # Separate bars by `model`
        palette="coolwarm",
        dodge=True,
        height=5,
        aspect=1.2,
    )
    # Customize plot
    plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the title
    plt.suptitle("Success Rate by Query ID and Model Breakdown", fontsize=16)

    # Plot average durations
    # grouped["avg_duration"].unstack().plot(kind="bar", title="Average Duration by Query ID and Difficulty")
    # plt.ylabel("Average Duration")
    # plt.show()
    if show:
        plt.show()
    else:
        plt.savefig(root / "tpc-ds-perf.png")
    sns.catplot(
        data=df_transformed,
        kind="bar",
        x="avg_duration",
        y="query_id",  # `query_id` on the y-axis
        hue="case",  # Separate bars by `model`
        palette="coolwarm",
        dodge=True,
        height=5,
        aspect=1.2,
    )
    # Customize plot
    plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the title
    plt.suptitle("Average Timing by Query ID and Model Breakdown", fontsize=16)
    if show:
        plt.show()
    else:
        plt.savefig(root / "tpc-ds-timing.png")


if __name__ == "__main__":
    analyze(show=True)
