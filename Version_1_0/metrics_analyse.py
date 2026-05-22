import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_watermark_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    expected_cols = [
        "video_id",
        "video_uuid",
        "attack_name",
        "psnr_mean",
        "ssim_mean",
        "edit_distance",
        "normalized_edit_distance",
        "p_value",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["psnr_mean", "ssim_mean", "edit_distance", "normalized_edit_distance", "p_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_clean"] = df["attack_name"].eq("watermarked")
    df["is_attacked"] = ~df["is_clean"]

    df["detected_p05"] = df["p_value"] < 0.05
    df["detected_p01"] = df["p_value"] < 0.01

    out = {}

    out["rows_total"] = len(df)
    out["videos_unique"] = int(df["video_uuid"].nunique())

    clean = df[df["is_clean"]].copy()
    attacked = df[df["is_attacked"]].copy()

    out["clean_summary"] = clean[
        ["psnr_mean", "ssim_mean", "edit_distance", "normalized_edit_distance", "p_value"]
    ].agg(["count", "mean", "std", "median", "min", "max"]).round(6)

    out["attacked_summary"] = attacked[
        ["edit_distance", "normalized_edit_distance", "p_value"]
    ].agg(["count", "mean", "std", "median", "min", "max"]).round(6)

    out["clean_detection_rate_p05"] = float(clean["detected_p05"].mean()) if len(clean) else np.nan
    out["clean_detection_rate_p01"] = float(clean["detected_p01"].mean()) if len(clean) else np.nan
    out["attacked_detection_rate_p05"] = float(attacked["detected_p05"].mean()) if len(attacked) else np.nan
    out["attacked_detection_rate_p01"] = float(attacked["detected_p01"].mean()) if len(attacked) else np.nan

    attack_stats = attacked.groupby("attack_name").agg(
        n=("attack_name", "size"),
        edit_distance_mean=("edit_distance", "mean"),
        edit_distance_median=("edit_distance", "median"),
        normalized_edit_distance_mean=("normalized_edit_distance", "mean"),
        normalized_edit_distance_median=("normalized_edit_distance", "median"),
        p_value_mean=("p_value", "mean"),
        p_value_median=("p_value", "median"),
        detection_rate_p05=("detected_p05", "mean"),
        detection_rate_p01=("detected_p01", "mean"),
    ).sort_values(
        ["detection_rate_p05", "normalized_edit_distance_mean"],
        ascending=[False, True]
    ).round(6)

    out["attack_stats"] = attack_stats

    per_video = df.groupby("video_uuid").agg(
        video_id=("video_id", "first"),
        rows=("video_uuid", "size"),
        n_attacks=("is_attacked", "sum"),
        clean_present=("is_clean", "max"),
    ).copy()

    clean_per_video = clean.groupby("video_uuid").agg(
        clean_psnr_mean=("psnr_mean", "mean"),
        clean_ssim_mean=("ssim_mean", "mean"),
        clean_p_value_mean=("p_value", "mean"),
    )

    attacked_per_video = attacked.groupby("video_uuid").agg(
        attacked_norm_edit_mean=("normalized_edit_distance", "mean"),
        attacked_edit_distance_mean=("edit_distance", "mean"),
        attacked_p_value_mean=("p_value", "mean"),
        attacked_detect_rate_p05=("detected_p05", "mean"),
    )

    per_video = per_video.join(clean_per_video, how="left").join(attacked_per_video, how="left")
    out["per_video"] = per_video.round(6)

    return df, out


def save_analysis_outputs(csv_path: str, out_dir: str = "analysis_output"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, out = analyze_watermark_csv(csv_path)

    out["clean_summary"].to_csv(out_dir / "clean_summary.csv")
    out["attacked_summary"].to_csv(out_dir / "attacked_summary.csv")
    out["attack_stats"].to_csv(out_dir / "attack_stats.csv")
    out["per_video"].to_csv(out_dir / "per_video_stats.csv")

    overview = pd.DataFrame([
        {"metric": "rows_total", "value": out["rows_total"]},
        {"metric": "videos_unique", "value": out["videos_unique"]},
        {"metric": "clean_detection_rate_p05", "value": out["clean_detection_rate_p05"]},
        {"metric": "clean_detection_rate_p01", "value": out["clean_detection_rate_p01"]},
        {"metric": "attacked_detection_rate_p05", "value": out["attacked_detection_rate_p05"]},
        {"metric": "attacked_detection_rate_p01", "value": out["attacked_detection_rate_p01"]},
    ])
    overview.to_csv(out_dir / "overview.csv", index=False)

    attack_stats = out["attack_stats"].reset_index()

    if len(attack_stats):
        # 1. Detection rate by attack
        plt.figure(figsize=(10, 5))
        order = attack_stats.sort_values("detection_rate_p05", ascending=False)
        plt.bar(order["attack_name"], order["detection_rate_p05"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Detection rate (p < 0.05)")
        plt.title("Watermark detection rate by attack")
        plt.tight_layout()
        plt.savefig(out_dir / "detection_rate_by_attack.png", dpi=180)
        plt.close()

        # 2. Normalized edit distance by attack
        plt.figure(figsize=(10, 5))
        order = attack_stats.sort_values("normalized_edit_distance_mean", ascending=True)
        plt.bar(order["attack_name"], order["normalized_edit_distance_mean"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean normalized edit distance")
        plt.title("Message distortion by attack")
        plt.tight_layout()
        plt.savefig(out_dir / "normalized_edit_distance_by_attack.png", dpi=180)
        plt.close()

        # 3. Median p-value by attack
        plt.figure(figsize=(10, 5))
        order = attack_stats.sort_values("p_value_median", ascending=True)
        plt.bar(order["attack_name"], order["p_value_median"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Median p-value")
        plt.title("Statistical significance by attack")
        plt.tight_layout()
        plt.savefig(out_dir / "pvalue_by_attack.png", dpi=180)
        plt.close()

        # 4. Boxplot: normalized edit distance by attack
        plt.figure(figsize=(11, 5))
        plot_df = df[df["is_attacked"]].copy()
        attack_order = (
            plot_df.groupby("attack_name")["normalized_edit_distance"]
            .mean()
            .sort_values()
            .index
        )
        data = [
            plot_df.loc[plot_df["attack_name"] == attack, "normalized_edit_distance"].dropna().values
            for attack in attack_order
        ]
        plt.boxplot(data, tick_labels=list(attack_order), showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Normalized edit distance")
        plt.title("Distribution of message distortion by attack")
        plt.tight_layout()
        plt.savefig(out_dir / "normalized_edit_distance_boxplot.png", dpi=180)
        plt.close()

    clean = df[df["is_clean"]].copy()
    if len(clean) and clean["psnr_mean"].notna().any() and clean["ssim_mean"].notna().any():
        # 5. PSNR vs SSIM for clean watermarked videos
        plt.figure(figsize=(6, 5))
        plt.scatter(clean["psnr_mean"], clean["ssim_mean"], alpha=0.8)
        plt.xlabel("PSNR mean")
        plt.ylabel("SSIM mean")
        plt.title("Clean watermarked video quality")
        plt.tight_layout()
        plt.savefig(out_dir / "clean_psnr_ssim_scatter.png", dpi=180)
        plt.close()

    return out_dir


# пример запуска
if __name__ == "__main__":
    save_analysis_outputs(
        csv_path="outputs/metrics/metrics.csv",
        out_dir="outputs/metrics/analysis"
    )

