import matplotlib.pyplot as plt

from src.config import OUTPUTS_DIR


def plot_predictions(y_test, predictions):
    """Plot actual vs predicted CO₂ emissions."""

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6,
        label="Predictions"
    )

    # Perfect prediction reference line
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual CO₂ Emissions (g/km)")
    plt.ylabel("Predicted CO₂ Emissions (g/km)")
    plt.title("Multiple Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUTS_DIR / "actual_vs_predicted.png",
        dpi=300
    )

    plt.show()