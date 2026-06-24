"""
main.py  –  Entry point for the Iris Data Classification model.

Runs the training pipeline on startup, prints validation metrics, and launches
an interactive CLI for custom predictions.
"""

import sys
from data_classification import IrisClassifier


# ── ANSI colour helpers ──────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GREY   = "\033[90m"
    BLUE   = "\033[94m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}
  ╔══════════════════════════════════════════╗
  ║      Iris Data Classification AI 🌸      ║
  ║      Model: K-Nearest Neighbors (KNN)    ║
  ╚══════════════════════════════════════════╝
{C.RESET}
  {C.GREY}Type 'stats' to view dataset splits.
  Type 'train' to re-train the model with a different K.
  Type 'exit' or 'quit' to quit.{C.RESET}
""")


def print_metrics(metrics: dict, target_names: list[str]):
    print(f"\n  {C.CYAN}{C.BOLD}📊 Model Performance Metrics (Test Set):{C.RESET}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Overall Accuracy:  {C.GREEN}{metrics['accuracy']:.4%}{C.RESET}")
    print(f"  Macro Precision:   {C.GREEN}{metrics['precision']:.4%}{C.RESET}")
    print(f"  Macro Recall:      {C.GREEN}{metrics['recall']:.4%}{C.RESET}")
    print(f"  Macro F1-Score:    {C.GREEN}{metrics['f1_score']:.4%}{C.RESET}\n")

    print(f"  {C.BOLD}Per-Class Performance:{C.RESET}")
    print(f"  {C.BOLD}{'Species':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}{C.RESET}")
    for idx, name in enumerate(target_names):
        p = metrics['per_class_precision'][idx]
        r = metrics['per_class_recall'][idx]
        f = metrics['per_class_f1_score'][idx]
        print(f"  {name.capitalize():<15} {p:.2%}     {r:.2%}     {f:.2%}")

    print(f"\n  {C.BOLD}Confusion Matrix:{C.RESET}")
    print(f"  {C.GREY}                     Predicted{C.RESET}")
    print(f"  {C.GREY}                  Set  Ver  Vir{C.RESET}")
    cm = metrics['confusion_matrix']
    for i, name in enumerate(["Setosa", "Versicolor", "Virginica"]):
        row_str = "  ".join(f"{val:>3}" for val in cm[i])
        # Add labels for Actual/Predicted
        label = "Actual " if i == 0 else "       "
        print(f"  {C.GREY}{label}{C.RESET}{name:<12} [{row_str}]")
    print()


def print_stats(clf: IrisClassifier):
    print(f"\n  {C.YELLOW}── Dataset Splits & Info ──────────────────{C.RESET}")
    print(f"  Total samples:      {len(clf.X)}")
    print(f"  Training set size:  {len(clf.X_train)} samples")
    print(f"  Testing set size:   {len(clf.X_test)} samples")
    print(f"  Feature set:        {', '.join(clf.feature_names)}")
    print(f"  Classes:            {', '.join(clf.target_names)}")
    print(f"  KNN Neighbors (K):  {clf.n_neighbors}")
    print(f"  Test Split Ratio:   {clf.test_size:.0%}")
    print(f"  StandardScaler:     Applied (fitted on train)\n")


def get_numeric_input(prompt: str) -> float:
    while True:
        val_str = input(prompt).strip()
        
        # Check command exits inside numeric inputs
        if val_str.lower() in ["exit", "quit"]:
            print(f"\n  {C.CYAN}Exiting. Goodbye! 🌸{C.RESET}")
            sys.exit(0)
            
        try:
            val = float(val_str)
            if val <= 0:
                print(f"  {C.RED}Error: Measurement must be positive. Please try again.{C.RESET}")
                continue
            return val
        except ValueError:
            print(f"  {C.RED}Error: Invalid number. Please enter a decimal or integer.{C.RESET}")


def run_pipeline(k_neighbors: int = 5) -> IrisClassifier:
    clf = IrisClassifier(n_neighbors=k_neighbors)
    clf.load_data()
    clf.split_data()
    clf.scale_features()
    clf.train()
    return clf


def main():
    banner()
    print(f"  {C.YELLOW}Training KNN Classifier on Iris dataset...{C.RESET}")
    clf = run_pipeline()
    metrics = clf.evaluate()
    print_metrics(metrics, clf.target_names)
    
    print(f"  {C.BOLD}Interactive Custom Prediction:{C.RESET}")
    print(f"  Provide flower measurements to predict its species.\n")

    while True:
        try:
            print(f"  {C.CYAN}Enter measurements (in cm) or special command:{C.RESET}")
            
            # First prompt is sepal length or general command check
            first_input = input(f"  {C.GREEN}Sepal Length >{C.RESET} ").strip()
            
            if not first_input:
                continue
                
            cmd = first_input.lower()
            if cmd in ["exit", "quit"]:
                print(f"\n  {C.CYAN}Exiting. Goodbye! 🌸{C.RESET}")
                break
                
            if cmd == "stats":
                print_stats(clf)
                continue
                
            if cmd == "train":
                try:
                    k_str = input(f"  {C.YELLOW}Enter new value for K (current: {clf.n_neighbors}) >{C.RESET} ").strip()
                    k = int(k_str)
                    if k <= 0:
                        print(f"  {C.RED}Error: K must be positive.{C.RESET}\n")
                        continue
                    clf = run_pipeline(k_neighbors=k)
                    metrics = clf.evaluate()
                    print_metrics(metrics, clf.target_names)
                except ValueError:
                    print(f"  {C.RED}Error: Invalid integer for K.{C.RESET}\n")
                continue
            
            # If not a command, parse as float
            try:
                sepal_len = float(first_input)
                if sepal_len <= 0:
                    print(f"  {C.RED}Error: Measurement must be positive. Restarting prompt.{C.RESET}\n")
                    continue
            except ValueError:
                print(f"  {C.RED}Error: Invalid input. Type 'stats', 'train', 'exit' or enter a positive number.{C.RESET}\n")
                continue
                
            # Get rest of numeric inputs
            sepal_wid = get_numeric_input(f"  {C.GREEN}Sepal Width  >{C.RESET} ")
            petal_len = get_numeric_input(f"  {C.GREEN}Petal Length >{C.RESET} ")
            petal_wid = get_numeric_input(f"  {C.GREEN}Petal Width  >{C.RESET} ")
            
            # Predict
            raw_feats = [sepal_len, sepal_wid, petal_len, petal_wid]
            species = clf.predict(raw_feats)
            
            print(f"\n  {C.YELLOW}── Prediction Result ──────────────────────{C.RESET}")
            print(f"  Predicted Species: {C.GREEN}{C.BOLD}{species.upper()}{C.RESET}")
            print(f"  Input Features:    Sepal: {sepal_len}x{sepal_wid} cm, Petal: {petal_len}x{petal_wid} cm")
            print(f"  ───────────────────────────────────────────\n")
            
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {C.CYAN}Session interrupted. Goodbye! 🌸{C.RESET}")
            break


if __name__ == "__main__":
    main()
