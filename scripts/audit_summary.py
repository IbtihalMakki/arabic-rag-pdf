import json
from pathlib import Path


def one_line(text: str, max_len: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main() -> None:
    data = json.loads(Path("audit_results.json").read_text(encoding="utf-8"))

    print("--- Required 10 ---")
    for i, record in enumerate(data["records"][:10], start=1):
        print(
            f"{i}. expected={record['expected_class']} behavior={record['observed_behavior']} "
            f"decision={record['decision']}"
        )
        print("   Q:", record["question"])
        print("   A:", one_line(record["final_answer"]))

    print("\n--- Stress 7 ---")
    for record in data["records"][10:17]:
        print(f"behavior={record['observed_behavior']} decision={record['decision']}")
        print("   Q:", record["question"])
        print("   A:", one_line(record["final_answer"]))


if __name__ == "__main__":
    main()
