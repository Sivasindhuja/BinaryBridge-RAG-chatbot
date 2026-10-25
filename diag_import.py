import traceback
import warnings
from pathlib import Path


def main() -> None:
    output_path = Path(__file__).resolve().parent / "full_traceback.txt"

    with output_path.open("w", encoding="utf-8") as handle:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import google.generativeai  # type: ignore

            handle.write("google.generativeai imported successfully!\n")
        except Exception:
            traceback.print_exc(file=handle)
            handle.write(
                "\nThe main project can still run because RAG.py now falls back to a local answer generator.\n"
            )


if __name__ == "__main__":
    main()
