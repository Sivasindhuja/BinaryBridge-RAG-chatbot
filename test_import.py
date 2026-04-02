import sys
import warnings


def main() -> int:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai  # type: ignore

        print("google.generativeai imported successfully!")
        return 0
    except Exception as exc:
        print(f"google.generativeai is not available in this environment: {exc}")
        print("The project still works with the local fallback pipeline.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
