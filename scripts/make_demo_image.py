"""Generate a small demo image for attribute-classifier smoke runs."""

from pathlib import Path


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("error: Pillow is required (pip install 'odd-suite[attr]')")
        return 2

    target = Path(__file__).resolve().parent.parent / "examples" / "images" / "demo.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (224, 224))
    pixels = image.load()
    for y in range(224):
        for x in range(224):
            pixels[x, y] = (90 + x % 60, 110 + y % 70, 140 + (x + y) % 50)
    image.save(target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
