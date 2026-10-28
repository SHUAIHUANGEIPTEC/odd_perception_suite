#!/usr/bin/env bash
# Format first-party native sources with clang-format (see .clang-format).
# vendor/sparse4d/ is a verbatim vendored delivery and is intentionally excluded.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v clang-format >/dev/null 2>&1; then
    echo "error: clang-format not found (install clang-format / clang-format-17)" >&2
    exit 2
fi

mapfile -t files < <(
    find . -path ./vendor -prune -o \
         \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) \
         -print
)

if [ ${#files[@]} -eq 0 ]; then
    echo "no first-party native sources found (vendor/ excluded)"
    exit 0
fi

printf '%s\n' "${files[@]}" | xargs clang-format -i
echo "formatted ${#files[@]} file(s) with $(clang-format --version)"
