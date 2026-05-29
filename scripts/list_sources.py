#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bizembed.sources import DATA_SOURCES


def main() -> None:
    rows = [
        {
            "name": source.name,
            "title": source.title,
            "method": source.method,
            "source": source.source,
            "requires_service_key": source.requires_service_key,
            "page_url": source.page_url,
            "fields": list(source.fields),
            "notes": source.notes,
        }
        for source in DATA_SOURCES
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
