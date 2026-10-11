#!/usr/bin/env python
"""Create disjoint CARLA route XML train/evaluation splits by route ID."""

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


def write_routes(routes, output: Path) -> None:
    root = ET.Element("routes")
    for route in routes:
        root.append(copy.deepcopy(route))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xml", type=Path, required=True)
    parser.add_argument("--holdout_ids", required=True, help="Comma-separated route IDs")
    parser.add_argument("--train_output", type=Path, required=True)
    parser.add_argument("--holdout_output", type=Path, required=True)
    args = parser.parse_args()

    holdout_ids = {item.strip() for item in args.holdout_ids.split(",") if item.strip()}
    if not holdout_ids:
        raise SystemExit("--holdout_ids must not be empty")
    root = ET.parse(args.input_xml).getroot()
    routes = list(root.findall("route"))
    seen_ids = [route.attrib.get("id") for route in routes]
    missing = sorted(holdout_ids - set(seen_ids))
    if missing:
        raise SystemExit(f"holdout route IDs absent from input: {','.join(missing)}")
    if len(seen_ids) != len(set(seen_ids)):
        raise SystemExit("input XML contains duplicate route IDs")

    holdout = [route for route in routes if route.attrib["id"] in holdout_ids]
    train = [route for route in routes if route.attrib["id"] not in holdout_ids]
    if not train or not holdout:
        raise SystemExit("both train and holdout splits must contain at least one route")
    write_routes(train, args.train_output)
    write_routes(holdout, args.holdout_output)
    print(
        f"ROUTE_SPLIT_OK train={len(train)} holdout={len(holdout)} "
        f"train_output={args.train_output} holdout_output={args.holdout_output}"
    )


if __name__ == "__main__":
    main()
