"""Check installed dependency licenses against the repository policy."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

LOGGER = logging.getLogger(__name__)
POLICY_PATH = Path(__file__).resolve().parents[1] / "license-policy.json"
AGPL_PATTERN = re.compile(
    r"\bAGPL(?:\b|v?\d)|affero\s+general\s+public\s+license", re.IGNORECASE
)
SSPL_PATTERN = re.compile(
    r"\bSSPL(?:\b|v?\d)|server\s+side\s+public\s+license", re.IGNORECASE
)
GPL_PATTERN = re.compile(r"\bGPL(?:\b|v?\d)", re.IGNORECASE)


class LicenseRecord(TypedDict, total=False):
    """License information for one package in the pip-licenses report."""

    Name: str
    License: str


@dataclass(frozen=True)
class LicensePolicy:
    """Validated license labels and package-specific exceptions.

    Attributes:
        approved_labels: License labels accepted without an exception.
        package_exceptions: Exact package and license pairs with approval reasons.
    """

    approved_labels: frozenset[str]
    package_exceptions: dict[tuple[str, str], str]


def normalize_package_name(name: str) -> str:
    """Normalize a package name for exact exception matching.

    Args:
        name: Package name from dependency metadata.

    Returns:
        A lowercase package name with equivalent separators normalized.
    """
    return re.sub(r"[-_.]+", "-", name).casefold()


def load_policy(path: Path) -> LicensePolicy:
    """Load and validate the repository license policy.

    Args:
        path: Path to the JSON policy file.

    Returns:
        The validated license policy.

    Raises:
        OSError: If the policy file cannot be read.
        json.JSONDecodeError: If the policy file is not valid JSON.
        TypeError: If the policy contains values of invalid types.
        ValueError: If the policy contains invalid or duplicate entries.
    """
    raw_policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_policy, dict):
        raise TypeError("License policy must be a JSON object.")

    labels = raw_policy.get("approved_license_labels")
    exceptions = raw_policy.get("package_exceptions")
    if not isinstance(labels, list) or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError("approved_license_labels must be a list of non-empty strings.")
    if not isinstance(exceptions, list):
        raise TypeError("package_exceptions must be a list.")

    exception_map: dict[tuple[str, str], str] = {}
    for item in exceptions:
        if not isinstance(item, dict):
            raise TypeError("Each package exception must be a JSON object.")
        package = item.get("package")
        license_label = item.get("license")
        reason = item.get("reason")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (package, license_label, reason)
        ):
            raise ValueError(
                "Each package exception requires non-empty package, license, and reason."
            )
        key = (normalize_package_name(package), license_label)
        if key in exception_map:
            raise ValueError(
                f"Duplicate package exception for {package}: {license_label}"
            )
        exception_map[key] = reason

    return LicensePolicy(
        approved_labels=frozenset(labels),
        package_exceptions=exception_map,
    )


def check_license_report(
    records: list[LicenseRecord], policy: LicensePolicy
) -> list[str]:
    """Find policy violations in a pip-licenses JSON report.

    Args:
        records: Package license records from the report.
        policy: Approved license labels and package-specific exceptions.

    Returns:
        Policy violation messages, or an empty list when all records pass.
    """
    errors: list[str] = []
    for record in records:
        name = record.get("Name")
        license_label = record.get("License")
        if not isinstance(name, str) or not name.strip():
            errors.append("Report contains an entry without a package name.")
            continue
        if not isinstance(license_label, str) or not license_label.strip():
            errors.append(f"{name}: license metadata is missing.")
            continue

        license_label = license_label.strip()
        if AGPL_PATTERN.search(license_label):
            errors.append(f"{name}: AGPL license is not allowed ({license_label}).")
            continue
        if SSPL_PATTERN.search(license_label):
            errors.append(f"{name}: SSPL license is not allowed ({license_label}).")
            continue

        exception = policy.package_exceptions.get(
            (normalize_package_name(name), license_label)
        )
        if exception:
            continue
        if "proprietary" in license_label.casefold():
            errors.append(
                f"{name}: proprietary license requires a package-specific approval "
                f"({license_label})."
            )
            continue
        if GPL_PATTERN.search(license_label):
            continue
        if license_label not in policy.approved_labels:
            errors.append(
                f"{name}: unrecognized license label requires review ({license_label})."
            )

    return errors


def main() -> int:
    """Run the dependency license policy check.

    Returns:
        Exit code 0 on success, 1 for policy violations, or 2 for invalid input.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python scripts/check_licenses.py <license-report.json>")
        return 2

    report_path = Path(sys.argv[1])
    try:
        records = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("License report must be a JSON list of package objects.")
        violations = check_license_report(
            records=records,
            policy=load_policy(POLICY_PATH),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2

    if violations:
        for violation in violations:
            LOGGER.error("%s", violation)
        return 1

    LOGGER.info("Dependency license policy check passed for %d packages.", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
