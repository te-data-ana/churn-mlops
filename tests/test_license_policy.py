import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_licenses.py"


def run_license_check(
    tmp_path: Path, record: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    report_path = tmp_path / "license-report.json"
    report_path.write_text(json.dumps([record]), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), str(report_path)],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_gpl_is_reported_but_not_blocked(tmp_path: Path) -> None:
    result = run_license_check(
        tmp_path=tmp_path,
        record={"Name": "dev-tool", "License": "GNU General Public License v2 (GPLv2)"},
    )

    assert result.returncode == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("package", "license_label", "message"),
    [
        ("agpl-package", "AGPL-3.0-only", "AGPL"),
        ("agpl-package", "AGPLv3", "AGPL"),
        ("agpl-package", "GNU Affero General Public License v3", "AGPL"),
        ("sspl-package", "SSPL-1.0", "SSPL"),
        ("sspl-package", "Server Side Public License", "SSPL"),
    ],
)
def test_agpl_and_sspl_are_blocked(
    tmp_path: Path, package: str, license_label: str, message: str
) -> None:
    result = run_license_check(
        tmp_path=tmp_path,
        record={"Name": package, "License": license_label},
    )

    assert result.returncode == 1
    assert message in result.stderr


@pytest.mark.unit
def test_proprietary_license_exception_is_package_specific(tmp_path: Path) -> None:
    approved = run_license_check(
        tmp_path=tmp_path,
        record={
            "Name": "nvidia-nccl-cu12",
            "License": "LicenseRef-NVIDIA-Proprietary",
        },
    )
    unexpected = run_license_check(
        tmp_path=tmp_path,
        record={
            "Name": "another-package",
            "License": "LicenseRef-NVIDIA-Proprietary",
        },
    )

    assert approved.returncode == 0
    assert unexpected.returncode == 1
    assert "package-specific approval" in unexpected.stderr


@pytest.mark.unit
def test_unrecognized_license_label_requires_review(tmp_path: Path) -> None:
    result = run_license_check(
        tmp_path=tmp_path,
        record={"Name": "new-package", "License": "LicenseRef-New-License"},
    )

    assert result.returncode == 1
    assert "unrecognized license label" in result.stderr


@pytest.mark.unit
def test_missing_license_metadata_fails(tmp_path: Path) -> None:
    result = run_license_check(tmp_path=tmp_path, record={"Name": "missing-license"})

    assert result.returncode == 1
    assert "license metadata is missing" in result.stderr
