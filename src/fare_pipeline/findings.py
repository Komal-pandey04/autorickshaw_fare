"""Read/update reports/findings.yaml (the form the marking tool reads)."""
import yaml

from .config import ROOT, load_params


def update_findings(section: str, values: dict) -> None:
    path = ROOT / load_params()["paths"]["findings_out"]
    data = yaml.safe_load(path.read_text())
    data[section] = {**data.get(section, {}), **values}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
