from typing import Any

from astropy.table import Table

PROJECT_NAME = "wallaby_hires"


def discover(source_identifier: str) -> Table:
    return Table()


def stage(casda_client: Any, query_results: Table) -> tuple[dict[str, str], dict[str, str]]:
    return {}, {}


def prepare_metadata(
    source_identifier: str,
    query_results: Table,
    data_url_by_scan_id: dict[str, str] | None = None,
    checksum_url_by_scan_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return []
