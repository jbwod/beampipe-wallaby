from typing import Any

from astropy.table import Table

from app.core.archive.adapters.casda import CASDA_TAP_URL, query as casda_query

PROJECT_NAME = "wallaby_hires"

# Query template for visibility files
VISIBILITY_QUERY_TEMPLATE = "SELECT * FROM ivoa.obscore WHERE filename LIKE '{source_identifier}%'"

def ping() -> None:
    print("wallaby_hires pong")
    
def discover(source_identifier: str) -> Table:
    query = VISIBILITY_QUERY_TEMPLATE.format(source_identifier=source_identifier)
    results = casda_query(query, tap_url=CASDA_TAP_URL)
    return results


def stage(casda_client: Any, query_results: Table) -> tuple[dict[str, str], dict[str, str]]:
    return {}, {}


def prepare_metadata(
    source_identifier: str,
    query_results: Table,
    data_url_by_scan_id: dict[str, str] | None = None,
    checksum_url_by_scan_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return []
