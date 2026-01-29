import logging
import re
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from astropy.table import Table

from app.core.archive.adapters.casda import CASDA_TAP_URL, _extract_scan_id, query as casda_query
from app.core.archive.adapters.vizier import VIZIER_TAP_URL, query as vizier_query
from app.core.utils.astro import degrees_to_dms, degrees_to_hms

logger = logging.getLogger(__name__)

PROJECT_NAME = "wallaby_hires"

# Query template for visibility files
VISIBILITY_QUERY_TEMPLATE = "SELECT * FROM ivoa.obscore WHERE filename LIKE '{source_identifier}%'"

# Query template for finding eval files by SBID
SBID_EVALUATION_QUERY_TEMPLATE = "SELECT * FROM casda.observation_evaluation_file WHERE sbid = '{sbid}'"

# Query template for RA, DEC, VSys from Vizier HIPASS catalog
RA_DEC_VSYS_QUERY_TEMPLATE = (
    'SELECT RAJ2000, DEJ2000, VSys FROM "J/AJ/128/16/table2" WHERE HIPASS LIKE \'{source_name}\''
)


def ping() -> None:
    print("wallaby_hires pong")

def discover(source_identifier: str) -> Table:
    query = VISIBILITY_QUERY_TEMPLATE.format(source_identifier=source_identifier)
    logger.info(f"Querying CASDA for source: {source_identifier}")
    try:
        results = casda_query(query, tap_url=CASDA_TAP_URL)
        logger.info(f"Found {len(results)} results for {source_identifier}")
        return results
    except Exception as e:
        logger.error(f"Error querying CASDA for {source_identifier}: {e}")
        raise


def stage(casda_client: Any, query_results: Table) -> tuple[dict[str, str], dict[str, str]]:
    return {}, {}


def prepare_metadata(
    source_identifier: str,
    query_results: Table,
    data_url_by_scan_id: dict[str, str] | None = None,
    checksum_url_by_scan_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare metadata for datasets."""
    return []
