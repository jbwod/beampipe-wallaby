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


def _extract_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from a URL (handles query parameters)."""
    decoded_url = unquote(url)
    parsed = urlparse(decoded_url)
    query_params = parse_qs(parsed.query)
    response_disposition = query_params.get("response-content-disposition", [])
    for value in response_disposition:
        match = re.search(r'filename="?([^";]+)"?', value)
        if match:
            return match.group(1)
    filename = parsed.path.split("/")[-1]
    return filename or None


def query_ra_dec_vsys(source_identifier: str) -> Optional[dict[str, Any]]:
    """Query Vizier TAP for RA, DEC, and VSys from HIPASS catalog.
    
    (tap_query_RA_DEC_VSYS)
    (usage in process_SOURCE_str)
    This is probably done in workflow rather than here, but just testing the TAP query.
    """
    # Extract the part after "HIPASS" if present (case-insensitive).
    #
    # We want Vizier's HIPASS table keys like "J1318-21"
    s = source_identifier.strip()
    m = re.search(r"hipass", s, flags=re.IGNORECASE)
    if m:
        extracted_name = s[m.end() :].strip()
    else:
        extracted_name = s

    query = RA_DEC_VSYS_QUERY_TEMPLATE.format(source_name=extracted_name)
    logger.info(f"Executing RA/DEC/VSys query: {query}")

    try:
        results = vizier_query(query, tap_url=VIZIER_TAP_URL)
        logger.info(f"RA/DEC/VSys query returned {len(results)} results")

        if len(results) == 0:
            logger.warning(f"No RA/DEC/VSys data found for {source_identifier}")
            return None

        # NumPy to Python types
        ra_deg_raw = results["RAJ2000"][0]
        dec_deg_raw = results["DEJ2000"][0]
        vsys_raw = results["VSys"][0]

        ra_deg = ra_deg_raw.item() if hasattr(ra_deg_raw, "item") else float(ra_deg_raw)
        dec_deg = dec_deg_raw.item() if hasattr(dec_deg_raw, "item") else float(dec_deg_raw)
        vsys = vsys_raw.item() if hasattr(vsys_raw, "item") else float(vsys_raw)

        logger.info(f"Retrieved RA={ra_deg}, DEC={dec_deg}, VSys={vsys} for {source_identifier}")
        ra_h, ra_m, ra_s = degrees_to_hms(ra_deg)
        dec_d, dec_m, dec_s = degrees_to_dms(dec_deg)
        ra_s = round(ra_s, 2)
        dec_s = round(dec_s, 2)
        ra_string = f"{ra_h}h{ra_m}m{ra_s}s"
        dec_string = f"{dec_d}.{dec_m}.{dec_s}"

        logger.info(
            f"Converted RA={ra_h}h {ra_m}m {ra_s:.2f}s, "
            f"DEC={dec_d}d {dec_m} {dec_s:.2f}s for {source_identifier}"
        )

        return {
            "ra_degrees": ra_deg,
            "dec_degrees": dec_deg,
            "vsys": vsys,
            "ra_string": ra_string,
            "dec_string": dec_string,
            "ra_hms": (ra_h, ra_m, ra_s),
            "dec_dms": (dec_d, dec_m, dec_s),
        }
    except Exception as e:
        logger.error(f"Error executing RA/DEC/VSys query: {e}")
        return None


def query_sbid_evaluation(sbid: int) -> Table:
    """Query CASDA for evaluation files by SBID."""
    query = SBID_EVALUATION_QUERY_TEMPLATE.format(sbid=str(sbid))
    return casda_query(query, tap_url=CASDA_TAP_URL)


def get_evaluation_file_for_sbid(sbid: int) -> Optional[str]:
    """Get the evaluation file for a given SBID (largest file by size)."""
    try:
        eval_results = query_sbid_evaluation(sbid)
        if len(eval_results) == 0:
            return None
        if "filename" in eval_results.colnames and "filesize" in eval_results.colnames:
            largest_file_row = eval_results[eval_results["filesize"].argmax()]
            return str(largest_file_row["filename"])
        elif "filename" in eval_results.colnames:
            return str(eval_results["filename"][0])
        else:
            return None
    except Exception as e:
        logger.warning(f"Error querying evaluation file for SBID {sbid}: {e}")
        return None


def prepare_metadata(
    source_identifier: str,
    query_results: Table,
    data_url_by_scan_id: dict[str, str] | None = None,
    checksum_url_by_scan_id: dict[str, str] | None = None,
    include_evaluation_files: bool = True,
    include_ra_dec_vsys: bool = True,
) -> list[dict[str, Any]]:
    """Prepare metadata for datasets."""
    metadata_list = []

    # Start the RA/DEC/VSys
    ra_dec_vsys_data = None
    if include_ra_dec_vsys:
        logger.info(f"Querying RA/DEC/VSys for source: {source_identifier}")
        ra_dec_vsys_data = query_ra_dec_vsys(source_identifier)
        if ra_dec_vsys_data:
            logger.info(
                f"Source coordinates: RA={ra_dec_vsys_data['ra_string']}, "
                f"DEC={ra_dec_vsys_data['dec_string']}, VSys={ra_dec_vsys_data['vsys']}"
            )

    # # Extract filenames from query results if staged
    # filenames = query_results["filename"] if "filename" in query_results.colnames else []
    filenames = []

    obs_ids = query_results["obs_id"] if "obs_id" in query_results.colnames else []
    obs_publisher_dids = (
        query_results["obs_publisher_did"] if "obs_publisher_did" in query_results.colnames else []
    )
    scan_ids = [
        _extract_scan_id(str(obs_publisher_did)) if obs_publisher_did is not None else None
        for obs_publisher_did in obs_publisher_dids
    ]
    sbid_list = []
    for obs_id in obs_ids:
        # obs_id (format: "ASKAP-12345")
        if isinstance(obs_id, str) and "ASKAP-" in obs_id:
            sbid = obs_id.replace("ASKAP-", "")
            sbid_list.append(int(sbid) if sbid.isdigit() else None)
        else:
            sbid_list.append(None)

    # Get evaluation files for each unique SBID (largest)
    sbid_to_eval_file = {}
    if include_evaluation_files:
        unique_sbids = {sbid for sbid in sbid_list if sbid is not None}
        logger.info(f"Querying evaluation files for {len(unique_sbids)} unique SBIDs...")
        for sbid in unique_sbids:
            eval_file = get_evaluation_file_for_sbid(sbid)
            if eval_file:
                sbid_to_eval_file[sbid] = eval_file
                logger.info(f"SBID {sbid}: evaluation file = {eval_file}")
            else:
                logger.warning(f"No evaluation file found for SBID {sbid}")

    # Scan-id keyed matching from CASDA job results (preferred and deterministic).
    # if data_url_by_scan_id is not None, then we have a scan-id keyed mapping from the CASDA job results.
    if data_url_by_scan_id:
        missing_scan_ids = [sid for sid in scan_ids if sid and sid not in data_url_by_scan_id]
        if missing_scan_ids:
            logger.warning(f"Missing staged URLs for {len(missing_scan_ids)} scan ids")
    if checksum_url_by_scan_id:
        missing_scan_ids = [sid for sid in scan_ids if sid and sid not in checksum_url_by_scan_id]
        if missing_scan_ids:
            logger.warning(f"Missing checksum URLs for {len(missing_scan_ids)} scan ids")

    # Create metadata for each dataset
    for i, filename in enumerate(filenames):
        sbid = sbid_list[i] if i < len(sbid_list) else None
        obs_publisher_did = str(obs_publisher_dids[i]) if i < len(obs_publisher_dids) else None
        scan_id = _extract_scan_id(obs_publisher_did) if obs_publisher_did else None

        # staged_url = data_url_by_scan_id.get(scan_id) if scan_id and data_url_by_scan_id else None
        # checksum_url = (
        #     checksum_url_by_scan_id.get(scan_id) if scan_id and checksum_url_by_scan_id else None
        # )
        # if staged_url:
        #     staged_filename = _extract_filename_from_url(staged_url)
        #     if staged_filename and staged_filename != filename:
        #         logger.warning(
        #             "Staged URL filename mismatch for scan id %s: expected %s, got %s",
        #             scan_id,
        #             filename,
        #             staged_filename,
        #         )
        # if checksum_url:
        #     checksum_filename = _extract_filename_from_url(checksum_url)
        #     if checksum_filename:
        #         base_checksum = checksum_filename.removesuffix(".checksum")
        #         if base_checksum != filename:
        #             logger.warning(
        #                 "Checksum URL filename mismatch for scan id %s: expected %s, got %s",
        #                 scan_id,
        #                 filename,
        #                 checksum_filename,
        #             )
        staged_url = ""
        checksum_url = ""

        dataset_metadata = {
            "source_identifier": source_identifier,
            "dataset_id": filename,
            "visibility_filename": filename,
            "sbid": sbid,
            "obs_id": str(obs_ids[i]) if i < len(obs_ids) else None,
            "staged_url": staged_url,
            "checksum_url": checksum_url,
            "evaluation_file": sbid_to_eval_file.get(sbid) if sbid and include_evaluation_files else None,
        }

        # Add RA/DEC/VSys data if available (same for all datasets from same source)
        if ra_dec_vsys_data:
            dataset_metadata.update(
                {
                    "ra_degrees": ra_dec_vsys_data["ra_degrees"],
                    "dec_degrees": ra_dec_vsys_data["dec_degrees"],
                    "vsys": ra_dec_vsys_data["vsys"],
                    "ra_string": ra_dec_vsys_data["ra_string"],
                    "dec_string": ra_dec_vsys_data["dec_string"],
                }
            )

        # Add any additional columns from query results
       # might be a way to convert the astropy table using pandas or something instead of this
        # Quick fix for now to get the thing to start spitting out some data
        for colname in query_results.colnames:
            if colname not in ["filename", "obs_id"]:
                try:
                    value = query_results[colname][i]
                    # Convert NumPy types to native so we can keep it in a json
                    if hasattr(value, "item"):  # NumPy scalar (int32, float64, etc.)
                        value = value.item()  # Converts to native Python int/float
                    elif isinstance(value, np.ndarray):
                        value = value.tolist()  # Convert arrays to lists
                    elif isinstance(value, (np.integer, np.floating)):
                        value = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, (np.str_, np.bytes_)):
                        value = str(value)
                    # value is now a native Python type (int, float, str, list, etc.)
                    dataset_metadata[colname.lower()] = value
                except (IndexError, KeyError):
                    pass

        metadata_list.append(dataset_metadata)

    return metadata_list


"""
misc. og-funcs from wallaby-hires
- Visibility query: 92, 797-819 (tap_query_filename_visibility)
- SBID evaluation query: 831-853 (tap_query_sbid_evaluation)
- RA/DEC/VSys query: 599-633 (tap_query_RA_DEC_VSYS)
- Staging: 1124, 1642, 1953 (casda.stage_data)
- Coordinate conversion: 541-585 (degrees_to_hms, degrees_to_dms)
"""