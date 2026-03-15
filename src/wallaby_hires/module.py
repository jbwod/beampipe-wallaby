import logging
import re
from typing import Any, Optional

from astropy.table import Table

from app.core.archive.adapters import get_adapter, list_adapter_names
from app.core.projects.contracts import DiscoverBundle, get_discover_enrichment
from app.core.archive.adapters.casda import _extract_scan_id
from app.core.utils.astro import degrees_to_dms, degrees_to_hms, to_python_value

logger = logging.getLogger(__name__)

PROJECT_NAME = "wallaby_hires"
REQUIRED_ADAPTERS = ["casda", "vizier"]
DISCOVERY_ENRICHMENT_KEYS = ["ra_dec_vsys", "sbid_to_eval_file"]

GRAPH_PATH = None

GRAPH_GITHUB_URL = None

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


def _query_with_adapter(
    adapter_name: str,
    query: str,
    adapters: dict[str, Any] | None = None,
) -> Table:
    if adapters and adapter_name in adapters:
        return adapters[adapter_name].query(query)
    adapter = get_adapter(adapter_name)
    if adapter is None:
        raise ValueError(
            f"Adapter '{adapter_name}' not found. Available: {list_adapter_names()}"
        )
    return adapter.query(query)


def discover(source_identifier: str, adapters: dict[str, Any] | None = None) -> DiscoverBundle:
    query = VISIBILITY_QUERY_TEMPLATE.format(source_identifier=source_identifier)
    logger.info(f"Q: {source_identifier}")
    try:
        query_results = _query_with_adapter(
            adapter_name="casda",
            query=query,
            adapters=adapters,
        )
        logger.info(f"R: {len(query_results)} results for {source_identifier}")

        if len(query_results) == 0:
            return {
                "query_results": query_results,
                "enrichments": {
                    "ra_dec_vsys": None,
                    "sbid_to_eval_file": {},
                },
            }

        # query source-level enrichment once
        ra_dec_vsys_data = query_ra_dec_vsys(source_identifier, adapters=adapters)

        # query evaluation files once per unique sbid
        obs_ids = query_results["obs_id"] if "obs_id" in query_results.colnames else []
        sbid_list: list[int | None] = []
        for obs_id in obs_ids:
            if isinstance(obs_id, str) and "ASKAP-" in obs_id:
                sbid = obs_id.replace("ASKAP-", "")
                sbid_list.append(int(sbid) if sbid.isdigit() else None)
            else:
                sbid_list.append(None)

        sbid_to_eval_file: dict[int, dict[str, str]] = {}
        unique_sbids = {sbid for sbid in sbid_list if sbid is not None}
        logger.info(f"Querying evaluation files for {len(unique_sbids)} unique SBIDs...")
        for sbid in unique_sbids:
            eval_info = get_evaluation_file_for_sbid(sbid, adapters=adapters)
            if eval_info:
                sbid_to_eval_file[sbid] = eval_info
                logger.info(f"SBID {sbid}: evaluation file = {eval_info['filename']}")
            else:
                logger.warning(f"No evaluation file found for SBID {sbid}")

        return {
            "query_results": query_results,
            "enrichments": {
                "ra_dec_vsys": ra_dec_vsys_data,
                "sbid_to_eval_file": sbid_to_eval_file,
            },
        }
    except Exception as e:
        logger.error(f"E: {source_identifier}: {e}")
        raise


def manifest(
    metadata_by_source: dict[str, list[dict[str, Any]]],
    *,
    staged_urls_by_scan_id: dict[str, str],
    eval_urls_by_sbid: dict[str, str],
    checksum_urls_by_scan_id: dict[str, str] | None = None,
    eval_checksum_urls_by_sbid: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build manifest sources with sbids[].datasets structure.
    Each sbid has evaluation_file, evaluation_file_url, and datasets (visibility files with staged_url, checksum_url).
    """
    checksums = checksum_urls_by_scan_id or {}
    eval_checksums = eval_checksum_urls_by_sbid or {}
    sources: list[dict[str, Any]] = []

    for source_identifier, records in metadata_by_source.items():
        sbid_groups: dict[str, dict[str, Any]] = {}
        ra_string = ""
        dec_string = ""
        vsys = None

        for rec in records:
            meta = rec.get("metadata_json") or {}
            datasets = meta.get("datasets") or []
            for ds in datasets:
                sbid = str(ds.get("sbid") or "")
                scan_id = ds.get("scan_id")
                if scan_id is None and ds.get("obs_publisher_did"):
                    scan_id = _extract_scan_id(str(ds["obs_publisher_did"]))
                scan_id = str(scan_id or sbid)
                name = ds.get("name") or ds.get("dataset_id") or ds.get("visibility_filename") or ""
                staged_url = (
                    staged_urls_by_scan_id.get(scan_id)
                    or ds.get("staged_url")
                    or staged_urls_by_scan_id.get(sbid)
                )
                checksum_url = checksums.get(scan_id) or checksums.get(sbid) or ""
                evaluation_file = ds.get("evaluation_file") or ""
                evaluation_file_url = (
                    eval_urls_by_sbid.get(sbid)
                    or ds.get("evaluation_file_url")
                    or ds.get("evaluation_file_access_url")
                    or ""
                )
                evaluation_file_checksum_url = eval_checksums.get(sbid) or ""
                if not ra_string and ds.get("ra_string"):
                    ra_string = str(ds["ra_string"])
                if not dec_string and ds.get("dec_string"):
                    dec_string = str(ds["dec_string"])
                if vsys is None and ds.get("vsys") is not None:
                    vsys = ds["vsys"]

                if sbid not in sbid_groups:
                    sbid_groups[sbid] = {
                        "sbid": sbid,
                        "evaluation_file": evaluation_file,
                        "evaluation_file_url": evaluation_file_url,
                        "evaluation_file_checksum_url": evaluation_file_checksum_url,
                        "datasets": [],
                    }
                sbid_groups[sbid]["datasets"].append({
                    "name": name,
                    "staged_url": staged_url or "",
                    "checksum_url": checksum_url,
                })
            flags = meta.get("discovery_flags") or {}
            if not ra_string and flags.get("ra_string"):
                ra_string = str(flags["ra_string"])
            if not dec_string and flags.get("dec_string"):
                dec_string = str(flags["dec_string"])
            if vsys is None and flags.get("vsys") is not None:
                vsys = flags["vsys"]

        if sbid_groups:
            sources.append({
                "source_identifier": source_identifier,
                "ra_string": ra_string,
                "dec_string": dec_string,
                "vsys": vsys,
                "sbids": list(sbid_groups.values()),
            })

    return sources

# expected format:
# [
#     {
#         "source_identifier": ...,
#         "ra_string": ...,
#         "dec_string": ...,
#         "vsys": ...,
#         "datasets": [
#             {
#                 "name": ...,
#                 "staged_url": ...,
#                 "evaluation_file": ...,
#                 "evaluation_file_url": ...,
#             },
#             ...
#         ]
#     },
#     ...
# ]


def query_ra_dec_vsys(
    source_identifier: str,
    adapters: dict[str, Any] | None = None,
) -> Optional[dict[str, Any]]:
    """Query Vizier TAP for RA, DEC, and VSys from HIPASS catalog.
    
    (tap_query_RA_DEC_VSYS)
    (usage in process_SOURCE_str)
    This is probably done in workflow rather than here, but just testing the TAP query.
    """
    # Extract the part after "HIPASS" if present (case-insensitive).
    #
    # We want Vizier's HIPASS table keys like "J1318-21" rather than full
    # source identifiers like "HIPASSJ1318-21".
    s = source_identifier.strip()
    m = re.search(r"hipass", s, flags=re.IGNORECASE)
    if m:
        extracted_name = s[m.end() :].strip()
    else:
        extracted_name = s

    query = RA_DEC_VSYS_QUERY_TEMPLATE.format(source_name=extracted_name)
    logger.info(f"Executing RA/DEC/VSys query: {query}")

    try:
        results = _query_with_adapter(
            adapter_name="vizier",
            query=query,
            adapters=adapters,
        )
        logger.info(f"RA/DEC/VSys query returned {len(results)} results")

        if len(results) == 0:
            logger.warning(f"No RA/DEC/VSys data found for {source_identifier}")
            return None

        # Extract values and convert NumPy types to native Python types
        ra_deg = to_python_value(results["RAJ2000"][0])
        dec_deg = to_python_value(results["DEJ2000"][0])
        vsys = to_python_value(results["VSys"][0])

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
            "ra_string": ra_string, # what will be used in the workflow
            "dec_string": dec_string, # what will be used in the workflow
            "ra_hms": (ra_h, ra_m, ra_s), # what will be used in the workflow
            "dec_dms": (dec_d, dec_m, dec_s),
        }
    except Exception as e:
        logger.error(f"Error executing RA/DEC/VSys query: {e}")
        return None


def query_sbid_evaluation(sbid: int, adapters: dict[str, Any] | None = None) -> Table:
    """Query CASDA for evaluation files by SBID."""
    query = SBID_EVALUATION_QUERY_TEMPLATE.format(sbid=str(sbid))
    return _query_with_adapter(
        adapter_name="casda",
        query=query,
        adapters=adapters,
    )


def get_evaluation_file_for_sbid(
    sbid: int, adapters: dict[str, Any] | None = None
) -> Optional[dict[str, str]]:
    """Get the evaluation file for a given SBID.

    Prefers format='calibration' (calibration-metadata-processing-logs) for workflow
    linmos.primarybeam.ASKAP_PB.image. Falls back to largest by filesize if no calibration.

    Returns dict with 'filename' and 'access_url' (from observation_evaluation_file table).
    access_url is required for CASDA staging (ID=evaluation-N, not filename).
    """
    try:
        eval_results = query_sbid_evaluation(sbid, adapters=adapters)
        if len(eval_results) == 0:
            return None
        if "filename" not in eval_results.colnames:
            return None

        # Simply return the largest calibration file (by filesize)
        if "format" in eval_results.colnames:
            calib_mask = [
                str(r).lower() == "calibration" for r in eval_results["format"]
            ]
            if any(calib_mask):
                calib_rows = eval_results[calib_mask]
                if len(calib_rows) == 0:
                    return None
                if "filesize" in calib_rows.colnames:
                    row = calib_rows[calib_rows["filesize"].argmax()]
                else:
                    row = calib_rows[0]
                filename = str(row["filename"])
                access_url = (
                    str(row["access_url"])
                    if "access_url" in calib_rows.colnames and row["access_url"]
                    else None
                )
                return {"filename": filename, "access_url": access_url}
        return None
    except Exception as e:
        logger.warning(f"Error querying evaluation file for SBID {sbid}: {e}")
        return None


def prepare_metadata(
    source_identifier: str,
    query_results: DiscoverBundle,
    data_url_by_scan_id: dict[str, str] | None = None,
    checksum_url_by_scan_id: dict[str, str] | None = None,
    include_evaluation_files: bool = True,
    include_ra_dec_vsys: bool = True,
    adapters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Prepare metadata from discover bundle.

    include_evaluation_files/include_ra_dec_vsys are wallaby-specific shaping
    toggles; core does not set them and uses defaults.
    """
    metadata_list = []

    query_results_table = query_results["query_results"]

    ra_dec_vsys_data = get_discover_enrichment(
        query_results,
        "ra_dec_vsys",
        expected_type=dict,
        module_name=PROJECT_NAME,
    )
    if not include_ra_dec_vsys:
        ra_dec_vsys_data = None

    # Extract filenames from query results
    filenames = (
        query_results_table["filename"] if "filename" in query_results_table.colnames else []
    )
    if len(filenames) == 0:
        logger.warning("No filename column values found in query results for %s", source_identifier)
    # Extract other useful metadata if available
    obs_ids = query_results_table["obs_id"] if "obs_id" in query_results_table.colnames else []
    obs_publisher_dids = (
        query_results_table["obs_publisher_did"]
        if "obs_publisher_did" in query_results_table.colnames
        else []
    )
    scan_ids = [
        _extract_scan_id(str(obs_publisher_did)) if obs_publisher_did is not None else None
        for obs_publisher_did in obs_publisher_dids
    ]
    sbid_list = []
    for obs_id in obs_ids:
        # Extract SBID from obs_id (format: "ASKAP-12345")
        if isinstance(obs_id, str) and "ASKAP-" in obs_id:
            sbid = obs_id.replace("ASKAP-", "")
            sbid_list.append(int(sbid) if sbid.isdigit() else None)
        else:
            sbid_list.append(None)

    # Get evaluation files for each unique SBID (if requested)
    sbid_to_eval_file = (
        get_discover_enrichment(
            query_results,
            "sbid_to_eval_file",
            default={},
            expected_type=dict,
            module_name=PROJECT_NAME,
        )
        if include_evaluation_files
        else {}
    )

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
        #     staged_filename = extract_filename_from_url(staged_url)
        #     if staged_filename and staged_filename != filename:
        #         logger.warning(
        #             "Staged URL filename mismatch for scan id %s: expected %s, got %s",
        #             scan_id,
        #             filename,
        #             staged_filename,
        #         )
        # if checksum_url:
        #     checksum_filename = extract_filename_from_url(checksum_url)
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
        eval_info = sbid_to_eval_file.get(sbid) if sbid and include_evaluation_files else None
        evaluation_file = (
            eval_info["filename"] if isinstance(eval_info, dict) else eval_info
        ) if eval_info else None
        evaluation_file_access_url = (
            eval_info.get("access_url")
        )

        dataset_metadata = {
            "source_identifier": source_identifier,
            "dataset_id": filename,
            "visibility_filename": filename,
            "sbid": sbid,
            "obs_id": str(obs_ids[i]) if i < len(obs_ids) else None,
            "staged_url": staged_url,
            "checksum_url": checksum_url,
            "evaluation_file": evaluation_file,
            "evaluation_file_access_url": evaluation_file_access_url,
        }

        # Add RA/DEC/VSys data if available (same for all datasets from same source)
        if isinstance(ra_dec_vsys_data, dict):
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
        for colname in query_results_table.colnames:
            if colname not in ["filename", "obs_id"]:
                try:
                    value = to_python_value(query_results_table[colname][i])
                    dataset_metadata[colname.lower()] = value
                except (IndexError, KeyError):
                    pass

        metadata_list.append(dataset_metadata)

    discovery_flags = (
        {"ra_dec_vsys_complete": ra_dec_vsys_data is not None} if include_ra_dec_vsys else {}
    )
    return (metadata_list, discovery_flags)


"""
misc. og-funcs from wallaby-hires
- Visibility query: 92, 797-819 (tap_query_filename_visibility)
- SBID evaluation query: 831-853 (tap_query_sbid_evaluation)
- RA/DEC/VSys query: 599-633 (tap_query_RA_DEC_VSYS)
- Staging: 1124, 1642, 1953 (casda.stage_data)
- Coordinate conversion: 541-585 (degrees_to_hms, degrees_to_dms)
"""