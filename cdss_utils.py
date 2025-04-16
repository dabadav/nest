import os
import base64
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import scipy.stats as stats

from ai_cdss.constants import *

def expand_session(session: pd.DataFrame):
    """
    For each prescription, generate expected sessions and merge with actual performed sessions.
    Missing sessions are marked as NOT_PERFORMED with appropriate filling.
    """
    # Ensure dates are datetime
    session["SESSION_DATE"] = pd.to_datetime(session["SESSION_DATE"])
    last_session_per_patient = session.groupby("PATIENT_ID")["SESSION_DATE"].max().to_dict()
    
    # Step 1: Generate expected sessions DataFrame
    prescriptions = session.drop_duplicates(subset=["PRESCRIPTION_ID", "PATIENT_ID", "PROTOCOL_ID", "PRESCRIPTION_STARTING_DATE", "PRESCRIPTION_ENDING_DATE", "WEEKDAY_INDEX"])
    expected_rows = []

    for _, row in prescriptions.iterrows():
        start = row["PRESCRIPTION_STARTING_DATE"]
        end = row["PRESCRIPTION_ENDING_DATE"]
        weekday = row["WEEKDAY_INDEX"]

        if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
            continue

        # Cap end date at last actual session of the patient
        last_patient_session = last_session_per_patient.get(row["PATIENT_ID"], pd.Timestamp.today())
        if end > last_patient_session:
            end = last_patient_session

        expected_dates = generate_expected_sessions(start, end, int(weekday))
    
        for date in expected_dates:
            template_row = row.copy()
            template_row["SESSION_DATE"] = date
            template_row["STATUS"] = "NOT_PERFORMED"
            template_row["ADHERENCE"] = 0.0
            template_row["SESSION_DURATION"] = 0
            template_row["REAL_SESSION_DURATION"] = 0
            # Set session outcome-related columns to NaN
            for col in METRIC_COLUMNS:
                template_row[col] = np.nan

            # Append to expected rows
            expected_rows.append(template_row.to_dict())

    expected_df = pd.DataFrame(expected_rows)

    if expected_df.empty:
        return session  # No new data, return original

    # Filter expected_df to remove performed sessions
    performed_index = pd.MultiIndex.from_frame(session[["PRESCRIPTION_ID", "SESSION_DATE"]])
    expected_index = pd.MultiIndex.from_frame(expected_df[["PRESCRIPTION_ID", "SESSION_DATE"]])  
    mask = ~expected_index.isin(performed_index)
    expected_df = expected_df.loc[mask]

    # Merge expected sessions with performed sessions
    merged = pd.concat([session, expected_df], ignore_index=True)
    
    # Propagate last performed session values (forward fill by prescription)
    merged.sort_values(by=["PRESCRIPTION_ID", "SESSION_DATE"], inplace=True)
    fill_columns = METRIC_COLUMNS
    merged[fill_columns] = merged.groupby("PRESCRIPTION_ID")[fill_columns].ffill()
    merged.fillna({'DM_VALUE': 0, 'PE_VALUE': 0}, inplace=True)

    # For missing session-specific outcome columns, set to NaN
    is_missing = merged["STATUS"] == "NOT_PERFORMED"
    for col in SESSION_COLUMNS:
        merged.loc[is_missing, col] = np.nan

    return merged.reset_index(drop=True)

def generate_expected_sessions(start_date, end_date, target_weekday):
    """
    Generate all expected session dates between start_date and end_date for the given target weekday.
    If the prescription end date is in the future, use today as the end limit (assuming future sessions are not yet done).
    """
    expected_dates = []
    today = pd.Timestamp.today()
    
    # If the prescription is still ongoing, cap the end_date at today
    if end_date is None:
        return expected_dates  # no valid end date
    if end_date > today:
        end_date = today
    
    # Find the first occurrence of the target weekday on or after start_date
    if start_date.weekday() != target_weekday:
        days_until_target = (target_weekday - start_date.weekday()) % 7
        start_date = start_date + pd.Timedelta(days=days_until_target)
    
    # Generate dates every 7 days (weekly) from the adjusted start_date up to end_date
    current_date = start_date

    while current_date <= end_date:
        expected_dates.append(current_date)
        current_date += pd.Timedelta(days=7)
    
    return expected_dates

def filter_study_range(group):
    day_0 = group["SESSION_DATE"].min()
    study_range = group["SESSION_DATE"] <= day_0 + pd.Timedelta(days=43)
    return group[study_range]

def check_session(session: pd.DataFrame) -> pd.DataFrame:
    """
    Check for data discrepancies in session DataFrame, export findings to ~/.ai_cdss/logs/,
    log summary, and return cleaned DataFrame.

    Parameters
    ----------
    session : pd.DataFrame
        Session DataFrame to check and clean.

    Returns
    -------
    pd.DataFrame
        Cleaned session DataFrame.
    """
    # Step 0: Setup paths
    log_dir = os.path.expanduser("~/.ai_cdss/logs/")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = os.path.join(log_dir, "data_check.log")

    # Setup logging (re-setup per function call to ensure correct log file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Step 1: Patient registered but no data yet (no prescription)
    print("Patient registered but no data yet.")
    patients_no_data = session[session["PRESCRIPTION_ID"].isna()]
    if not patients_no_data.empty:
        export_file = os.path.join(log_dir, f"patients_no_data_{timestamp}.csv")
        patients_no_data[["PATIENT_ID", "PRESCRIPTION_ID", "SESSION_ID"]].to_csv(export_file, index=False)
        logger.warning(f"{len(patients_no_data)} patients found without prescription. Check exported file: {export_file}")
    else:
        logger.info("No patients without prescription found.")

    # Drop these rows
    session = session.drop(patients_no_data.index)

    # Sessions in session table but not in recording table (no adherence)
    print("Sessions in session table but not in recording table")
    patient_session_discrepancy = session[session["ADHERENCE"].isna()]
    if not patient_session_discrepancy.empty:
        export_file = os.path.join(log_dir, f"patient_session_discrepancy_{timestamp}.csv")
        patient_session_discrepancy[["PATIENT_ID", "PRESCRIPTION_ID", "SESSION_ID"]].to_csv(export_file, index=False)
        logger.warning(f"{len(patient_session_discrepancy)} sessions found without adherence. Check exported file: {export_file}")
    else:
        logger.info("No sessions without adherence found.")

    # Drop these rows
    session = session.drop(patient_session_discrepancy.index)

    # Final info
    logger.info(f"Session data cleaned. Final shape: {session.shape}")

    return session

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on,
    how: str = "left",
    export_dir: str = "~/.ai_cdss/logs/",
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """
    Perform a safe merge and independently report unmatched rows from left and right DataFrames.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame.
    right : pd.DataFrame
        Right DataFrame.
    on : str or list
        Column(s) to join on.
    how : str, optional
        Type of merge to be performed. Default is "left".
    export_dir : str, optional
        Directory to export discrepancy reports and logs.
    left_name : str, optional
        Friendly name for the left DataFrame, for logging.
    right_name : str, optional
        Friendly name for the right DataFrame, for logging.
    drop_inconsistencies : bool, optional
        If True, drop inconsistent rows (left-only). Default is False.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    # Prepare export directory
    export_dir = os.path.expanduser(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(export_dir, "data_check.log")

    # Setup logger — clean, no extra clutter
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Step 1: Outer merge for discrepancy check
    discrepancy_check = left.merge(right, on=on, how="outer", indicator=True)

    left_only = discrepancy_check[discrepancy_check["_merge"] == "left_only"]
    right_only = discrepancy_check[discrepancy_check["_merge"] == "right_only"]

    # Step 2: Export and log discrepancies if found

    if not left_only.empty:
        export_file = os.path.join(export_dir, f"{left_name}_only_{timestamp}.csv")
        try:
            left_only[BY_ID + ["SESSION_DURATION", "SCORE", "DM_VALUE", "PE_VALUE"]].to_csv(export_file, index=False)
        except KeyError as e:
            left_only.to_csv(export_file, index=False)
            
        logger.warning(
            f"{len(left_only)} rows found only in '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    if not right_only.empty:
        export_file = os.path.join(export_dir, f"{right_name}_only_{timestamp}.csv")
        try:
            right_only[BY_PPS + ["SESSION_DURATION", "SCORE", "DM_VALUE", "PE_VALUE"]].to_csv(export_file, index=False)
        except KeyError as e:
            right_only.to_csv(export_file, index=False)
        logger.warning(
            f"{len(right_only)} rows from '{right_name}' DataFrame did not match '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    # Step 3: Actual requested merge
    merged = left.merge(right, on=on, how=how)

    return merged

def get_confidence_intervals(popt, pcov, confidence=0.95):
    alpha = 1.0 - confidence
    dof = max(0, len(popt) - 1)
    tval = stats.t.ppf(1.0 - alpha / 2., dof)
    perr = np.sqrt(np.diag(pcov))
    intervals = [(param - tval * err, param + tval * err) for param, err in zip(popt, perr)]
    return intervals

def generate_patient_protocol_report(study_data, figures_dir, output_html):
    html = ['<html><head><title>DM Model Report</title></head><body>']
    html.append('<h1>Report: Protocols x Patients</h1>')
    html.append('<table border="1" style="border-collapse: collapse;">')

    # Sort unique patient and protocol IDs
    patients = sorted(study_data[PATIENT_ID].unique())
    protocols = sorted(study_data[PROTOCOL_ID].unique())

    # Table header
    html.append('<tr><th>Protocol \\ Patient</th>')
    for patient in patients:
        html.append(f'<th>{patient}</th>')
    html.append('</tr>')

    # Table rows
    for protocol in protocols:
        html.append(f'<tr><td><strong>{protocol}</strong></td>')
        for patient in patients:
            # Prepare file path
            filename = f'{patient}_{protocol}.png'
            filepath = os.path.join(figures_dir, filename)

            # Check and embed image
            if os.path.exists(filepath):
                with open(filepath, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
                    img_tag = f'<img src="data:image/png;base64,{encoded}" width="200">'
                    html.append(f'<td>{img_tag}</td>')
            else:
                html.append('<td> - </td>')  # Empty cell if no image
        html.append('</tr>')

    html.append('</table></body></html>')

    # Save HTML
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, 'w') as f:
        f.write('\n'.join(html))

    print(f"✅ Report generated at: {output_html}")

