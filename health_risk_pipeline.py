import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_data(file_path):
    """Load data from an Excel file."""
    return pd.read_excel(file_path)


def validate_columns(df, required_columns, file_name):
    """Validate that all required columns are present in the DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_name}: {missing}")


def calculate_risk(merged_df):
    """Calculate the Health_Risk_Score for each node."""
    merged_df['Health_Risk_Score'] = (
        merged_df['Indicator_Value'] *
        merged_df['Exposure_Duration_hour'] *
        merged_df['Population'] *
        merged_df['Vulnerability_Index']
    )
    # Apply multiplier for critical facilities
    merged_df.loc[merged_df['Critical_Facility'] == 1, 'Health_Risk_Score'] *= 1.5
    return merged_df


def assign_priority(df):
    """Sort by risk score, assign priority rank and risk level."""
    df = df.sort_values('Health_Risk_Score', ascending=False).reset_index(drop=True)
    df['Priority_Rank'] = range(1, len(df) + 1)
    n = len(df)
    high_cutoff = int(0.3 * n)
    medium_cutoff = int(0.7 * n)
    df['Risk_Level'] = 'Low'
    df.loc[:high_cutoff - 1, 'Risk_Level'] = 'High'
    df.loc[high_cutoff:medium_cutoff - 1, 'Risk_Level'] = 'Medium'
    return df


def generate_ai_reasoning(df):
    """
    Generate human-readable AI reasoning for each node's risk level.
    This is explainable AI: rule-based analysis of contributing factors.
    """
    reasons = []
    for _, row in df.iterrows():
        factors = []
        if row['Population'] > df['Population'].median():
            factors.append("high population exposure")
        if row['Exposure_Duration_hour'] > df['Exposure_Duration_hour'].median():
            factors.append("long contamination duration")
        if row['Vulnerability_Index'] > df['Vulnerability_Index'].median():
            factors.append("high vulnerability")
        if row['Indicator_Value'] > df['Indicator_Value'].median():
            factors.append("elevated contamination indicator")
        if row['Critical_Facility'] == 1:
            factors.append("presence of a critical facility")
        
        if not factors:
            reason = f"Node {int(row['Node_ID'])} has low risk due to balanced factor levels."
        else:
            reason = f"Node {int(row['Node_ID'])} is {row['Risk_Level'].lower()} risk due to {' and '.join(factors)}."
        reasons.append(reason)
    df['AI_Reasoning_Text'] = reasons
    return df


def generate_risk_reason(df):
    """
    Generate short AI risk reason based on rule-based weighting logic.
    Explainable AI: transparent justification for risk classification.
    """
    reasons = []
    for _, row in df.iterrows():
        parts = []
        if row['Population'] > df['Population'].median():
            parts.append("high population")
        if row['Exposure_Duration_hour'] > df['Exposure_Duration_hour'].median():
            parts.append("long exposure")
        if row['Vulnerability_Index'] > df['Vulnerability_Index'].median():
            parts.append("high vulnerability")
        if row['Indicator_Value'] > df['Indicator_Value'].median():
            parts.append("high indicator")
        if row['Critical_Facility'] == 1:
            parts.append("critical facility")
        
        if not parts:
            reason = "Balanced factors"
        else:
            reason = " x ".join(parts)
        reasons.append(reason)
    df['AI_Risk_Reason'] = reasons
    return df


def generate_executive_summary(df):
    """
    Generate natural language executive summary for decision-makers.
    Explainable AI: counts and insights from rule-based analysis.
    """
    total_zones = len(df)
    high_risk = df[df['Risk_Level'] == 'High']
    num_high = len(high_risk)
    critical_facilities = df['Critical_Facility'].sum()
    
    # Estimate response window based on max exposure
    max_exposure = df['Exposure_Duration_hour'].max()
    response_window = max(1, max_exposure // 2)  # Half of max exposure, min 1 hour
    
    # Get top 3 priority zones
    top_zones = df.head(3)
    
    summary = f"""================ PUBLIC HEALTH AI DECISION SUMMARY ================

Total Zones Analyzed: {total_zones}
High Risk Zones: {num_high}
Zones with Critical Facilities: {int(critical_facilities)}

TOP PRIORITY ZONES:
"""
    for i, (_, row) in enumerate(top_zones.iterrows(), 1):
        summary += f"""{i}. Node {int(row['Node_ID'])} – {row['Risk_Level']} RISK
   Reason: {row['AI_Risk_Reason']}

"""
    
    summary += f"""Recommended First Response Window: {response_window} hours
=================================================================="""
    return summary


def generate_heatmap(df, output_path):
    # Generate mock spatial coordinates
    np.random.seed(42)  # For reproducibility
    df['X'] = np.random.uniform(0, 100, len(df))
    df['Y'] = np.random.uniform(0, 100, len(df))
    
    # Create custom colormap: Green -> Yellow -> Red
    cmap = LinearSegmentedColormap.from_list("risk", ["green", "yellow", "red"])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['X'], df['Y'], c=df['Health_Risk_Score'], cmap=cmap, s=50, alpha=0.7)
    plt.colorbar(scatter, label='Health Risk Level')
    plt.title('Public Health Risk Layer')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(output_path)
    plt.close()


def main():
    """Main function to run the data pipeline."""
    scenario_file = 'scenario_indicators.xlsx'
    vulnerability_file = 'vulnerability_data_filled.xlsx'
    output_excel = 'health_risk_output.xlsx'
    output_png = 'public_health_risk_heatmap.png'
    summary_txt = 'ai_incident_summary.txt'
    
    # Load data
    scenario_df = load_data(scenario_file)
    vulnerability_df = load_data(vulnerability_file)
    
    # Validate columns
    validate_columns(scenario_df, ['Node_ID', 'Indicator_Value', 'Exposure_Duration_hour'], scenario_file)
    validate_columns(vulnerability_df, ['Node_ID', 'Population', 'Vulnerability_Index', 'Critical_Facility'], vulnerability_file)
    
    # Merge datasets on Node_ID
    merged_df = pd.merge(scenario_df, vulnerability_df, on='Node_ID')
    
    # Calculate risk scores
    merged_df = calculate_risk(merged_df)
    
    # Assign priority and risk levels
    final_df = assign_priority(merged_df)
    
    # AI Layer 1: Generate AI reasoning
    final_df = generate_ai_reasoning(final_df)
    
    # AI Layer 2: Generate risk reason
    final_df = generate_risk_reason(final_df)
    
    # Save processed dataset with AI columns
    final_df.to_excel(output_excel, index=False)
    
    # Generate and save heatmap
    generate_heatmap(final_df, output_png)
    
    # AI Layer 3: Generate executive summary
    summary = generate_executive_summary(final_df)
    print(summary)
    with open(summary_txt, 'w') as f:
        f.write(summary)


if __name__ == '__main__':
    main()