import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import json

# Parameters are now read from environment variables provided by docker-compose
CONFIG = {
    "INPUT_FILE": os.getenv("INPUT_FILE", "2026.09.02.txt"),
    "OUTPUT_GRAPH": os.getenv("OUTPUT_GRAPH", "liquidity_analysis.png"),
    "DIAGNOSTIC_FILE": os.getenv("DIAGNOSTIC_FILE", "distributions_diagnostic.txt"),
    "OUTPUT_EXCEL": os.getenv("OUTPUT_EXCEL", "aggregated_trades.xlsx"),
    "VOL_QUANTILE": int(os.getenv("VOL_QUANTILE", 10)),
    "IMPACT_QUANTILE": int(os.getenv("IMPACT_QUANTILE", 10)),
    "START_TIME": os.getenv("START_TIME", "07:00:00")
}

def load_and_preprocess(file_path, start_time_str):
    # Load the data. The file is tab-separated based on inspection.
    df = pd.read_csv(file_path, sep='\t')
    
    # Convert TRADETIME and TRADETIME_MSEC to a single datetime object for easier filtering/plotting
    def parse_time(row):
        try:
            t = datetime.strptime(str(row['TRADETIME']), "%H:%M:%S")
            return t + timedelta(microseconds=int(row['TRADETIME_MSEC']))
        except:
            return None

    df['timestamp'] = df.apply(parse_time, axis=1)
    df = df.dropna(subset=['timestamp'])
    
    # Filter by START_TIME
    start_time = datetime.strptime(start_time_str, "%H:%M:%S")
    df = df[df['timestamp'].dt.time >= start_time.time()]
    
    return df

def aggregate_trades(df):
    if df.empty:
        return pd.DataFrame()

    aggregated = []
    current_group = None
    
    for _, row in df.iterrows():
        sign = row['BUYSELL']
        price = row['PRICE']
        qty = row['QTY']
        timestamp = row['timestamp']
        
        if current_group is None or current_group['sign'] != sign:
            current_group = {
                'sign': sign,
                'first_time': timestamp,
                'first_price': price,
                'last_price': price,
                'total_qty': qty,
                'count': 1
            }
            aggregated.append(current_group)
        else:
            current_group['last_price'] = price
            current_group['total_qty'] += qty
            current_group['count'] += 1
    
    agg_df = pd.DataFrame(aggregated)
    if not agg_df.empty:
        # Filter out groups that consist of only one trade
        agg_df = agg_df[agg_df['count'] > 1].copy()
        
        if not agg_df.empty:
            agg_df['impact'] = agg_df['last_price'] - agg_df['first_price']
            agg_df.loc[agg_df['sign'] == 'SELL', 'impact'] *= -1
            agg_df = agg_df.rename(columns={'first_time': 'timestamp'})
    
    return agg_df

def process_side(agg_df, side, vol_q_count, impact_q_count):
    side_df = agg_df[agg_df['sign'] == side].copy()
    if side_df.empty:
        return side_df, None, None
    
    vol_quantiles = np.unique(np.quantile(side_df['total_qty'], np.linspace(0, 1, vol_q_count + 1)))
    
    def get_vol_q_idx(v):
        return np.searchsorted(vol_quantiles, v) - 1

    side_df['vol_q_idx'] = side_df['total_qty'].apply(lambda x: max(0, min(len(vol_quantiles)-2, get_vol_q_idx(x))))
    
    impact_distributions = {} 
    for q_idx in range(len(vol_quantiles) - 1):
        subset = side_df[side_df['vol_q_idx'] == q_idx]
        if not subset.empty:
            impacts = subset['impact'].values
            iq = np.unique(np.quantile(impacts, np.linspace(0, 1, impact_q_count + 1)))
            impact_distributions[q_idx] = iq.tolist() # Convert to list for serialization
        else:
            impact_distributions[q_idx] = []

    return side_df, vol_quantiles.tolist(), impact_distributions

def main():
    print(f"Starting processing with config: {CONFIG}")
    try:
        df = load_and_preprocess(CONFIG["INPUT_FILE"], CONFIG["START_TIME"])
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if df.empty:
        print("No data found after filtering.")
        return

    agg_df = aggregate_trades(df)
    if agg_df.empty:
        print("No aggregated trades.")
        return

    buy_df, buy_vol_q, buy_impact_dist = process_side(agg_df, 'BUY', CONFIG["VOL_QUANTILE"], CONFIG["IMPACT_QUANTILE"])
    sell_df, sell_vol_q, sell_impact_dist = process_side(agg_df, 'SELL', CONFIG["VOL_QUANTILE"], CONFIG["IMPACT_QUANTILE"])

    # --- Diagnostic Data Saving ---
    diagnostic_data = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "buy_side": {
            "vol_quantiles": buy_vol_q,
            "impact_distributions": buy_impact_dist
        },
        "sell_side": {
            "vol_quantiles": sell_vol_q,
            "impact_distributions": sell_impact_dist
        }
    }
    try:
        with open(CONFIG["DIAGNOSTIC_FILE"], 'w') as f:
            f.write("=== LIQUIDITY ANALYSIS DIAGNOSTICS ===\n")
            f.write(json.dumps(diagnostic_data, indent=4))
        print(f"Diagnostic data saved to {CONFIG['DIAGNOSTIC_FILE']}")
    except Exception as e:
        print(f"Failed to save diagnostic file: {e}")
    # ------------------------------

    # --- Exporting Aggregated Trades to Excel ---
    try:
        export_df = pd.concat([buy_df, sell_df], ignore_index=True)
        if not export_df.empty:
            def get_impact_q_idx(row):
                side = row['sign']
                imp = row['impact']
                v_idx = int(row['vol_q_idx'])
                dist = buy_impact_dist.get(v_idx, []) if side == 'BUY' else sell_impact_dist.get(v_idx, [])
                if len(dist) > 1:
                    idx = np.searchsorted(dist, imp) - 1
                    return max(0, min(len(dist) - 2, idx))
                return -1

            export_df['impact_q_idx'] = export_df.apply(get_impact_q_idx, axis=1)
            cols_to_keep = ['timestamp', 'sign', 'first_price', 'last_price', 'total_qty', 'impact', 'vol_q_idx', 'impact_q_idx']
            actual_cols = [c for c in cols_to_keep if c in export_df.columns]
            export_df[actual_cols].to_excel(CONFIG["OUTPUT_EXCEL"], index=False)
            print(f"Aggregated trades exported to {CONFIG['OUTPUT_EXCEL']}")
        else:
            print("No aggregated data available for Excel export.")
    except Exception as e:
        print(f"Failed to export Excel file: {e}")

    cmap_buy = plt.get_cmap('viridis')
    cmap_sell = plt.get_cmap('inferno')

    def assign_colors(side_df, vol_q, impact_dist, cmap):
        if side_df.empty: return []
        colors = []
        for _, row in side_df.iterrows():
            v_idx = int(row['vol_q_idx'])
            impacts_q = impact_dist.get(v_idx, [])
            if len(impacts_q) > 1:
                idx = np.searchsorted(impacts_q, row['impact'])
                norm_val = idx / (len(impacts_q) - 1)
                colors.append(mcolors.to_hex(cmap(norm_val)))
            else:
                colors.append("gray")
        return colors

    buy_df['color'] = assign_colors(buy_df, buy_vol_q, buy_impact_dist, cmap_buy)
    sell_df['color'] = assign_colors(sell_df, sell_vol_q, sell_impact_dist, cmap_sell)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    def plot_side(ax, side_df, title):
        if side_df.empty:
            ax.set_title(f"{title} (No Data)")
            return
        for _, row in side_df.iterrows():
            ax.vlines(row['timestamp'], row['first_price'], row['last_price'], 
                      color=row['color'], linewidth=2, alpha=0.8)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        start = side_df['timestamp'].min()
        end = side_df['timestamp'].max()
        current_hour = start.replace(minute=0, second=0, microsecond=0)
        while current_hour <= end:
            ax.axvline(x=current_hour, color='gray', linestyle='-', alpha=0.3)
            current_hour += timedelta(hours=1)

    plot_side(ax1, buy_df, "Aggregated BUY Trades (Vertical Price Range)")
    plot_side(ax2, sell_df, "Aggregated SELL Trades (Vertical Price Range)")

    plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_GRAPH"])
    print(f"Graph saved to {CONFIG['OUTPUT_GRAPH']}")

if __name__ == "__main__":
    main()
