"""
Data Fetcher - Downloads market data from Yahoo Finance
"""
import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class DataFetcher:
    """Fetches and caches market data from Yahoo Finance"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch(self, 
              symbols: Optional[list] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for given symbols.
        
        Returns:
            Dict mapping symbol to DataFrame with columns:
            [Open, High, Low, Close, Volume, Adj Close]
        """
        symbols = symbols or config.data.symbols
        start_date = start_date or config.data.start_date
        end_date = end_date or config.data.end_date
        
        data = {}
        for symbol in symbols:
            cache_path = self.cache_dir / f"{symbol.replace('^', '')}_{start_date}_{end_date}.csv"
            
            if use_cache and cache_path.exists():
                df = pd.read_csv(cache_path, index_col=0, parse_dates=False)
            else:
                print(f"Downloading {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
            # If the CSV had multi-level headers, the first data row might be strings
            # and the index might not be proper datetime.
            # Flatten multi-index if present (from yf.download)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(-1) # drop Ticker level if exists
                
            # Ensure index is datetime (coercing bad rows to NaT) and numeric data
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            # Save cleaned cache
            df.to_csv(cache_path)
            
            data[symbol] = df
            
        return data
    
    def get_aligned_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch all symbols and align them by date.
        
        Returns:
            DataFrame with columns like: SPY_Close, SPY_Volume, VIX_Close, etc.
        """
        raw_data = self.fetch(use_cache=use_cache)
        
        # Combine into single DataFrame
        dfs = []
        for symbol, df in raw_data.items():
            # Clean symbol name (remove ^)
            clean_name = symbol.replace("^", "")
            # Rename columns with symbol prefix
            df_renamed = df.add_prefix(f"{clean_name}_")
            dfs.append(df_renamed)
        
        # Align by date (inner join)
        aligned = pd.concat(dfs, axis=1).dropna()
        
        return aligned
    
    def get_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/val/test based on config dates.
        
        Returns:
            Dict with keys 'train', 'val', 'test'
        """
        train = df[df.index <= config.data.train_end]
        val = df[(df.index > config.data.train_end) & (df.index <= config.data.val_end)]
        test = df[df.index > config.data.val_end]
        
        return {
            'train': train,
            'val': val,
            'test': test
        }


if __name__ == "__main__":
    # Test the fetcher
    fetcher = DataFetcher()
    data = fetcher.get_aligned_data()
    print(f"Fetched data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Columns: {list(data.columns)}")
    
    splits = fetcher.get_splits(data)
    for name, df in splits.items():
        print(f"{name}: {len(df)} days ({df.index[0]} to {df.index[-1]})")
