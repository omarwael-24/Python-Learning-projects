import os
import csv
import json
import logging
import argparse
import re
from datetime import datetime
from typing import Generator, List, Dict, Any, Optional, Counter
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from collections import defaultdict

# ==========================================
# 1. System Configuration & Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HealthApp_Analyzer")

# ==========================================
# 2. Custom Exceptions (Error Handling)
# ==========================================
class LogSystemError(Exception):
    """Base class for exceptions in this module."""
    pass

class LogFileNotFoundError(LogSystemError):
    """Raised when the target log file cannot be found."""
    pass

class LogParsingError(LogSystemError):
    """Raised when a log line cannot be parsed correctly."""
    pass

# ==========================================
# 3. Data Models (The Schema)
# ==========================================
@dataclass
class LogEntry:
    """
    Represents a single parsed log line with typed data.
    Now includes a real datetime object for time-series analysis.
    """
    raw_date: str
    timestamp: datetime
    log_type: str
    message: str
    source_module: str = "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the object to a dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "log_type": self.log_type,
            "message": self.message,
            "module": self.source_module
        }

@dataclass
class AnalysisReport:
    """Holds the results of the statistical analysis."""
    total_logs: int = 0
    error_count: int = 0
    warning_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    top_errors: List[tuple] = field(default_factory=list)

# ==========================================
# 4. The Core Logic Modules
# ==========================================

class FileNavigator:
    """Responsible for locating files across the operating system."""
    
    @staticmethod
    def find_file(filename: str, root_dir: str = None) -> str:
        """
        Recursively searches for a file starting from root_dir.
        
        Args:
            filename: Name of the file to find.
            root_dir: Starting directory (defaults to User Home).
            
        Returns:
            Absolute path to the file.
        """
        start_path = root_dir or os.path.expanduser('~')
        logger.info(f"Scanning system for '{filename}' starting at: {start_path}...")
        
        # System walk
        for root, dirs, files in os.walk(start_path):
            # Optimization: Skip hidden directories to speed up search
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            if filename in files:
                full_path = os.path.join(root, filename)
                logger.info(f"Target Acquired: {full_path}")
                return full_path
                
        logger.error(f"File '{filename}' not found in {start_path}")
        raise LogFileNotFoundError(f"Could not find {filename}")

class LogParser:
    """Engine responsible for extracting meaning from raw text."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Regex to capture: Date | Type | Message
        # Example: 2024-01-01 12:00:00 | ERROR | Connection Failed
        self.pattern = re.compile(r"^(.*?)\|(.*?)\|(.*)$")

    def parse_stream(self) -> Generator[LogEntry, None, None]:
        """
        Yields LogEntry objects. 
        Uses Regex for more robust parsing than simple split.
        """
        logger.info(f"Opening file stream: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match = self.pattern.match(line)
                if match:
                    date_str, l_type, msg = match.groups()
                    try:
                        # Attempt to parse date string to datetime object
                        # Adjust format based on your actual log file structure
                        dt_obj = self._parse_date(date_str.strip())
                        
                        yield LogEntry(
                            raw_date=date_str.strip(),
                            timestamp=dt_obj,
                            log_type=l_type.strip(),
                            message=msg.strip()
                        )
                    except ValueError:
                        # If date parsing fails, log a warning but continue
                        logger.warning(f"Date parse error at line {line_num}")
                        continue
                else:
                    # Fallback for lines that don't match the strict regex
                    logger.debug(f"Skipping malformed line {line_num}")

    def _parse_date(self, date_str: str) -> datetime:
        """Helper to convert string dates to datetime objects."""
        # Handles multiple formats if necessary. Currently assumes ISO-like.
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            # Fallback simple parser or current time if format is unknown
            # In a real scenario, we would use strict formatting
            return datetime.now()

class AnalyticsEngine:
    """Performs statistical analysis on the log stream."""
    
    def __init__(self):
        self.total = 0
        self.type_counts = Counter()
        self.message_counts = Counter()
        self.min_date = None
        self.max_date = None

    def process_entry(self, entry: LogEntry):
        """Updates stats with a single log entry."""
        self.total += 1
        self.type_counts[entry.log_type] += 1
        
        # Only track message frequency for Errors to find hotspots
        if entry.log_type.upper() == "ERROR":
            self.message_counts[entry.message] += 1

        # Track time range
        if self.min_date is None or entry.timestamp < self.min_date:
            self.min_date = entry.timestamp
        if self.max_date is None or entry.timestamp > self.max_date:
            self.max_date = entry.timestamp

    def generate_report(self) -> AnalysisReport:
        """Compiles the final statistics."""
        return AnalysisReport(
            total_logs=self.total,
            error_count=self.type_counts.get("ERROR", 0),
            warning_count=self.type_counts.get("WARNING", 0),
            start_time=self.min_date,
            end_time=self.max_date,
            top_errors=self.message_counts.most_common(5)
        )

# ==========================================
# 5. Export Strategies (Output Layer)
# ==========================================

class IExporter(ABC):
    """Interface for all export strategies."""
    @abstractmethod
    def export(self, data: List[LogEntry], stats: AnalysisReport, filename: str):
        pass

class CSVExporter(IExporter):
    """Exports raw data to CSV."""
    def export(self, data: List[LogEntry], stats: AnalysisReport, filename: str):
        logger.info(f"Writing CSV to {filename}...")
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Type", "Message"])
            for entry in data:
                writer.writerow([entry.raw_date, entry.log_type, entry.message])

class JSONExporter(IExporter):
    """Exports data and stats to JSON."""
    def export(self, data: List[LogEntry], stats: AnalysisReport, filename: str):
        logger.info(f"Writing JSON to {filename}...")
        output = {
            "meta_data": asdict(stats),
            "logs": [entry.to_dict() for entry in data]
        }
        # Handle datetime serialization for JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, default=str)

class HTMLExporter(IExporter):
    """
    Exports a beautiful HTML dashboard.
    This demonstrates 'Presentation Layer' logic in backend code.
    """
    def export(self, data: List[LogEntry], stats: AnalysisReport, filename: str):
        logger.info(f"Generating HTML Dashboard at {filename}...")
        
        html_template = f"""
        <html>
        <head>
            <title>Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f4f4f9; padding: 20px; }}
                .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .stat-box {{ display: inline-block; margin-right: 20px; padding: 10px; background: #e0e0e0; border-radius: 5px; }}
                .error {{ color: red; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>📊 System Health Report</h1>
            
            <div class="card">
                <h2>Summary</h2>
                <div class="stat-box">Total Logs: <strong>{stats.total_logs}</strong></div>
                <div class="stat-box">Errors: <strong class="error">{stats.error_count}</strong></div>
                <div class="stat-box">Warnings: <strong>{stats.warning_count}</strong></div>
                <p>Time Range: {stats.start_time} to {stats.end_time}</p>
            </div>

            <div class="card">
                <h2>⚠️ Top 5 Recurring Errors</h2>
                <ul>
                    {''.join([f"<li>{msg}: {count} times</li>" for msg, count in stats.top_errors])}
                </ul>
            </div>

            <div class="card">
                <h2>Recent Logs (Last 100)</h2>
                <table>
                    <tr><th>Time</th><th>Type</th><th>Message</th></tr>
                    {''.join([f"<tr><td>{log.raw_date}</td><td>{log.log_type}</td><td>{log.message}</td></tr>" for log in data[-100:]])}
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)

# ==========================================
# 6. Pipeline Orchestrator (Facade)
# ==========================================

class LogPipeline:
    """
    Connects all components: 
    Finder -> Parser -> Analytics -> Exporters
    """
    def __init__(self, filename: str, output_base: str, filter_type: str = None):
        self.filename = filename
        self.output_base = output_base
        self.filter_type = filter_type.upper() if filter_type else None
        
        # Components
        self.navigator = FileNavigator()
        self.analytics = AnalyticsEngine()
        self.exporters: List[IExporter] = [
            CSVExporter(),
            JSONExporter(),
            HTMLExporter()
        ]

    def run(self):
        """Executes the full pipeline."""
        try:
            # 1. Find File
            path = self.navigator.find_file(self.filename)
            parser = LogParser(path)
            
            # 2. Process Stream
            processed_logs = []
            
            print(f"\n--- Starting Processing for {self.filename} ---")
            for entry in parser.parse_stream():
                # Apply Filter (Business Logic)
                if self.filter_type and entry.log_type.upper() != self.filter_type:
                    continue
                
                # Analyze
                self.analytics.process_entry(entry)
                
                # Store (For export)
                # Note: In huge big data apps, we wouldn't store all in memory,
                # but for this scale, list buffering is acceptable.
                processed_logs.append(entry)
                
            # 3. Generate Stats
            report = self.analytics.generate_report()
            self._print_summary(report)
            
            # 4. Export Data
            if processed_logs:
                for exporter in self.exporters:
                    ext = ""
                    if isinstance(exporter, CSVExporter): ext = ".csv"
                    elif isinstance(exporter, JSONExporter): ext = ".json"
                    elif isinstance(exporter, HTMLExporter): ext = ".html"
                    
                    exporter.export(processed_logs, report, self.output_base + ext)
            else:
                logger.warning("No logs found matching criteria. Skipping export.")

        except LogFileNotFoundError:
            logger.critical("Pipeline aborted: Source file missing.")
        except KeyboardInterrupt:
            logger.warning("Pipeline stopped by user.")
        except Exception as e:
            logger.critical(f"Unexpected Fatal Error: {e}", exc_info=True)

    def _print_summary(self, report: AnalysisReport):
        """Prints a quick summary to the console."""
        print("\n" + "="*40)
        print("      PIPELINE COMPLETE      ")
        print("="*40)
        print(f"Total Processed: {report.total_logs}")
        print(f"Errors Found:    {report.error_count}")
        print(f"Warnings Found:  {report.warning_count}")
        print("="*40 + "\n")

# ==========================================
# 7. Entry Point (CLI)
# ==========================================

if __name__ == "__main__":
    # Argument Parsing for robust CLI usage
    parser = argparse.ArgumentParser(
        description="Enterprise Log Analysis Tool V2.0",
        epilog="Example: python main.py HealthApp_2k.log --filter ERROR"
    )
    
    parser.add_argument("filename", help="Name of the log file to search for")
    parser.add_argument(
        "--output", 
        default="final_report", 
        help="Base name for output files (without extension)"
    )
    parser.add_argument(
        "--filter", 
        help="Only process logs of this type (e.g., ERROR, INFO)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Initialize and Run
    pipeline = LogPipeline(
        filename=args.filename,
        output_base=args.output,
        filter_type=args.filter
    )
    
    pipeline.run()
