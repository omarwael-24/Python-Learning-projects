import os
import json
import csv

def require_search(func):
    """Decorator to ensure the file path is found before any operation."""
    def wrapper(self, *args, **kwargs):
        if not self.found_path:
            print(f"--- [Status] Path missing. Searching for: {self.file_name} ---")
            self.search_file()
        if not self.found_path:
            return "--- [Error] File could not be located on this system. ---"
        return func(self, *args, **kwargs)
    return wrapper

class LogAnalytics:
    def __init__(self, file_name):
        self.file_name = file_name
        self.found_path = ""

    def search_file(self):
        """Walks through the home directory to find the file."""
        search_root = os.path.expanduser('~')
        for root, _, files in os.walk(search_root):
            if self.file_name in files:
                self.found_path = os.path.join(root, self.file_name)
                print(f"--- [Success] File found at: {self.found_path} ---")
                return self.found_path
        return None

    @require_search
    def stream_parsed_data(self):
        """
        A Generator function that yields one log entry at a time.
        This is memory-efficient for massive log files.
        """
        with open(self.found_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    yield {
                        "date": parts[0].strip(),
                        "type": parts[1].strip(),
                        "message": parts[2].strip()
                    }

    def export_to_csv(self, output_name="log_report.csv"):
        """Exports streamed data to a CSV file."""
        print(f"--- [Action] Exporting to CSV: {output_name} ---")
        fieldnames = ["date", "type", "message"]
        with open(output_name, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.stream_parsed_data():
                writer.writerow(entry)
        print("--- [Done] CSV Export Complete. ---")

    def export_to_json(self, output_name="log_report.json"):
        """Exports all data to a JSON file."""
        print(f"--- [Action] Exporting to JSON: {output_name} ---")
        # Note: Converting generator to list for standard JSON dump
        data = list(self.stream_parsed_data())
        with open(output_name, "w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=4, ensure_ascii=False)
        print("--- [Done] JSON Export Complete. ---")

# --- Execution ---
if __name__ == "__main__":
    TARGET_FILE = "HealthApp_2k.log"
    
    # Initialize the engine
    app = LogAnalytics(TARGET_FILE)
    
    # Run exports
    app.export_to_csv("health_data.csv")
    app.export_to_json("health_data.json")