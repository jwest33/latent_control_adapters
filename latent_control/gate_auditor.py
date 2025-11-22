"""
Gate Auditing and Monitoring

Provides comprehensive logging, monitoring, and detection for control token gates.

Features:
- Real-time activation logging
- Unexpected token usage detection
- Statistical analysis of gate usage
- Audit trail export

All gate activations are logged for transparency and security analysis.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class GateAuditor:
    """
    Audits and monitors control token gate activations.

    Provides transparency by logging all gate usage and detecting anomalies.
    """

    def __init__(self, log_path: str, log_prompts: bool = False, log_responses: bool = False):
        """
        Initialize gate auditor.

        Args:
            log_path: Path to audit log file (.jsonl format)
            log_prompts: If True, log full prompts (privacy consideration)
            log_responses: If True, log full responses (privacy consideration)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_prompts = log_prompts
        self.log_responses = log_responses

        # In-memory cache of recent logs
        self.recent_logs = []
        self.max_recent = 1000

    def log_activation(
        self,
        gate_name: str,
        token: str,
        action: str,
        metadata: Optional[dict] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ):
        """
        Log a gate activation event.

        Args:
            gate_name: Name of activated gate
            token: Control token used
            action: Action type (e.g., "generation", "enabled", "disabled")
            metadata: Additional metadata dict
            prompt: Full prompt (logged only if log_prompts=True)
            response: Full response (logged only if log_responses=True)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gate_name": gate_name,
            "token": token,
            "action": action,
            "metadata": metadata or {},
        }

        # Optionally include prompt/response
        if self.log_prompts and prompt:
            log_entry["prompt"] = prompt
        elif prompt:
            # Log only a hash or preview for privacy
            log_entry["prompt_hash"] = hash(prompt) % (10**8)
            log_entry["prompt_preview"] = prompt[:50] if len(prompt) > 50 else prompt

        if self.log_responses and response:
            log_entry["response"] = response
        elif response:
            log_entry["response_hash"] = hash(response) % (10**8)
            log_entry["response_preview"] = response[:100] if len(response) > 100 else response

        # Append to log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Add to recent logs cache
        self.recent_logs.append(log_entry)
        if len(self.recent_logs) > self.max_recent:
            self.recent_logs.pop(0)

    def detect_unexpected_token(
        self,
        prompt: str,
        detected_tokens: List[str],
        authorized_tokens: List[str],
    ) -> List[str]:
        """
        Detect unexpected control tokens in prompts.

        Args:
            prompt: User prompt to check
            detected_tokens: Tokens detected in prompt
            authorized_tokens: Tokens user is authorized to use

        Returns:
            List of unauthorized token names
        """
        unauthorized = [t for t in detected_tokens if t not in authorized_tokens]

        if unauthorized:
            self.log_activation(
                gate_name="security_alert",
                token=",".join(unauthorized),
                action="unauthorized_token_detected",
                metadata={
                    "detected": detected_tokens,
                    "authorized": authorized_tokens,
                    "unauthorized": unauthorized,
                },
                prompt=prompt,
            )

        return unauthorized

    def load_audit_log(self, max_entries: Optional[int] = None) -> List[dict]:
        """
        Load audit log from file.

        Args:
            max_entries: Maximum entries to load (most recent)

        Returns:
            List of log entry dicts
        """
        if not self.log_path.exists():
            return []

        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        # Return most recent if limit specified
        if max_entries:
            return logs[-max_entries:]

        return logs

    def get_usage_statistics(self, time_window_hours: Optional[int] = None) -> Dict:
        """
        Get usage statistics for gates.

        Args:
            time_window_hours: Optional time window in hours (None = all time)

        Returns:
            Statistics dictionary
        """
        logs = self.load_audit_log()

        # Filter by time window if specified
        if time_window_hours:
            cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
            logs = [
                log
                for log in logs
                if datetime.fromisoformat(log["timestamp"]).timestamp() >= cutoff
            ]

        # Compute statistics
        total_activations = len(logs)
        gate_counts = defaultdict(int)
        action_counts = defaultdict(int)
        hourly_usage = defaultdict(int)

        for log in logs:
            gate_counts[log["gate_name"]] += 1
            action_counts[log["action"]] += 1

            # Hourly usage
            hour = datetime.fromisoformat(log["timestamp"]).replace(
                minute=0, second=0, microsecond=0
            )
            hourly_usage[hour.isoformat()] += 1

        stats = {
            "total_activations": total_activations,
            "unique_gates": len(gate_counts),
            "gate_usage": dict(gate_counts),
            "action_distribution": dict(action_counts),
            "hourly_usage": dict(hourly_usage),
            "time_window_hours": time_window_hours,
        }

        return stats

    def detect_anomalies(self, threshold_stddev: float = 3.0) -> List[Dict]:
        """
        Detect anomalous gate usage patterns.

        Args:
            threshold_stddev: Number of standard deviations for anomaly threshold

        Returns:
            List of anomaly dicts
        """
        logs = self.load_audit_log()

        if len(logs) < 10:
            return []  # Not enough data

        # Analyze usage frequency per gate
        gate_counts = defaultdict(int)
        for log in logs:
            gate_counts[log["gate_name"]] += 1

        # Compute mean and std
        counts = list(gate_counts.values())
        mean_count = sum(counts) / len(counts)
        std_count = (sum((x - mean_count) ** 2 for x in counts) / len(counts)) ** 0.5

        # Detect outliers
        anomalies = []
        for gate, count in gate_counts.items():
            z_score = (count - mean_count) / std_count if std_count > 0 else 0

            if abs(z_score) > threshold_stddev:
                anomalies.append({
                    "gate_name": gate,
                    "count": count,
                    "mean": mean_count,
                    "std": std_count,
                    "z_score": z_score,
                    "anomaly_type": "high_usage" if z_score > 0 else "low_usage",
                })

        return anomalies

    def export_audit_trail(
        self,
        output_path: str,
        format: str = "json",
        time_window_hours: Optional[int] = None,
    ):
        """
        Export audit trail for analysis or compliance.

        Args:
            output_path: Path to export file
            format: Export format ("json", "csv", "xlsx")
            time_window_hours: Optional time window in hours
        """
        logs = self.load_audit_log()

        # Filter by time window
        if time_window_hours:
            cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
            logs = [
                log
                for log in logs
                if datetime.fromisoformat(log["timestamp"]).timestamp() >= cutoff
            ]

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

        elif format == "csv":
            # Flatten logs for CSV
            df = pd.DataFrame(logs)
            df.to_csv(output_path, index=False)

        elif format == "xlsx":
            df = pd.DataFrame(logs)
            df.to_excel(output_path, index=False, engine="openpyxl")

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Audit trail exported to {output_path} ({len(logs)} entries)")

    def generate_report(self, output_path: str, time_window_hours: Optional[int] = 24):
        """
        Generate comprehensive audit report.

        Args:
            output_path: Path to save report
            time_window_hours: Time window for report (default: 24 hours)
        """
        stats = self.get_usage_statistics(time_window_hours)
        anomalies = self.detect_anomalies()

        report = {
            "generated_at": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "summary": {
                "total_activations": stats["total_activations"],
                "unique_gates_used": stats["unique_gates"],
            },
            "gate_usage": stats["gate_usage"],
            "action_distribution": stats["action_distribution"],
            "anomalies": anomalies,
            "hourly_usage": stats["hourly_usage"],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Audit report saved to {output_path}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("AUDIT REPORT SUMMARY")
        print(f"{'=' * 60}")
        print(f"Time window: {time_window_hours} hours")
        print(f"Total activations: {stats['total_activations']}")
        print(f"Unique gates: {stats['unique_gates']}")
        print(f"Anomalies detected: {len(anomalies)}")
        print(f"{'=' * 60}\n")

    def clear_log(self, backup_path: Optional[str] = None):
        """
        Clear audit log (with optional backup).

        Args:
            backup_path: Optional path to backup log before clearing
        """
        if backup_path and self.log_path.exists():
            import shutil
            shutil.copy(self.log_path, backup_path)
            print(f"Audit log backed up to {backup_path}")

        # Clear log file
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

        self.recent_logs.clear()
        print("Audit log cleared")
