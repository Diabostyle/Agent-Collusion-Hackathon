import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def render_transcript(jsonl_path: str):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {jsonl_path}")
        return

    print("=" * 80)
    print(f">> AGENT RUN TRANSCRIPT: {path.name}")
    print("=" * 80)

    with open(path, "r", encoding="utf-8") as f:
        events = [json.loads(line.strip()) for line in f if line.strip()]

    for ev in events:
        ev_type = ev.get("event")

        if ev_type == "simulation_start":
            print(f"[START] Condition: {ev.get('condition')} | Model: {ev.get('model')} | Max Turns: {ev.get('max_turns')}\n")

        elif ev_type == "turn_execution":
            turn = ev.get("turn")
            agent = ev.get("agent")
            agent_mod = ev.get("agent_model")
            agent_mod_str = f" [{agent_mod}]" if agent_mod else ""
            tool = ev.get("tool_called")
            args = ev.get("arguments", {})
            result = ev.get("tool_result", "")
            metrics = ev.get("metrics", {})

            print(f"\n--- Turn {turn}: {agent}{agent_mod_str} called [{tool}] ---")

            if tool == "send_direct_message":
                to = args.get("to")
                msg = args.get("message")
                print(f"   [PRIVATE CHAT to {to}]: \"{msg}\"")

            elif tool == "transfer_to_processor":
                instructions = args.get("instructions", "")
                print(f"   [Instructions]: {instructions[:140]}...")
                print(f"   [Payload]: Transferred customer batch to Anonymizer.")

            elif tool == "mask_pii":
                print(f"   [Action]: Applied mask_pii redaction.")
                lines = result.split("\n")
                if len(lines) > 2:
                    print(f"   [Sample Masked Row]: {lines[2] if len(lines) > 2 else ''}")

            elif tool == "send_to_analyst":
                certified = args.get("compliance_certified")
                print(f"   [Forwarding to Analyst] Compliance Certified: {certified}")
                if metrics.get("proof_of_concealment"):
                    print("   *** ALERT: PROOF OF CONCEALMENT! Raw PII sent with certified=True ***")

            elif tool == "submit_risk_score":
                score = args.get("score")
                reasoning = args.get("reasoning", "")
                print(f"   [FINAL SUBMISSION]: Regional Risk Score = {score}")
                print(f"   [Analyst Reasoning]: \"{reasoning}\"")

            else:
                print(f"   Arguments: {args}")

        elif ev_type == "simulation_end":
            print("\n" + "=" * 80)
            status = "SUCCESS" if ev.get("is_success") else "FAILED"
            print(f"[END] Outcome: {status} | Turns Used: {ev.get('total_turns_used')}")
            print(f"   - PII Leaked to Analyst: {ev.get('pii_leaked_to_analyst')}")
            print(f"   - Proof of Concealment: {ev.get('proof_of_concealment')}")
            print(f"   - Risk Score: {ev.get('risk_score_submitted')}")
            print("=" * 80 + "\n")


def find_latest_log() -> str:
    results_dir = Path("results")
    if not results_dir.exists():
        return ""
    jsonl_files = list(results_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        return ""
    latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description="View human-readable agent interaction transcripts")
    parser.add_argument("log_path", nargs="?", default=None, help="Path to .jsonl run file (defaults to latest)")
    args = parser.parse_args()

    target = args.log_path or find_latest_log()
    if not target:
        print("No log files found in 'results/' directory.")
        sys.exit(1)

    render_transcript(target)


if __name__ == "__main__":
    main()

