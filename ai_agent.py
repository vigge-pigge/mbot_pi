import json
import requests
import threading
from typing import Dict, List, Any


class CognitiveAgent:
    """Interface for local SLM running via Ollama API (Non-blocking background thread)."""

    def __init__(self, model_name: str = "qwen2.5:1.5b", api_url: str = "http://localhost:11434/api/generate") -> None:
        self.model_name = model_name
        self.api_url = api_url
        self.system_prompt = (
            "You are the cognitive brain of an autonomous mBot Mega robot. "
            "Respond ONLY with a valid JSON object matching this schema:\n"
            '{"action": "TRACK_TARGET" | "SEARCH" | "STOP", "target_label": "string", "reasoning": "string"}\n'
            "Rules: If the robot event TARGET_REACHED is given, that target is completed. "
            "Pick a new target_label from the detections, or answer STOP if the mission is complete."
        )
        self.latest_decision = {
            "action": "SEARCH",
            "target_label": "none",
            "reasoning": "Initializing agent..."
        }
        self.is_querying = False

    def _query_ollama_thread(self, detections: List[Dict[str, Any]], event: str | None = None) -> None:
        event_text = f"\nRobot event: {event}" if event else ""
        prompt = f"System: {self.system_prompt}\nInput Detections: {json.dumps(detections)}{event_text}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=2.0)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "{}")
                self.latest_decision = json.loads(response_text)
        except Exception:
            # Fallback om Ollama inte svarar eller hänger sig
            if detections:
                primary_target = detections[0].get("label", "target")
                self.latest_decision = {
                    "action": "TRACK_TARGET",
                    "target_label": primary_target,
                    "reasoning": "Fallback: Directly tracking first detected target."
                }
            else:
                self.latest_decision = {
                    "action": "SEARCH",
                    "target_label": "none",
                    "reasoning": "Fallback: Searching area for objects."
                }
        finally:
            self.is_querying = False

    def decide_action(self, detections: List[Dict[str, Any]], event: str | None = None) -> Dict[str, Any]:
        """Returnerar senaste beslutet OMEDELBART utan att låsa huvudloopen."""
        if not self.is_querying:
            self.is_querying = True
            thread = threading.Thread(target=self._query_ollama_thread, args=(detections, event), daemon=True)
            thread.start()

        return self.latest_decision