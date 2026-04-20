"""Alert rule engine for security events."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Security rule definition."""
    name: str
    enabled: bool
    condition: Callable[[Dict], bool]
    event_type: str
    severity: str
    description_template: str


@dataclass
class TriggeredAlert:
    """Result of rule evaluation."""
    event_id: str
    rule_name: str
    event_type: str
    severity: str
    description: str
    frame_ids: List[str]
    timestamp: datetime


class RuleEngine:
    """Evaluate security rules against detections."""
    
    def __init__(self, rules_config: dict, sqlite_store=None):
        """Initialize rule engine.
        
        Args:
            rules_config: Configuration dictionary of security rules
            sqlite_store: Optional SQLiteStore instance for context queries
        """
        self.rules_config = rules_config
        self.sqlite_store = sqlite_store
        self.rules: Dict[str, Rule] = {}
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from configuration."""
        if not self.rules_config:
            logger.warning("No rules configuration provided")
            return
        
        rules_dict = self.rules_config.get('rules', {})
        
        for rule_name, rule_config in rules_dict.items():
            if rule_config.get('enabled', True):
                rule = self._parse_rule(rule_name, rule_config)
                if rule:
                    self.rules[rule_name] = rule
                    logger.info(f"Loaded rule: {rule_name}")
    
    def _parse_rule(self, rule_name: str, rule_config: dict) -> Optional[Rule]:
        """Parse a rule configuration into executable form.
        
        Args:
            rule_name: Name of the rule
            rule_config: Rule configuration dict
            
        Returns:
            Rule object or None if invalid
        """
        try:
            event_type = rule_config.get('type', 'unknown')
            severity = rule_config.get('severity', 'medium')
            description = rule_config.get('description', rule_name)
            
            condition = self._build_condition(rule_name, rule_config)
            
            return Rule(
                name=rule_name,
                enabled=rule_config.get('enabled', True),
                condition=condition,
                event_type=event_type,
                severity=severity,
                description_template=description
            )
        except Exception as e:
            logger.error(f"Failed to parse rule {rule_name}: {e}")
            return None
    
    def _build_condition(self, rule_name: str, config: dict) -> Callable:
        """Build a condition function from configuration.
        
        Args:
            rule_name: Rule name for logging
            config: Rule configuration
            
        Returns:
            Callable that evaluates the condition
        """
        # Common rule types
        if 'person_detection' in rule_name:
            threshold = config.get('confidence_threshold', 0.7)
            return lambda data: self._check_person_detection(data, threshold)
        
        elif 'crowd' in rule_name.lower():
            count_threshold = config.get('person_count_threshold', 5)
            return lambda data: self._check_crowd(data, count_threshold)
        
        elif 'unauthorized' in rule_name.lower() or 'restricted_zone' in rule_name:
            zones = config.get('zones', [])
            return lambda data: self._check_restricted_zone(data, zones)
        
        elif 'vehicle' in rule_name.lower():
            classes = config.get('vehicle_classes', ['car', 'truck', 'bus'])
            return lambda data: self._check_vehicle_detection(data, classes)
        
        else:
            # Default: always true (alert on any event of this type)
            return lambda data: True
    
    def _check_person_detection(self, frame_data: dict, confidence_threshold: float) -> bool:
        """Check if people were detected above threshold.
        
        Args:
            frame_data: Frame detection data
            confidence_threshold: Minimum confidence score
            
        Returns:
            True if people detected above threshold
        """
        detections = frame_data.get('detections', [])
        for det in detections:
            if det.get('class') == 'person' and det.get('confidence', 0) >= confidence_threshold:
                return True
        return False
    
    def _check_crowd(self, frame_data: dict, person_count_threshold: int) -> bool:
        """Check if crowd detected.
        
        Args:
            frame_data: Frame detection data
            person_count_threshold: Minimum people count
            
        Returns:
            True if crowd detected
        """
        detections = frame_data.get('detections', [])
        person_count = sum(1 for d in detections if d.get('class') == 'person')
        return person_count >= person_count_threshold
    
    def _check_restricted_zone(self, frame_data: dict, restricted_zones: List[str]) -> bool:
        """Check if objects detected in restricted zones.
        
        Args:
            frame_data: Frame detection data
            restricted_zones: List of zone names
            
        Returns:
            True if detection in restricted zone
        """
        frame_zone = frame_data.get('zone', '')
        if frame_zone in restricted_zones:
            detections = frame_data.get('detections', [])
            # Check if any unauthorized objects are present
            unauthorized_classes = ['person', 'vehicle', 'drone']
            for det in detections:
                if det.get('class') in unauthorized_classes:
                    return True
        return False
    
    def _check_vehicle_detection(self, frame_data: dict, vehicle_classes: List[str]) -> bool:
        """Check if vehicles detected.
        
        Args:
            frame_data: Frame detection data
            vehicle_classes: List of vehicle class names
            
        Returns:
            True if vehicle detected
        """
        detections = frame_data.get('detections', [])
        for det in detections:
            if det.get('class') in vehicle_classes:
                return True
        return False
    
    def evaluate(self, frame_data: dict, frame_id: str) -> List[TriggeredAlert]:
        """Evaluate all rules against frame data.
        
        Args:
            frame_data: Frame detection data including detections, zone, caption, etc.
            frame_id: Frame identifier
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                if rule.condition(frame_data):
                    alert = self._create_alert(
                        rule=rule,
                        frame_data=frame_data,
                        frame_id=frame_id
                    )
                    alerts.append(alert)
                    logger.info(f"Rule triggered: {rule_name} -> {alert.event_type} ({alert.severity})")
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return alerts
    
    def _create_alert(self, rule: Rule, frame_data: dict, frame_id: str) -> TriggeredAlert:
        """Create an alert from a triggered rule.
        
        Args:
            rule: Triggered rule
            frame_data: Frame data for context
            frame_id: Associated frame ID
            
        Returns:
            TriggeredAlert object
        """
        event_id = str(uuid.uuid4())
        caption = frame_data.get('caption', '')
        detections = frame_data.get('detections', [])
        
        # Build description
        description = rule.description_template
        if detections:
            classes = [d.get('class', 'unknown') for d in detections]
            unique_classes = list(set(classes))
            description += f" - Detected: {', '.join(unique_classes)}"
        
        return TriggeredAlert(
            event_id=event_id,
            rule_name=rule.name,
            event_type=rule.event_type,
            severity=rule.severity,
            description=description,
            frame_ids=[frame_id],
            timestamp=datetime.now(timezone.utc)
        )
    
    def add_rule(self, rule_name: str, rule_config: dict) -> bool:
        """Add a new rule dynamically.
        
        Args:
            rule_name: Name of the rule
            rule_config: Rule configuration dict
            
        Returns:
            True if rule added successfully
        """
        try:
            rule = self._parse_rule(rule_name, rule_config)
            if rule:
                self.rules[rule_name] = rule
                logger.info(f"Added rule: {rule_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add rule {rule_name}: {e}")
            return False
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule removed
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rule: {rule_name}")
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            True if enabled
        """
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled rule: {rule_name}")
            return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            True if disabled
        """
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled rule: {rule_name}")
            return True
        return False
    
    def get_active_rules(self) -> List[str]:
        """Get list of active rule names.
        
        Returns:
            List of enabled rule names
        """
        return [name for name, rule in self.rules.items() if rule.enabled]
    
    def correlate_events(self, alerts: List[TriggeredAlert], time_window_seconds: int = 60) -> List[TriggeredAlert]:
        """Correlate multiple alerts into single events.
        
        Args:
            alerts: List of triggered alerts
            time_window_seconds: Time window for correlation
            
        Returns:
            Correlated alerts (deduplicated/grouped)
        """
        if not alerts:
            return []
        
        # Group by (event_type, severity)
        groups: Dict[tuple, List[TriggeredAlert]] = {}
        for alert in alerts:
            key = (alert.event_type, alert.severity)
            if key not in groups:
                groups[key] = []
            groups[key].append(alert)
        
        # Return grouped alerts (simplified - could be more sophisticated)
        correlated = []
        for group in groups.values():
            # Merge frame IDs from all alerts in group
            merged_alert = group[0]
            all_frame_ids = []
            for alert in group:
                all_frame_ids.extend(alert.frame_ids)
            merged_alert.frame_ids = list(set(all_frame_ids))
            correlated.append(merged_alert)
        
        return correlated
