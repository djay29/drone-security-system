"""Alert dispatcher for sending notifications."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert delivery channels."""
    CONSOLE = "console"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"


class AlertDispatcher:
    """Dispatch alerts through multiple channels."""
    
    def __init__(self, config: dict = None):
        """Initialize dispatcher.
        
        Args:
            config: Configuration for alert channels
        """
        self.config = config or {}
        self.channels: Dict[AlertChannel, Callable] = {}
        self.sqlite_store = None
        self._setup_channels()
    
    def set_sqlite_store(self, sqlite_store) -> None:
        """Set SQLite store for storing alert records.
        
        Args:
            sqlite_store: SQLiteStore instance
        """
        self.sqlite_store = sqlite_store
    
    def _setup_channels(self) -> None:
        """Set up alert channels based on configuration."""
        channels_config = self.config.get('channels', [])
        
        for channel_config in channels_config:
            channel_type = channel_config.get('type', 'console')
            enabled = channel_config.get('enabled', False)
            
            if not enabled:
                continue
            
            if channel_type == 'console':
                self.channels[AlertChannel.CONSOLE] = self._send_console
            elif channel_type == 'email':
                self.channels[AlertChannel.EMAIL] = self._send_email
            elif channel_type == 'webhook':
                self.channels[AlertChannel.WEBHOOK] = self._send_webhook
            elif channel_type == 'slack':
                self.channels[AlertChannel.SLACK] = self._send_slack
    
    def dispatch(self, alert_data: dict, event_id: str = None) -> List[str]:
        """Send alert through configured channels.
        
        Args:
            alert_data: Alert data including event_type, severity, description, frame_ids
            event_id: Optional event ID to associate alert with
            
        Returns:
            List of alert IDs that were sent
        """
        if event_id is None:
            event_id = alert_data.get('event_id', str(uuid.uuid4()))
        
        alert_ids = []
        
        for channel, handler in self.channels.items():
            try:
                alert_id = self._send_to_channel(
                    handler=handler,
                    channel=channel,
                    alert_data=alert_data,
                    event_id=event_id
                )
                alert_ids.append(alert_id)
            except Exception as e:
                logger.error(f"Failed to dispatch to {channel.value}: {e}")
        
        return alert_ids
    
    def _send_to_channel(self, handler: Callable, channel: AlertChannel, 
                        alert_data: dict, event_id: str) -> str:
        """Send alert to specific channel and record it.
        
        Args:
            handler: Channel handler function
            channel: AlertChannel enum
            alert_data: Alert data
            event_id: Event ID
            
        Returns:
            Alert ID
        """
        alert_id = str(uuid.uuid4())
        message = self._format_message(alert_data)
        
        # Send through channel
        handler(alert_data, message)
        
        # Store alert record if store available
        if self.sqlite_store:
            try:
                self.sqlite_store.store_alert(
                    alert_id=alert_id,
                    event_id=event_id,
                    channel=channel.value,
                    message=message,
                    ts=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.error(f"Failed to store alert record: {e}")
        
        logger.info(f"Alert sent via {channel.value}: {alert_id}")
        return alert_id
    
    def _format_message(self, alert_data: dict) -> str:
        """Format alert data into message string.
        
        Args:
            alert_data: Alert data
            
        Returns:
            Formatted message
        """
        severity = alert_data.get('severity', 'unknown').upper()
        event_type = alert_data.get('event_type', 'security_event')
        description = alert_data.get('description', 'Security event detected')
        timestamp = alert_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        frame_count = len(alert_data.get('frame_ids', []))
        
        message = f"""SECURITY ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Severity: [{severity}]
Type: {event_type}
Time: {timestamp}
Description: {description}
Frames: {frame_count}
"""
        return message
    
    def _send_console(self, alert_data: dict, message: str) -> None:
        """Send alert to console/stdout.
        
        Args:
            alert_data: Alert data
            message: Formatted message
        """
        print(message)
        logger.warning(f"ALERT: {alert_data['description']}")
    
    def _send_email(self, alert_data: dict, message: str) -> None:
        """Send alert via email.
        
        Args:
            alert_data: Alert data
            message: Formatted message
        """
        # Email implementation pending
        logger.info(f"Email channel not yet implemented")
        pass
    
    def _send_webhook(self, alert_data: dict, message: str) -> None:
        """Send alert via webhook.
        
        Args:
            alert_data: Alert data
            message: Formatted message
        """
        # Webhook implementation pending
        logger.info(f"Webhook channel not yet implemented")
        pass
    
    def _send_slack(self, alert_data: dict, message: str) -> None:
        """Send alert to Slack.
        
        Args:
            alert_data: Alert data
            message: Formatted message
        """
        # Slack implementation pending
        logger.info(f"Slack channel not yet implemented")
        pass
    
    def add_channel(self, channel_name: str, channel_config: dict) -> bool:
        """Add alert channel dynamically.
        
        Args:
            channel_name: Name of the channel (console, email, webhook, slack)
            channel_config: Channel configuration
            
        Returns:
            True if channel added successfully
        """
        try:
            if channel_name == 'console':
                self.channels[AlertChannel.CONSOLE] = self._send_console
            elif channel_name == 'email':
                self.channels[AlertChannel.EMAIL] = self._send_email
            elif channel_name == 'webhook':
                self.channels[AlertChannel.WEBHOOK] = self._send_webhook
            elif channel_name == 'slack':
                self.channels[AlertChannel.SLACK] = self._send_slack
            else:
                logger.warning(f"Unknown channel type: {channel_name}")
                return False
            
            logger.info(f"Added alert channel: {channel_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add channel {channel_name}: {e}")
            return False
    
    def remove_channel(self, channel_name: str) -> bool:
        """Remove an alert channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            True if channel removed
        """
        try:
            channel = AlertChannel(channel_name)
            if channel in self.channels:
                del self.channels[channel]
                logger.info(f"Removed alert channel: {channel_name}")
                return True
            return False
        except ValueError:
            logger.warning(f"Unknown channel: {channel_name}")
            return False
    
    def get_active_channels(self) -> List[str]:
        """Get list of active channels.
        
        Returns:
            List of active channel names
        """
        return [ch.value for ch in self.channels.keys()]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if acknowledged
        """
        if self.sqlite_store:
            try:
                self.sqlite_store.ack_alert(alert_id)
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to acknowledge alert: {e}")
                return False
        return False
