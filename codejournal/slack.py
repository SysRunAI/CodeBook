
import requests
from datetime import datetime

def send_slack_message(webhook_url, message, level="info"):
    """Send a formatted Slack message based on the level."""
    # Define colors and prefixes for different levels
    if webhook_url is None:
        return
    level_settings = {
        "info": {"color": "#36a64f", "prefix": ":information_source: INFO"},
        "warn": {"color": "#ffcc00", "prefix": ":warning: WARNING"},
        "priority": {"color": "#ff0000", "prefix": ":rotating_light: PRIORITY"},
    }

    settings = level_settings.get(level, level_settings["info"])
    prefix = settings["prefix"]
    color = settings["color"]

    # Prepare the message payload
    payload = {
        "attachments": [
            {
                "color": color,
                "fields": [
                    {"title": prefix, "value": message, "short": False},
                ],
                "footer": f"Notification sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            }
        ]
    }

    # Send the message to Slack
    response = requests.post(webhook_url, json=payload)

    if response.status_code != 200:
        print(f"Failed to send message. Status code: {response.status_code}, Error: {response.text}")

def notify_info(webhook_url, message):
    send_slack_message(webhook_url, message, level="info")

def notify_warn(webhook_url, message):
    send_slack_message(webhook_url, message, level="warn")

def notify_priority(webhook_url, message):
    send_slack_message(webhook_url, message, level="priority")


__all__ = ["notify_info", "notify_warn", "notify_priority"]
