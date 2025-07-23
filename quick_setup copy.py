#!/usr/bin/env python3
"""
ProPulse Group - Quick Configuration Setup
Interactive setup for Fiverr automation
"""

import json
import os
from pathlib import Path
from getpass import getpass

def create_config_structure():
    """Create the basic configuration structure"""
    return {
        "fiverr": {
            "username": "",
            "email": "",
            "auto_respond": True,
            "response_delay": 300,  # 5 minutes
            "working_hours": {
                "start": "09:00",
                "end": "18:00",
                "timezone": "UTC"
            }
        },
        "automation": {
            "check_interval": 60,  # seconds
            "max_daily_responses": 50,
            "auto_accept_orders": False,
            "demo_mode": True
        },
        "ai": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.7
        },
        "dashboard": {
            "port": 8080,
            "host": "localhost",
            "refresh_rate": 30
        },
        "notifications": {
            "email_alerts": True,
            "webhook_url": "",
            "slack_channel": ""
        }
    }

def setup_config():
    """Interactive configuration setup"""
    print("🔧 ProPulse Group - Quick Setup")
    print("=" * 40)
    
    config = create_config_structure()
    
    # Fiverr credentials
    print("\n📝 Fiverr Account Setup:")
    config["fiverr"]["username"] = input("Fiverr Username: ").strip()
    config["fiverr"]["email"] = input("Fiverr Email: ").strip()
    
    # Automation preferences
    print("\n🤖 Automation Preferences:")
    auto_respond = input("Enable auto-respond? (y/n): ").lower().startswith('y')
    config["automation"]["auto_respond"] = auto_respond
    
    if auto_respond:
        delay = input("Response delay in minutes (default: 5): ").strip()
        if delay.isdigit():
            config["automation"]["response_delay"] = int(delay) * 60
    
    # Working hours
    print("\n⏰ Working Hours:")
    start_time = input("Start time (HH:MM, default: 09:00): ").strip()
    if start_time:
        config["fiverr"]["working_hours"]["start"] = start_time
        
    end_time = input("End time (HH:MM, default: 18:00): ").strip()
    if end_time:
        config["fiverr"]["working_hours"]["end"] = end_time
    
    # Save configuration
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "fiverr_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to {config_path}")
    
    # Create environment template
    create_env_template()
    
    return config

def create_env_template():
    """Create .env template file"""
    env_content = """# ProPulse Group Environment Variables
# Copy this file to .env and fill in your actual values

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Shopify Integration (Optional)
SHOPIFY_ACCESS_TOKEN=your_shopify_access_token_here
SHOPIFY_SHOP_URL=your_shop_url_here

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Database (Optional - uses SQLite by default)
DATABASE_URL=sqlite:///propulse.db

# Webhook URLs (Optional)
SLACK_WEBHOOK_URL=your_slack_webhook_url
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
"""
    
    with open(".env.template", 'w') as f:
        f.write(env_content)
    
    print("📄 Environment template created: .env.template")
    print("   Please copy to .env and add your actual API keys")

if __name__ == "__main__":
    setup_config()