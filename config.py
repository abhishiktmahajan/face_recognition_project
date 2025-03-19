import json
import os

# Default configuration
default_config = {
    "device": "cuda",  # or "cpu"
    "detection": {
        "keep_all": True,
        "confidence_threshold": 0.7
    },
    "recognition": {
        "distance_threshold": 1.0,
        "model": "vggface2"  # or "casia-webface"
    },
    "display": {
        "show_distance": True,
        "show_confidence": True,
        "font_size": 20,
        "box_thickness": 2
    },
    "paths": {
        "known_faces_dir": "known_faces",
        "test_images_dir": "test_images",
        "results_dir": "results"
    }
}

def load_config(config_path="config.json"):
    """Load configuration from file or create default if not exists"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Update with any missing defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
        return config
    else:
        # Create default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

def save_config(config, config_path="config.json"):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Configuration saved to {config_path}")