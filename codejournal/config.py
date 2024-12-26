from dataclasses import dataclass, asdict, fields, make_dataclass
import json

class DataclassMeta(type):
    """Metaclass to automatically apply @dataclass to subclasses."""
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)
        return dataclass(cls_obj)  # Automatically apply @dataclass

class ConfigBase(metaclass=DataclassMeta):
    """Base class for configuration management."""

    def to_dict(self):
        """Export configuration to a dictionary."""
        state = asdict(self)
        cls_name = self.__class__.__name__
        state["__class__"] = cls_name
        return state

    @classmethod
    def from_dict(cls, config_dict):
        """Load configuration from a dictionary."""
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)

    def to_json(self, filepath):
        """Export configuration to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load(cls, filepath):
        """Polymorphic deserialization based on a key in the JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        # Check for a class type indicator, e.g., "__class__"
        if "__class__" in config_dict:
            class_name = config_dict["__class__"]
            subclass = next(
                (sub for sub in cls.__subclasses__() if sub.__name__ == class_name), None
            )
            if subclass:
                return subclass.from_dict(config_dict)
        # Default to the current class
        return cls.from_dict(config_dict)

__all__ = ["ConfigBase"]