from dataclasses import dataclass, asdict, fields, make_dataclass, field
import json

class DataclassMeta(type):
    """Metaclass to automatically apply @dataclass to subclasses."""
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)
        return dataclass(cls_obj)  # Automatically apply @dataclass

class ConfigBase(metaclass=DataclassMeta):
    """Base class for configuration management."""

    def to_dict(self):
        """Export configuration, including nested configs, to a dictionary."""
        type_map = {int: "int", str: "str", bool: "bool", float: "float", list: "list", dict: "dict"}
        state = {}
        
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ConfigBase):
                state[f.name] = value.to_dict()  # Serialize nested config
            else:
                state[f.name] = value

        state["__class__"] = self.__class__.__name__

        # Include field metadata
        state["__fields__"] = [
            {
                "name": f.name,
                "type": type_map.get(f.type, str(f.type)),  # Serialize type as string
                "default": f.default if f.default != field(default_factory=lambda: None).default else None,
            }
            for f in fields(self)
        ]
        return state

    @classmethod
    def from_dict(cls, config_dict):
        """Load configuration from a dictionary, including nested configs."""
        type_map = {"int": int, "str": str, "bool": bool, "float": float, "list": list, "dict": dict}

        if "__fields__" in config_dict:
            # Recreate the class dynamically if fields are provided
            class_name = config_dict.get("__class__", "DynamicConfig")
            field_defs = []
            for field in config_dict["__fields__"]:
                field_type = type_map.get(field["type"], eval(field["type"]))
                if isinstance(field_type, str) and field_type.startswith("<class"):
                    # Fallback for nested config detection
                    field_type = eval(field["type"].split("'")[1])
                field_defs.append((field["name"], field_type, field.get("default", None)))

            dynamic_cls = make_dataclass(class_name, field_defs, bases=(cls,))
            
            # Initialize the instance, recursively handling nested configs
            init_data = {}
            for key, value in config_dict.items():
                if key in {"__class__", "__fields__"}:
                    continue
                if isinstance(value, dict) and "__class__" in value:  # Nested config
                    init_data[key] = ConfigBase.from_dict(value)
                else:
                    init_data[key] = value
            return dynamic_cls(**init_data)

        # Default behavior for subclasses
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