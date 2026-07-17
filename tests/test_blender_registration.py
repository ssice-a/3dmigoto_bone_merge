from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


class _FakePropertyGroup:
    pass


class _FakeOperator:
    pass


class _FakePanel:
    pass


class _FakeUIList:
    pass


class _FakeScene:
    pass


class _FakeObject:
    pass


class _FakeCollection:
    pass


class BlenderRegistrationTests(unittest.TestCase):
    def test_addon_registers_lod_profile_runtime_classes(self):
        module_names = [
            PACKAGE_DIR.name,
            f"{PACKAGE_DIR.name}.operators",
            f"{PACKAGE_DIR.name}.panel",
            f"{PACKAGE_DIR.name}.properties",
        ]
        saved_modules = {name: sys.modules.get(name) for name in ["bpy", *module_names]}
        registered = []
        unregistered = []
        try:
            fake_bpy = types.ModuleType("bpy")
            fake_bpy.types = types.SimpleNamespace(
                PropertyGroup=_FakePropertyGroup,
                Operator=_FakeOperator,
                Panel=_FakePanel,
                UIList=_FakeUIList,
                Scene=_FakeScene,
                Object=_FakeObject,
                Collection=_FakeCollection,
            )
            property_factory = lambda **_kwargs: None
            fake_bpy.props = types.SimpleNamespace(
                BoolProperty=property_factory,
                CollectionProperty=property_factory,
                EnumProperty=property_factory,
                FloatProperty=property_factory,
                IntProperty=property_factory,
                PointerProperty=property_factory,
                StringProperty=property_factory,
            )
            fake_bpy.path = types.SimpleNamespace(abspath=os.path.abspath)
            fake_bpy.utils = types.SimpleNamespace(
                register_class=registered.append,
                unregister_class=unregistered.append,
            )
            sys.modules["bpy"] = fake_bpy
            for name in module_names:
                sys.modules.pop(name, None)

            addon = importlib.import_module(PACKAGE_DIR.name)
            addon.register()

            registered_names = {item.__name__ for item in registered}
            self.assertIn("BMC_LodProfileItem", registered_names)
            self.assertIn("BMC_UL_lod_profiles", registered_names)
            self.assertIn("BMC_OT_lod_profile_add", registered_names)
            self.assertIn("BMC_OT_sync_lod_profiles", registered_names)
            self.assertTrue(hasattr(fake_bpy.types.Scene, "bmc_lod_profiles"))

            addon.unregister()
            self.assertEqual(len(registered), len(unregistered))
            self.assertFalse(hasattr(fake_bpy.types.Scene, "bmc_lod_profiles"))
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
