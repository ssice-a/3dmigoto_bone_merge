"""Blender material helpers for marked texture bindings."""

from __future__ import annotations

import json

from .texture_converter import TextureConversionError, load_image_for_blender


def apply_material_from_texture_bindings(
    mesh_obj,
    texture_bindings: dict[str, dict[str, object]],
    *,
    clear_existing: bool = True,
) -> bool:
    if not texture_bindings or getattr(mesh_obj, "type", "") != "MESH":
        return False

    def binding_for_semantic(semantic: str, fallback_slot: str):
        for binding in texture_bindings.values():
            if isinstance(binding, dict) and str(binding.get("semantic", "") or "") == semantic:
                return binding
        fallback = texture_bindings.get(fallback_slot)
        return fallback if isinstance(fallback, dict) else None

    base_binding = binding_for_semantic("base_color", "ps-t7")
    normal_binding = binding_for_semantic("normal", "ps-t5")
    if base_binding is None and normal_binding is None:
        return False

    import bpy  # Imported lazily so tests can import this module without Blender.

    if clear_existing:
        mesh_obj.data.materials.clear()

    material_name = f"{mesh_obj.name}_Material"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True

    bsdf = _ensure_principled_bsdf_node(material, reset=clear_existing)
    if bsdf is None:
        mesh_obj.data.materials.append(material)
        return True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    if base_binding is not None:
        base_path = str(base_binding.get("source_path", "") or "")
        base_node = _add_image_texture_node(nodes, base_path, str(base_binding.get("slot", "") or "base color"))
        base_input = _bsdf_input(bsdf, "Base Color", "BaseColor")
        if base_node is not None and base_input is not None:
            links.new(base_node.outputs["Color"], base_input)

    if normal_binding is not None:
        normal_path = str(normal_binding.get("source_path", "") or "")
        normal_node = _add_image_texture_node(
            nodes,
            normal_path,
            str(normal_binding.get("slot", "") or "normal"),
            color_space="Non-Color",
        )
        normal_input = _bsdf_input(bsdf, "Normal")
        if normal_node is not None and normal_input is not None:
            normal_map = nodes.new(type="ShaderNodeNormalMap")
            links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], normal_input)

    material["bmc_texture_slots"] = json.dumps(texture_bindings, ensure_ascii=False)
    mesh_obj.data.materials.append(material)
    return True


def _principled_bsdf_node(nodes):
    node = nodes.get("Principled BSDF")
    if node is not None:
        return node
    for candidate in nodes:
        if getattr(candidate, "bl_idname", "") == "ShaderNodeBsdfPrincipled":
            return candidate
    return None


def _ensure_principled_bsdf_node(material, *, reset: bool):
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    if reset:
        nodes.clear()

    bsdf = _principled_bsdf_node(nodes)
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

    output = None
    for candidate in nodes:
        if getattr(candidate, "bl_idname", "") == "ShaderNodeOutputMaterial":
            output = candidate
            break
    if output is None:
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (320, 0)

    if "BSDF" in bsdf.outputs and "Surface" in output.inputs:
        has_link = any(link.from_node == bsdf and link.to_node == output for link in links)
        if not has_link:
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return bsdf


def _bsdf_input(bsdf_node, *names: str):
    for name in names:
        if name in bsdf_node.inputs:
            return bsdf_node.inputs[name]
    for socket in bsdf_node.inputs:
        identifier = str(getattr(socket, "identifier", "") or "")
        if identifier in names:
            return socket
    return None


def _add_image_texture_node(nodes, source_path: str, label: str, *, color_space: str | None = None):
    if not source_path:
        return None
    try:
        image = load_image_for_blender(source_path, color_space=color_space)
    except (FileNotFoundError, RuntimeError, TextureConversionError, OSError):
        return None
    node = nodes.new(type="ShaderNodeTexImage")
    node.label = label
    node.image = image
    return node
